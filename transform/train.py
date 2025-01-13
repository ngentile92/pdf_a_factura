from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3ForTokenClassification
import torch
import json
import os
from PIL import Image
import numpy as np

# 📄 ✅ Load the dataset from the JSON file
with open("../extract/facturas_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ Convert the data into a HuggingFace Dataset
dataset = Dataset.from_list(data)
# 🗺️ Crear un mapa de etiquetas basado en las respuestas únicas del dataset
all_answers = set()
for example in dataset:
    for field in example["fields"]:
        all_answers.add(field["answer"])

# ✅ Split the dataset into train, validation, and test sets
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
validation_test_split = split_dataset["test"].train_test_split(test_size=0.2, seed=42)

#print the sise of the dataset
print(f"Train size: {len(split_dataset['train'])}")
print(f"Validation size: {len(validation_test_split['train'])}")
print(f"Test size: {len(validation_test_split['test'])}")
final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": validation_test_split["train"],
    "test": validation_test_split["train"],
})

# Crear el mapa de etiquetas
label_map = {answer: idx for idx, answer in enumerate(all_answers)}

# ✅ Load the model and processor
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(label_map))
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# 📦 ✅ Preprocessing function
# 📦 ✅ Preprocessing function corregida
def preprocess_data(example):
    image_path = os.path.join("../images", f"{example['file_name']}.png")
    image = Image.open(image_path).convert("RGB")

    questions = [field["question"] for field in example["fields"]]
    answers = [field["answer"] for field in example["fields"]]

    # ✅ Manejar bounding boxes vacíos o inválidos
    boxes = []
    for field in example["fields"]:
        box = field.get("box", [0, 0, 0, 0])  # Rellenar si falta
        if not box or len(box) != 4:
            box = [0, 0, 0, 0]  # Usar box vacío si está incompleto
        boxes.append(box)

    # ✅ Asegurar que la longitud de boxes coincida con la longitud de las preguntas
    while len(boxes) < len(questions):
        boxes.append([0, 0, 0, 0])

    boxes = boxes[:len(questions)]

    # ✅ Encode the image, questions, and bounding boxes
    encoded = processor(
        images=image,
        text=questions,
        boxes=boxes,  # ✅ Pasar los boxes aquí
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # ✅ Convertir respuestas a etiquetas
    labels = torch.tensor([label_map.get(a, -100) for a in answers], dtype=torch.long)
    labels = torch.nn.functional.pad(labels, (0, encoded["input_ids"].size(1) - labels.size(0)), value=-100)

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "bbox": encoded["bbox"].squeeze(0),
        "labels": labels
    }

# ✅ Preprocess the complete dataset
encoded_dataset = final_dataset.map(preprocess_data, remove_columns=["file_name", "fields"])

# ⚙️ ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=5,
    load_best_model_at_end=True,
    learning_rate=5e-5,
    remove_unused_columns=False,
)

# ✅ Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=processor,
)

# 🚀 ✅ Train the model
trainer.train()

# 📊 ✅ Evaluate the model
results = trainer.evaluate(encoded_dataset["test"])
print("\n📊 Model results:", results)


# 📂 ✅ Guardar el modelo y el processor en una carpeta llamada 'trained_model'
save_directory = "./trained_model"
os.makedirs(save_directory, exist_ok=True)

print("\n💾 Guardando el modelo entrenado y el processor...")
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print("✅ Modelo y processor guardados en 'trained_model'")

#### 


from torch.nn.functional import softmax

# 📊 ✅ Realiza las predicciones sobre el conjunto de validación
predictions = trainer.predict(encoded_dataset["validation"])

# ✅ Convertir los logits a probabilidades usando softmax
probs = softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# ✅ Seleccionar la clase con mayor probabilidad
predicted_labels = np.argmax(probs, axis=-1)

# 🧩 ✅ Función corregida para decodificar y mostrar las predicciones
# 🧩 ✅ Función para decodificar y mostrar las predicciones
# 🧩 ✅ Función para decodificar y mostrar las predicciones
def decode_predictions(predicted_labels, probs, dataset, threshold=0.8):
    correct = 0
    total = len(dataset)

    print("\n📋 PREDICCIONES DEL MODELO:")
    for i in range(total):
        file_name = dataset[i]["file_name"]
        fields = dataset[i]["fields"]

        print(f"\n🧾 Factura: {file_name}")
        for j, field in enumerate(fields):
            question = field["question"]
            actual_answer = field["answer"]

            # ✅ Asegurarse de que las predicciones sean un array y manejar casos donde sea un escalar
            if isinstance(predicted_labels[i], np.ndarray):
                pred_label = predicted_labels[i][j] if j < len(predicted_labels[i]) else -100
                confidence = probs[i][j][pred_label] if pred_label != -100 else 0
            else:
                pred_label = predicted_labels[i]
                confidence = probs[i][pred_label] if pred_label != -100 else 0

            # ✅ Obtener la respuesta predicha o manejar el caso donde no haya predicción
            if confidence >= threshold and pred_label < len(fields):
                predicted_answer = fields[pred_label]["answer"]
            else:
                predicted_answer = "No prediction"

            # ✅ Comparar la predicción con la respuesta real
            if predicted_answer == actual_answer:
                result = "✅ Correcto"
                correct += 1
            else:
                result = "❌ Incorrecto"

            print(f"  - {question}")
            print(f"    🏷️ Real: {actual_answer}")
            print(f"    🤖 Predicción: {predicted_answer} ({result}) - Confianza: {confidence:.2f}")

    # 📊 Mostrar la precisión del modelo
    accuracy = correct / total * 100
    print(f"\n📈 Precisión del modelo: {accuracy:.2f}% ({correct}/{total} correctas)")

# ✅ Llamar a la función para mostrar las predicciones
decode_predictions(predicted_labels, probs, final_dataset["validation"])

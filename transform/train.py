from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3ForTokenClassification
import torch
import json
import os
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    "test": validation_test_split["test"],  # El test debería ser el split correcto
})

# Crear el mapa de etiquetas
label_map = {answer: idx for idx, answer in enumerate(all_answers)}

# ✅ Load the model and processor
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(label_map))
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# 📦 ✅ Preprocessing function
from PIL import Image
import torch
import os
from transformers import LayoutLMv3Processor

# ✅ Cargar el modelo y el procesador
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# ✅ Preprocesar datos con bounding boxes
def preprocess_data(example):
    if "fields" not in example or len(example["fields"]) == 0:
        return None  # Saltar ejemplos vacíos

    image_path = os.path.join("../images", f"{example['file_name']}.png")
    image = Image.open(image_path).convert("RGB")

    questions = [field["question"] for field in example["fields"]]
    answers = [field["answer"] for field in example["fields"]]
    boxes = [field["box"] for field in example["fields"]]

    # Tokenizar cada pregunta antes de pasarlas al procesador
    tokenized_questions = [processor.tokenizer.tokenize(q) for q in questions]

    # Procesar la imagen y las preguntas tokenizadas
    encoded = processor(
        images=image,
        text=tokenized_questions,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Convertir respuestas en etiquetas
    labels = [-100] * encoded["input_ids"].size(1)
    for i, answer in enumerate(answers):
        tokenized_answer = processor.tokenizer(answer).input_ids
        start_idx = labels.index(-100)  # Encontrar la posición de inicio para esta respuesta
        labels[start_idx:start_idx + len(tokenized_answer)] = [label_map[answer]] * len(tokenized_answer)

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "bbox": encoded["bbox"].squeeze(0),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

# ✅ Preprocess the complete dataset
encoded_dataset = final_dataset.map(preprocess_data, remove_columns=["file_name", "fields"])
# Verificar alineación entre input_ids y labels
# 🔍 Verificación de alineación entre tokens y etiquetas
# 🔍 Verificación de alineación entre tokens y etiquetas
for idx, example in enumerate(encoded_dataset["train"]):
    print(f"\n🧾 Ejemplo {idx + 1}:")
    print("Input IDs:", example["input_ids"][:10])
    
    # Aplanar las etiquetas
    flat_labels = [label for sublist in example["labels"] for label in sublist]

    print("Labels:", flat_labels[:10])  # Mostrar los primeros 10 labels aplanados

    # Verificar si los tokens etiquetados están alineados correctamente
    if all(label == -100 for label in flat_labels):
        print("⚠️ Advertencia: Todos los tokens están ignorados.")
    elif any(label >= len(label_map) for label in flat_labels if label != -100):
        print("❌ Error: Hay etiquetas fuera del rango del mapa de etiquetas.")
    else:
        print("✅ Tokens alineados correctamente.")

# ⚙️ ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
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

            # Handle out-of-bounds errors
            if i >= len(predicted_labels) or j >= len(predicted_labels[i]):
                pred_label = -100
                confidence = 0
            else:
                pred_label = predicted_labels[i][j]
                confidence = probs[i][j][pred_label] if pred_label != -100 else 0

            predicted_answer = field["answer"] if confidence >= threshold else "No prediction"

            # Compare the prediction with the actual answer
            if predicted_answer == actual_answer:
                result = "✅ Correcto"
                correct += 1
            else:
                result = "❌ Incorrecto"

            print(f"  - {question}")
            print(f"    🏷️ Real: {actual_answer}")
            print(f"    🤖 Predicción: {predicted_answer} ({result}) - Confianza: {confidence:.2f}")

    # Display the model accuracy
    accuracy = correct / total * 100
    print(f"\n📈 Precisión del modelo: {accuracy:.2f}% ({correct}/{total} correctas)")

# ✅ Llamar a la función para mostrar las predicciones
decode_predictions(predicted_labels, probs, final_dataset["validation"])

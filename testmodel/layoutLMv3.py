import pytesseract
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from torch.nn.functional import softmax
import re
import os

# 📂 Load trained model and processor
model_path = "../transform/trained_model"
processor = LayoutLMv3Processor.from_pretrained(model_path)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)

# 🔄 Label mappings
id2label = model.config.id2label

# 📄 Extract text and normalized bounding boxes
def extract_text_and_boxes(image_path):
    image = Image.open(image_path)
    width, height = image.size
    ocr_data = pytesseract.image_to_data(image, lang="spa", output_type=pytesseract.Output.DICT)

    words = []
    boxes = []

    for i, word in enumerate(ocr_data["text"]):
        if word.strip():
            words.append(word)
            
            # ✅ Normalizar coordenadas al rango 0-1000
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            normalized_box = [
                int((x / width) * 1000),
                int((y / height) * 1000),
                int(((x + w) / width) * 1000),
                int(((y + h) / height) * 1000),
            ]
            
            # ✅ Clipping adicional para asegurar el rango correcto
            clipped_box = [max(0, min(coord, 1000)) for coord in normalized_box]
            boxes.append(clipped_box)

    return words, boxes, image


# 🔧 Get predictions
def get_predictions(image_path):
    words, boxes, image = extract_text_and_boxes(image_path)
    encoding = processor(images=image, text=words, boxes=boxes, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    entities = {}
    for word, pred in zip(words, predictions):
        label = id2label.get(pred, "O")
        if label != "O":
            entities[label] = entities.get(label, "") + " " + word

    return entities

# 📂 Path to invoice image
image_path = "../images/Factura-C-00003-00000002.png"

# 🔧 Run inference
entities = get_predictions(image_path)
print("\nEntidades detectadas:")
for key, value in entities.items():
    print(f"{key}: {value.strip()}")

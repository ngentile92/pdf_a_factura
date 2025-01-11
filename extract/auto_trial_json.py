from google.cloud import vision
import json
import os
import pandas as pd
import unicodedata
import re
from difflib import SequenceMatcher

# ✅ Inicializar cliente de Google Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/nagge/Desktop/Nico/proyecto facturas/pdf_a_factura/key.json"
client = vision.ImageAnnotatorClient()

# 📄 Cargar el archivo CSV
csv_file = "facturas.csv"
df = pd.read_csv(csv_file, dtype=str)

# 🔧 Función para normalizar texto
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").upper().strip()
    text = re.sub(r"[^A-Z0-9.,$]", " ", text)  # Dejar letras, números y símbolos relevantes
    text = re.sub(r"\s+", " ", text)  # Eliminar espacios extras
    text = text.replace(".", "")  # Quitar puntos
    return text

# 🔍 Función para encontrar coincidencias parciales usando SequenceMatcher
def find_best_match(target, ocr_texts):
    target_normalized = normalize_text(target)
    best_match = None
    best_score = 0

    for ocr_text in ocr_texts:
        score = SequenceMatcher(None, target_normalized, normalize_text(ocr_text)).ratio()
        if score > best_score:
            best_score = score
            best_match = ocr_text

    return best_match if best_score > 0.6 else None  # Umbral reducido para más coincidencias

# 📦 Función para buscar campos y bounding boxes
def process_generic_fields(extracted_text, ocr_data, label, value):
    ocr_texts = extracted_text.split()
    match = find_best_match(value, ocr_texts)

    word_boxes = []
    if match:
        print(f"✅ Coincidencia encontrada: {value} en OCR: {match}")
        for page in ocr_data.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        if normalize_text(word_text) == normalize_text(match):
                            x_min = min(vertex.x for vertex in word.bounding_box.vertices)
                            y_min = min(vertex.y for vertex in word.bounding_box.vertices)
                            x_max = max(vertex.x for vertex in word.bounding_box.vertices)
                            y_max = max(vertex.y for vertex in word.bounding_box.vertices)
                            word_boxes.append([x_min, y_min, x_max, y_max])
    else:
        print(f"⚠️ No se encontró coincidencia para: {value}")

    return word_boxes

# 🧾 Función específica para buscar frases completas
def find_phrase_in_ocr(ocr_texts, phrase):
    words = phrase.split()
    matches = []

    for i in range(len(ocr_texts) - len(words) + 1):
        if all(normalize_text(word) in normalize_text(ocr_text) for word, ocr_text in zip(words, ocr_texts[i:])):
            matches.append(" ".join(ocr_texts[i:i+len(words)]))

    return matches

# 📦 Función principal para procesar una factura
def process_invoice(image_path, labels):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    ocr_data = response.full_text_annotation
    extracted_text = ocr_data.text.replace("\n", " ")  # Eliminar saltos de línea
    ocr_texts = extracted_text.split()

    print("\n🔎 TEXTO EXTRAÍDO POR VISION API:")
    print(extracted_text)

    fields = []
    for label, value in labels.items():
        if pd.isna(value) or not value:
            continue

        # 🔍 Procesar campos de forma genérica
        word_boxes = process_generic_fields(extracted_text, ocr_data, label, value)

        # ✅ Búsqueda específica para "Honorarios servicios profesionales"
        if label.lower() == "producto/servicio ofrecido":
            matches = find_phrase_in_ocr(ocr_texts, value)
            if matches:
                print(f"✅ Coincidencia de frase encontrada: {matches}")
                word_boxes = process_generic_fields(extracted_text, ocr_data, label, matches[0])

        # ✅ Búsqueda específica para "Subtotal"
        if label.lower() == "subtotal":
            normalized_value = normalize_text(value).replace("$", "").replace(".", "").replace(",", "")
            match = find_best_match(normalized_value, ocr_texts)
            if match:
                print(f"✅ Subtotal encontrado: {match}")
                word_boxes = process_generic_fields(extracted_text, ocr_data, label, match)

        # 📦 Combinar las bounding boxes de todas las palabras para formar un único box
        if word_boxes:
            x_min = min(box[0] for box in word_boxes)
            y_min = min(box[1] for box in word_boxes)
            x_max = max(box[2] for box in word_boxes)
            y_max = max(box[3] for box in word_boxes)

            normalized_box = [
                int((x_min / ocr_data.pages[0].width) * 1000),
                int((y_min / ocr_data.pages[0].height) * 1000),
                int((x_max / ocr_data.pages[0].width) * 1000),
                int((y_max / ocr_data.pages[0].height) * 1000),
            ]
        else:
            normalized_box = []  # Si no encuentra box, se deja vacío

        # ✅ Agregar la pregunta, respuesta y bounding box
        question = f"¿Cuál es {label.replace('_', ' ').lower()}?"
        fields.append({
            "question": question,
            "answer": value,
            "box": normalized_box
        })

        if word_boxes:
            print(f"📦 Campo agregado: {question} - {value} - {normalized_box}")
        else:
            print(f"⚠️ Campo agregado sin bounding box: {question} - {value}")

    return fields

# 🧾 Procesar todas las facturas y generar el dataset
dataset = []
for index, row in df.iterrows():
    file_name = row["factura"]
    labels = row.drop("factura").to_dict()
    image_path = os.path.join("../images", f"{file_name}.png")

    if os.path.exists(image_path):
        fields = process_invoice(image_path, labels)
        dataset.append({
            "file_name": file_name,
            "fields": fields
        })
    break  # ✅ Procesar solo una factura

# 💾 Guardar el dataset en formato JSON
output_file = "facturas_dataset.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ Dataset generado: {output_file}")

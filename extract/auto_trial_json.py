from google.cloud import vision
import json
import os
import shutil
import pandas as pd
import unicodedata
import re
from difflib import SequenceMatcher, ndiff

# ✅ Inicializar cliente de Google Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/nagge/Desktop/Nico/proyecto facturas/pdf_a_factura/key.json"
client = vision.ImageAnnotatorClient()

# 📄 Cargar el archivo CSV
csv_file = "facturas_corregidas.csv"
df = pd.read_csv(csv_file, dtype=str)

# 🗑️ Eliminar contenido de la carpeta de reportes de errores
error_reports_path = "../error_reports"
if os.path.exists(error_reports_path):
    shutil.rmtree(error_reports_path)
os.makedirs(error_reports_path)

# 🔧 Función para normalizar texto
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").upper().strip()
    text = re.sub(r"[^A-Z0-9.,$]", " ", text)  # Dejar letras, números y símbolos relevantes
    text = re.sub(r"\s+", " ", text)  # Eliminar espacios extras
    return text

# 🔍 Función mejorada para encontrar coincidencias de frases completas
def find_best_match(target, ocr_texts):
    target_normalized = normalize_text(target)
    best_match = None
    best_score = 0

    for ocr_text in ocr_texts:
        ocr_normalized = normalize_text(ocr_text)  # Normalizar el texto OCR
        score = SequenceMatcher(None, target_normalized, ocr_normalized).ratio()

        # Mostrar siempre el score
        print(f"🔍 Comparando: '{target_normalized}' vs. '{ocr_normalized}'")
        print(f"  Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = ocr_text

        # Si encontramos una coincidencia exacta, salir del loop
        if score == 1.0:
            print(f"✅ Coincidencia exacta encontrada: '{target_normalized}' vs. '{ocr_normalized}'")
            return best_match

        # Mostrar diferencias detalladas
        differences = list(ndiff(target_normalized.split(), ocr_normalized.split()))
        print("  Diferencias:")
        for diff in differences:
            if diff.startswith("-"):
                print(f"    ❌ {diff}")
            elif diff.startswith("+"):
                print(f"    ➕ {diff}")
            else:
                print(f"    ✅ {diff}")

    print(f"🔍 Mejor coincidencia para '{target}': {best_match} - Score: {best_score}")
    return best_match if best_score > 0.6 else None

# 📦 Función para buscar campos y bounding boxes genéricos
def process_generic_fields(extracted_text, ocr_data, label, value):
    ocr_texts = extracted_text.split()
    target_words = value.split()

    # Implementación correcta de una ventana deslizante (moving window) que genere solo combinaciones consecutivas
    if len(target_words) > 1:
        ocr_blocks = [" ".join(ocr_texts[i:i + len(target_words)])
                      for i in range(len(ocr_texts) - len(target_words) + 1)]
    else:
        ocr_blocks = ocr_texts

    # Buscar la mejor coincidencia entre los bloques
    match = find_best_match(value, ocr_blocks)

    word_boxes = []
    if match:
        for page in ocr_data.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = normalize_text("".join([symbol.text for symbol in word.symbols]))
                        if word_text in normalize_text(match):
                            x_min = min(vertex.x for vertex in word.bounding_box.vertices)
                            y_min = min(vertex.y for vertex in word.bounding_box.vertices)
                            x_max = max(vertex.x for vertex in word.bounding_box.vertices)
                            y_max = max(vertex.y for vertex in word.bounding_box.vertices)
                            word_boxes.append([x_min, y_min, x_max, y_max])

    # 🛠️ Combinar bounding boxes en uno solo
    if word_boxes:
        combined_box = combine_bounding_boxes(word_boxes)
        return combined_box

    return []

# 📦 Función para procesar campos específicos como frases completas
def process_specific_fields(extracted_text, ocr_data, label, value):
    ocr_texts = extracted_text.split()
    matches = find_phrase_in_ocr(ocr_texts, value)

    word_boxes = []
    if matches:
        for match in matches:
            for page in ocr_data.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = normalize_text("".join([symbol.text for symbol in word.symbols]))
                            if normalize_text(word_text) in normalize_text(match):
                                x_min = min(vertex.x for vertex in word.bounding_box.vertices)
                                y_min = min(vertex.y for vertex in word.bounding_box.vertices)
                                x_max = max(vertex.x for vertex in word.bounding_box.vertices)
                                y_max = max(vertex.y for vertex in word.bounding_box.vertices)
                                word_boxes.append([x_min, y_min, x_max, y_max])

    if word_boxes:
        x_min = min(box[0] for box in word_boxes)
        y_min = min(box[1] for box in word_boxes)
        x_max = max(box[2] for box in word_boxes)
        y_max = max(box[3] for box in word_boxes)
        return [x_min, y_min, x_max, y_max]

    return []

# 📦 Función para buscar frases completas en el texto OCR
def find_phrase_in_ocr(ocr_texts, phrase):
    words = phrase.split()
    matches = []

    for i in range(len(ocr_texts) - len(words) + 1):
        if all(normalize_text(word) == normalize_text(ocr_text) for word, ocr_text in zip(words, ocr_texts[i:i + len(words)])):
            matches.append(" ".join(ocr_texts[i:i + len(words)]))

    return matches

# 📦 Función para formatear número de comprobante
def format_invoice_number(number):
    number = str(number).strip()
    return f"{'0' * (8 - len(number))}{number}"

# 📦 Función principal para procesar una factura
# 📦 Función principal para procesar una factura
def process_invoice(image_path, labels):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    ocr_data = response.full_text_annotation
    extracted_text = ocr_data.text.replace("\n", " ")

    print(f"\n🔎 TEXTO EXTRAÍDO POR VISION API para: {image_path}")
    print(extracted_text)

    fields = []
    missing_fields = []

    for label, value in labels.items():
        if pd.isna(value) or not value:
            continue

        question = f"¿Cuál es {label.replace('_', ' ').lower()}?"
        field_data = {"question": question, "answer": value, "box": []}

        # Procesar según el tipo de campo
        if label.lower() == "comprobante nro":
            # check the lenght of the number
            value = format_invoice_number(value)
            field_data["box"] = process_generic_fields(extracted_text, ocr_data, label, value)
        else:
            field_data["box"] = process_generic_fields(extracted_text, ocr_data, label, value)

        if field_data["box"]:
            print(f"📦 Campo agregado: {question} - {value} - {field_data['box']}")
        else:
            missing_fields.append({"question": question, "expected": value})
            print(f"⚠️ Campo agregado sin bounding box: {question} - {value}")

        fields.append(field_data)

    return fields

# 📦 Función para procesar valores numéricos
def process_numeric_field_repeated(ocr_data, target_value):
    if "$" in target_value:
        target_value = target_value.replace("$", "").strip()

    # Crear diferentes variantes del formato
    variants = []
    
    # Formato original
    variants.append(target_value)
    
    # Formato sin punto
    no_dots = target_value.replace(".", "")
    variants.append(no_dots)
    
    # Formato sin coma
    if "," in target_value:
        no_comma = target_value.replace(",", "")
        variants.append(no_comma)

    # Agregar el formato original de búsqueda de componentes
    components = re.findall(r"\d{1,3}(?:\.\d{3})*,\d{2}", target_value)
    variants.extend(components)

    print(f"🔎 Buscando variantes del número: {variants}")

    found_boxes = []
    for page in ocr_data.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    word_text = word_text.replace("$", "").strip()
                    
                    # Verificar todas las variantes
                    for variant in variants:
                        if variant in word_text or word_text in variant:
                            bounding_box = word.bounding_box
                            x_min = min(vertex.x for vertex in bounding_box.vertices)
                            y_min = min(vertex.y for vertex in bounding_box.vertices)
                            x_max = max(vertex.x for vertex in bounding_box.vertices)
                            y_max = max(vertex.y for vertex in bounding_box.vertices)
                            found_boxes.append([x_min, y_min, x_max, y_max])

    if found_boxes:
        # Combinar todos los boxes encontrados
        x_min = min(box[0] for box in found_boxes)
        y_min = min(box[1] for box in found_boxes)
        x_max = max(box[2] for box in found_boxes)
        y_max = max(box[3] for box in found_boxes)
        return [x_min, y_min, x_max, y_max]

    print(f"⚠️ No se encontraron bounding boxes para los componentes de: {target_value}")
    return []
# Combinar múltiples bounding boxes en uno solo
def combine_bounding_boxes(boxes):
    if not boxes:
        return []

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)

    return [x_min, y_min, x_max, y_max]
# Función para detectar errores entre los textos extraídos y los valores esperados
def detect_errors(file_name, extracted_texts, expected_values):
    errors = []
    for label, expected_value in expected_values.items():
        if expected_value not in extracted_texts:
            # Combinar bounding boxes si hay múltiples
            boxes = [box for text, box in extracted_texts if text == expected_value]
            combined_box = combine_bounding_boxes(boxes)
            errors.append({
                "field": label,
                "expected": expected_value,
                "status": "Not Found",
                "box": combined_box
            })
    return errors

# 🧾 Procesar todas las facturas y generar el dataset
dataset = []
for index, row in df.iterrows():
    file_name = row["factura"]
    labels = row.drop("factura").to_dict()
    image_path = os.path.join("../images", f"{file_name}.png")

    if os.path.exists(image_path):
        fields = process_invoice(image_path, labels)
        dataset.append({"file_name": file_name, "fields": fields})


# 💾 Guardar el dataset en formato JSON
output_file = "facturas_dataset.json"
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"✅ Dataset generado: {output_file}")
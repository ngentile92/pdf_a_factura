import os
import json
import pandas as pd
from google.cloud import vision

# Configurar cliente de Google Vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"
client = vision.ImageAnnotatorClient()

# Cargar el CSV con los valores correctos
csv_path = "facturas_corregidas.csv"
df = pd.read_csv(csv_path)

# Función para procesar una imagen con la API de Google Vision
def process_image(file_path):
    with open(file_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response

# Función para extraer texto desde la respuesta de Google Vision
def extract_text_from_response(response):
    extracted_texts = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = ''.join([symbol.text for symbol in word.symbols])
                    extracted_texts.append(text)
    return extracted_texts

# Función para detectar errores entre los textos extraídos y los valores esperados
def detect_errors(file_name, extracted_texts, expected_values):
    errors = []
    for label, expected_value in expected_values.items():
        if expected_value not in extracted_texts:
            errors.append({
                "field": label,
                "expected": expected_value,
                "status": "Not Found"
            })
    return errors

# Función principal
output_errors = []
for index, row in df.iterrows():
    factura_file = f"../facturas/{row['factura']}.pdf"
    expected_values = row.to_dict()

    # Procesar la imagen
    response = process_image(factura_file)
    extracted_texts = extract_text_from_response(response)

    # Detectar errores
    errors = detect_errors(row['factura'], extracted_texts, expected_values)
    if errors:
        output_errors.append({"file_name": row['factura'], "errors": errors})

# Guardar los errores en un archivo JSON
with open("error_report.json", "w", encoding="utf-8") as f:
    json.dump(output_errors, f, indent=4, ensure_ascii=False)

print("Proceso completado. Revisa el archivo error_report.json para ver los errores detectados.")
import os
import json
import pandas as pd
from google.cloud import vision
from pdf2image import convert_from_path

# Configura las credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"

# Inicializa el cliente de Vision API
client = vision.ImageAnnotatorClient()

# Cargar el CSV con los valores correctos
csv_path = "facturas_corregidas.csv"
df = pd.read_csv(csv_path)

def process_image(file_path):
    """Procesa una imagen usando Google Vision API."""
    with open(file_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    
    response = client.text_detection(image=image)
    return response

def process_pdf(pdf_path):
    """Convierte solo la primera página de un PDF en imagen y la procesa."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=72)
    image_path = "temp_page_1.png"
    pages[0].save(image_path, "PNG")
    pages[0].save(f"../images/{os.path.basename(pdf_path).replace('.pdf', '.png')}", "PNG")
    response = process_image(image_path)
    os.remove(image_path)
    return response

def extract_text_and_bboxes(response):
    """Extrae texto y bounding boxes del resultado de Vision API."""
    text_data = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = ''.join([symbol.text for symbol in word.symbols])
                    bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                    text_data.append({"text": text, "bbox": bbox})
    
    return text_data

def label_data(text_data, values):
    """Etiqueta los datos extraídos basándose en los valores del CSV."""
    labeled_data = []
    missing_data = []

    for label, value in values.items():
        found = False
        for item in text_data:
            if value in item["text"]:
                labeled_data.append({
                    "text": item["text"],
                    "bbox": item["bbox"],
                    "label": label
                })
                found = True
                break
        if not found:
            missing_data.append({"field": label, "expected": value})

    return labeled_data, missing_data

def save_error_report(file_name, ocr_texts, missing_data):
    """Guarda un reporte de errores con el OCR completo y los valores no encontrados."""
    report = {
        "file_name": file_name,
        "ocr_texts": ocr_texts,
        "missing_data": missing_data
    }
    with open(f"../error_reports/{file_name}_error_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

def main():
    output_data = []

    for index, row in df.iterrows():
        factura_file = f"../facturas/{row['factura']}.pdf"
        values = row.to_dict()

        if factura_file.endswith(".pdf"):
            response = process_pdf(factura_file)
        else:
            response = process_image(factura_file)

        text_data = extract_text_and_bboxes(response)
        labeled_data, missing_data = label_data(text_data, values)

        output_data.extend(labeled_data)

        # Guardar reporte de errores si hay datos faltantes
        if missing_data:
            ocr_texts = [item["text"] for item in text_data]
            save_error_report(row['factura'], ocr_texts, missing_data)

    # Guardar el dataset etiquetado
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    os.makedirs("../error_reports", exist_ok=True)
    main()

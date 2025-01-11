import os
import json
import pandas as pd
from google.cloud import vision
from pdf2image import convert_from_path
from difflib import SequenceMatcher

# ✅ Configure Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"

# ✅ Initialize Vision API client
client = vision.ImageAnnotatorClient()

# ✅ Load the CSV with the correct values
csv_path = "facturas.csv"
df = pd.read_csv(csv_path)

# ✅ Create a folder to save images if it doesn't exist
os.makedirs("../images", exist_ok=True)

def process_image(file_path):
    """Process an image using Google Vision API."""
    with open(file_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    return response

def process_pdf(pdf_path):
    """Convert only the first page of a PDF to an image and process it."""
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
    image_path = f"../images/{os.path.basename(pdf_path).replace('.pdf', '.png')}"
    pages[0].save(image_path, "PNG")
    response = process_image(image_path)
    return response, image_path

def extract_text_and_bboxes(response):
    """Extract text and bounding boxes at the word level from the Vision API response."""
    text_data = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = ''.join([symbol.text for symbol in word.symbols])
                    bbox = [(v.x, v.y) for v in word.bounding_box.vertices]
                    text_data.append({"text": text, "bbox": bbox})

    return text_data

def normalize_text(text):
    """Normalize text by converting to lowercase and removing punctuation."""
    return text.lower().replace(".", "").replace(",", "").strip()

def find_best_match(concept, text_data):
    """Find the best match for a concept in the extracted text."""
    concept_normalized = normalize_text(concept)
    best_match = None
    best_score = 0

    for item in text_data:
        item_text_normalized = normalize_text(item["text"])
        score = SequenceMatcher(None, concept_normalized, item_text_normalized).ratio()
        if score > best_score:
            best_score = score
            best_match = item

    return best_match if best_score > 0.6 else None

def label_concepts_from_csv(text_data, values):
    """Label concepts from the CSV using approximate matching."""
    labeled_data = []
    factura_id = values["factura"].values[0].split('.')[0]

    for column in values.columns:
        if column == "factura":
            continue  # ✅ Skip the "factura" column

        concept = str(values[column].values[0])
        match = find_best_match(concept, text_data)

        if match:
            labeled_data.append({
                "factura_id": factura_id,
                "text": match["text"],
                "bbox": match["bbox"],
                "label": column
            })
        else:
            print(f"No match found for: {concept}")

    return labeled_data

def main():
    output_data = []

    for index, row in df.iterrows():
        factura_file = f"../facturas/{row['factura']}.pdf"
        values = pd.DataFrame([row])

        if factura_file.endswith(".pdf"):
            response, image_path = process_pdf(factura_file)
        else:
            response = process_image(factura_file)
            image_path = f"../images/{row['factura']}.png"

        # ✅ Extract only the file name (remove the path)
        image_name = os.path.basename(image_path)

        text_data = extract_text_and_bboxes(response)
        labeled_data = label_concepts_from_csv(text_data, values)

        output_data.append({
            "image": image_name,
            "words": [item["text"] for item in labeled_data],
            "bboxes": [item["bbox"] for item in labeled_data],
            "labels": [item["label"] for item in labeled_data]
        })

    # ✅ Save the labeled dataset
    with open("../extract/layoutlm_dataset.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

import os
import json
from collections import defaultdict
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import LayoutLMv3Processor
from pdf2image import convert_from_path

# Load the processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# Path to the folder containing the PDFs
pdf_folder = "../facturas/"

# Load the JSON dataset
with open("../extract/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Group data by factura_id
grouped_data = defaultdict(list)
for item in data:
    grouped_data[item["factura_id"]].append(item)

# Create lists to store the inputs
images = []
word_list = []
box_list = []
label_list = []

# Process each factura
for factura_id, items in grouped_data.items():
    # Load the corresponding PDF and convert to image
    pdf_path = os.path.join(pdf_folder, f"{factura_id}.pdf")
    if not os.path.exists(pdf_path):
        continue

    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
    image = pages[0].convert("RGB")
    images.append(image)

    # Create lists for words, boxes, and labels for this factura
    words = [item["text"] for item in items]
    boxes = [item["bbox"] for item in items]
    labels = [item.get("label", "O") for item in items]  # Default to "O" if no label is provided

    word_list.append(words)
    box_list.append(boxes)
    label_list.append(labels)

# Debug print to check the input lists before passing to the processor
print(f"Number of images: {len(images)}")
print(f"Number of word lists: {len(word_list)}")
print(f"Number of box lists: {len(box_list)}")
print(f"Number of label lists: {len(label_list)}")

for i in range(len(word_list)):
    print(f"Image {i+1}:")
    print(f"  Words: {type(word_list[i])}, Length: {len(word_list[i])}")
    print(f"  Boxes: {type(box_list[i])}, Length: {len(box_list[i])}")
    print(f"  Labels: {type(label_list[i])}, Length: {len(label_list[i])}")

# Ensure lengths match
for i in range(len(word_list)):
    assert len(word_list[i]) == len(box_list[i]), f"Mismatch in Image {i+1}: Words {len(word_list[i])}, Boxes {len(box_list[i])}"

# Encode inputs using the processor
encoded_inputs = processor(
    images=images,
    text=word_list,
    boxes=box_list,
    word_labels=label_list,
    padding="max_length",
    truncation=True
)
for key, value in encoded_inputs.items():
    print(f"{key}: Type={type(value)}, Example={value[:1]}")

# Normalize the encoded inputs
encoded_inputs_dict = {
    key: [v if isinstance(v, list) else [v] for v in value] for key, value in encoded_inputs.items()
}

# Check for None values
encoded_inputs_dict = {key: [v for v in value if v is not None] for key, value in encoded_inputs_dict.items()}

# Convert to a Dataset
dataset = Dataset.from_dict(encoded_inputs_dict)


# Split the dataset into train, validation, and test sets
train_test_valid = dataset.train_test_split(test_size=0.2)
train_test_split = train_test_valid["test"].train_test_split(test_size=0.5)
final_dataset = DatasetDict({
    "train": train_test_valid["train"],
    "validation": train_test_split["test"],
    "test": train_test_split["train"]
})

# Save the dataset to disk
final_dataset.save_to_disk("layoutlmv3_dataset")

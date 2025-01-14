import json
import os
from PIL import Image

# 📄 Cargar el archivo JSON con las facturas y sus boxes
with open("facturas_dataset.json", "r", encoding="utf-8") as f:
    factura_data = json.load(f)

# 📂 Directorio de las imágenes originales
image_dir = "../images"
# 📂 Directorio donde se guardarán los recortes
output_dir = "./recortes"
os.makedirs(output_dir, exist_ok=True)
#delete all files in the output_dir
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

# 📦 Función para recortar y guardar las imágenes
for factura in factura_data[:7]:  # Iterar solo las primeras 2 facturas
    file_name = factura["file_name"]
    print(f"Procesando factura: {file_name}")
    image_path = os.path.join(image_dir, f"{file_name}.png")

    if not os.path.exists(image_path):
        print(f"⚠️ Imagen no encontrada: {image_path}")
        continue

    # Abrir la imagen original
    image = Image.open(image_path)

    for field in factura["fields"]:
        question = field["question"].replace("?", "").replace("/", "-")
        box = field["box"]

        # Recortar la imagen usando las coordenadas del box
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))

        # Crear un nombre para la imagen recortada
        output_path = os.path.join(output_dir, f"{file_name}_{question}.png")

        # Guardar la imagen recortada
        cropped_image.save(output_path)
        print(f"✅ Imagen guardada: {output_path}")

print("✅ Recortes completados.")
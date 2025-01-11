import cv2
import pytesseract
import os
import pandas as pd
from pdf2image import convert_from_path
import numpy as np

# Ruta de las facturas y salida
FACTURAS_DIR = './facturas/'
OUTPUT_DIR = './output/'

# Función para procesar una factura (maneja tanto imágenes como PDFs)
def procesar_factura(archivo):
    # Si el archivo es PDF, convertirlo a imágenes
    if archivo.endswith('.pdf'):
        try:
            paginas = convert_from_path(archivo, 300)  # Resolución de 300 DPI
            # Procesar solo la primera página como ejemplo
            img = np.array(paginas[0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convertir a formato compatible con OpenCV
        except Exception as e:
            print(f"Error al procesar el PDF {archivo}: {e}")
            return None
    else:
        # Leer la imagen directamente
        img = cv2.imread(archivo)
    
    if img is None:
        print(f"Error al cargar {archivo}. Verifica que sea un archivo válido.")
        return None
    
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar OCR
    texto_extraido = pytesseract.image_to_string(gris, lang='spa')  # 'spa' para español
    return texto_extraido

# Procesar todas las facturas en la carpeta
def procesar_todas_facturas():
    resultados = []
    archivos = [os.path.join(FACTURAS_DIR, f) for f in os.listdir(FACTURAS_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    
    for archivo in archivos:
        print(f"Procesando: {archivo}")
        texto = procesar_factura(archivo)
        if texto:
            resultados.append({'archivo': archivo, 'texto': texto})
    
    # Guardar los resultados en un archivo CSV
    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv(os.path.join(OUTPUT_DIR, 'resultados_facturas.csv'), index=False, encoding='utf-8')
        print("Resultados guardados en output/resultados_facturas.csv")



if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    procesar_todas_facturas()


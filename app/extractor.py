import pdfplumber
import pytesseract
from PIL import Image
from unidecode import unidecode
import re
import io


def extraer_texto_pdf(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        texto = ""
        for pagina in pdf.pages:
            texto += pagina.extract_text() + "\n"
    return texto


# Función para extraer texto de imágenes
def extraer_texto_imagen(imagen):
    imagen_pil = Image.open(imagen)
    texto = pytesseract.image_to_string(imagen_pil, lang="spa")
    return unidecode(texto)

# Función principal para extraer datos
def extraer_datos_factura(contenido, nombre_archivo):
    if nombre_archivo.endswith(".pdf"):
        texto = extraer_texto_pdf(contenido)
    else:
        texto = extraer_texto_imagen(contenido)

    patrones = {
        "CUIT": r"CUIT:\s*([\d\-]+)",
        "Fecha de Emisión": r"Fecha de Emisi[oó]n:\s*([\d\/]+)",
        "Punto de Venta": r"Punto de Venta:\s*([\d]+)",
        "Número de Factura": r"Comp\. Nro:\s*([\d\-]+)",
        "Razón Social": r"Raz[oó]n Social:\s*([\w\s]+)",
        "Monto Total": r"Importe Total:\s*\$?([\d\.,]+)",
        "CAE": r"CAE\s*N[°º]:\s*([\d]+)",
    }

    datos = {}
    for campo, patron in patrones.items():
        match = re.search(patron, texto)
        datos[campo] = match.group(1) if match else None

    return datos

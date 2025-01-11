import re
import pandas as pd
from unidecode import unidecode

# Función para normalizar texto y eliminar caracteres no deseados
def preprocesar_texto(texto):
    # Normalizar caracteres especiales y convertir a minúsculas
    texto = unidecode(texto)
    # Eliminar espacios múltiples y líneas innecesarias
    texto = re.sub(r"\s+", " ", texto)
    return texto

# Función para extraer datos de una factura
def extraer_datos_factura(texto):
    # Preprocesar texto antes de buscar patrones
    texto = preprocesar_texto(texto)
    
    # Definir patrones
    patrones = {
        "CUIT": r"cuit:\s*([\d\-]+)",
        "Fecha de Emisión": r"fecha de emision:\s*([\d\/]+)",
        "Punto de Venta": r"punto de venta:\s*([\d]+)",
        "Número de Factura": r"comp\.? nro:\s*([\d\-]+)",
        "Razón Social": r"razon social:\s*([\w\s]+)",
        "Monto Total": r"importe total:\s*\$?\s*([\d\.,]+)",
        "CAE": r"cae\s*n[°º]?:\s*([\d]+)",
    }
    
    datos = {}
    for campo, patron in patrones.items():
        match = re.search(patron, texto, re.IGNORECASE)
        datos[campo] = match.group(1).strip() if match else None
    
    # Formatear datos específicos
    if datos.get("Monto Total"):
        # Convertir montos a float
        datos["Monto Total"] = float(datos["Monto Total"].replace(".", "").replace(",", "."))
    
    return datos

# Procesar todas las facturas
def procesar_facturas_estructuradas(resultados_texto):
    datos_facturas = []
    for resultado in resultados_texto:
        texto = resultado.get("texto", "")
        archivo = resultado.get("archivo", "Desconocido")
        
        datos = extraer_datos_factura(texto)
        datos["Archivo"] = archivo
        
        # Validar datos extraídos
        if not datos.get("CUIT") or not datos.get("Monto Total"):
            print(f"Advertencia: Datos incompletos en el archivo {archivo}")
        
        datos_facturas.append(datos)
    
    # Convertir a DataFrame y guardar en CSV
    df = pd.DataFrame(datos_facturas)
    df.to_csv('./output/datos_facturas.csv', index=False, encoding='utf-8')
    print("Datos extraídos guardados en output/datos_facturas.csv")

# Leer el archivo resultados_facturas.csv y procesar
df = pd.read_csv('./output/resultados_facturas.csv')
procesar_facturas_estructuradas(df.to_dict('records'))

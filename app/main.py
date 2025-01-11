from fastapi import FastAPI, File, UploadFile
from app.extractor import extraer_datos_factura, extraer_texto_pdf
import os

app = FastAPI()

# Endpoint de prueba
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de extracción de facturas"}

@app.post("/procesar-factura")
async def procesar_factura(file: UploadFile = File(...)):
    # Read the file as bytes
    contenido = await file.read()

    # Extract the file name
    nombre_archivo = file.filename

    # Call the extraction function
    datos = extraer_datos_factura(contenido, nombre_archivo)

    return {"datos": datos}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

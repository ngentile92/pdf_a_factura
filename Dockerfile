# Usar una imagen base de Python
FROM python:3.12

# Crear un directorio de trabajo
WORKDIR /app

# Copiar los archivos al contenedor
COPY . .

# Instalar las dependencias
RUN pip install -r requirements.txt

# Exponer el puerto 8080
EXPOSE 8080

# Comando de inicio
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

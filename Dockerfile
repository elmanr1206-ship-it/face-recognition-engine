# Usamos una versión de Python liviana
FROM python:3.11-slim

# Instalamos las dependencias de sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Definimos dónde va a vivir el código en el servidor
WORKDIR /app

# Copiamos los requerimientos e instalamos las librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo tu código al servidor
COPY . .

# El comando que prende el motor automáticamente en el puerto 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
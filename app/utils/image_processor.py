import cv2
import os
import numpy as np

def procesar_imagenes_dataset(ruta_entrada, tamano=(100, 100)):
    """
    Lee las fotos, las convierte a gris, las redimensiona y las aplana.
    """
    caras_vectorizadas = []
    nombres = []

    if not os.path.exists(ruta_entrada):
        print(f"Error: No encontré la carpeta {ruta_entrada}")
        return None, None

    for archivo in os.listdir(ruta_entrada):
        if archivo.endswith((".jpg", ".png", ".jpeg")):
            # 1. Leer imagen
            img_path = os.path.join(ruta_entrada, archivo)
            img = cv2.imread(img_path)
            
            # 2. Convertir a Escala de Grises (Solo nos importa el brillo)
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. Redimensionar (KISS: Todas iguales para que el álgebra no explote)
            redimensionada = cv2.resize(gris, tamano)
            
            # 4. Aplanar (De matriz NxN a vector columna N^2)
            vector = redimensionada.flatten()
            
            caras_vectorizadas.append(vector)
            # Guardamos el nombre del archivo como etiqueta (sin la extensión)
            nombres.append(os.path.splitext(archivo)[0])

    # Convertimos la lista en una matriz de NumPy (La "Matriz A" de la teoría)
    return np.array(caras_vectorizadas), nombres
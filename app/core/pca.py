import numpy as np

class PCAEngine:
    def __init__(self, n_components=15):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None

    def fit(self, X):
        """
        Entrena el modelo con una matriz de caras X.
        Cada fila de X es una imagen aplanada (vector).
        """
        # 1. Calculamos la Cara Promedio (Ψ)
        self.mean_face = np.mean(X, axis=0)
        
        # 2. Centramos los datos (Resta de la media: Φ = Γ - Ψ)
        X_centered = X - self.mean_face
        
        # 3. Aplicamos SVD para obtener los Eigenvectores
        # U: Vectores unitarios, S: Valores singulares, Vt: Eigenfaces
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 4. Guardamos las componentes principales (las Eigenfaces más importantes)
        self.eigenfaces = Vt[:self.n_components]
        
        # 5. Proyectamos las caras originales al espacio reducido (Pesos Ω)
        self.weights = np.dot(X_centered, self.eigenfaces.T)

    def predict(self, face_vector):
        """
        Recibe una cara nueva, la proyecta y busca el match más cercano.
        """
        # Centrar la cara nueva
        centered = face_vector - self.mean_face
        
        # Proyectar al espacio de Eigenfaces
        projection = np.dot(centered, self.eigenfaces.T)
        
        # Calcular Distancia Euclidiana entre la nueva cara y la base de datos
        # d = ||Ω_nueva - Ω_guardada||
        distances = np.linalg.norm(self.weights - projection, axis=1)
        
        return np.argmin(distances), np.min(distances)
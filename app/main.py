import os
import cv2
import numpy as np
import base64
import json
import os
from sqlalchemy import create_engine
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.pca import PCAEngine

# 🏗️ CONFIGURACIÓN DE LA BASE DE DATOS (TiDB)
# En Render, pega tu Connection String en la variable de entorno DATABASE_URL
DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://fJSazaxG3kcXfKw.root:TU_PASS@gateway01.../test")

engine = create_engine(
    DB_URL, 
    connect_args={"ssl": {"ca": "/etc/ssl/certs/ca-certificates.crt"}}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo de la tabla
class UsuarioRostro(Base):
    __tablename__ = "usuarios_rostros"
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    vector_pesos = Column(Text) # Guardamos los 20 números como un JSON string

# Crear la tabla si no existe
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inyectar la sesión de la DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

pca = PCAEngine(n_components=20)
nombres_db = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 🧠 LÓGICA DE MOTOR CON BASE DE DATOS ---

def entrenar_motor_desde_db(db: Session):
    """Carga TODAS las caras crudas y revive el motor PCA desde cero."""
    global nombres_db
    registros = db.query(UsuarioRostro).all()
    
    # El álgebra necesita mínimo 2 personas para no estallarse
    if len(registros) < 2:
        pca.mean_face = None
        pca.eigenfaces = None
        pca.weights = None
        nombres_db = [r.nombre for r in registros]
        return False

    all_raw_faces = []
    nombres_db = []
    
    for r in registros:
        nombres_db.append(r.nombre)
        # Cargamos los 10,000 pixeles originales
        all_raw_faces.append(json.loads(r.vector_pesos))
    
    # 🌟 LA MAGIA: Entrenamos el PCA desde cero con las caras puras
    X = np.array(all_raw_faces)
    pca.fit(X) 
    return True

# --- 📸 PROCESAMIENTO DE IMAGEN ---

def cazar_cara(imagen_bytes):
    # 1. Ver si el radar (XML) se quedó en coma en el servidor
    if face_cascade.empty():
        return None, "RADAR_MUERTO"

    # 2. Decodificar los bytes que manda el JS
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "FOTO_CORRUPTA"
        
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 🌟 EL HACK DEL TESO: Ecualizar el histograma.
    # Esto fuerza los contrastes y hace que OpenCV no se ponga de nena con las sombras.
    gris = cv2.equalizeHist(gris)
    
    if np.mean(gris) < 20: 
        return None, "OSCURIDAD"
        
    # 3. Buscar la cara (Bajé la exigencia del scaleFactor pa' que detecte más fácil)
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
    
    if len(caras) == 0: 
        return None, "NO_CARA"
        
    # 4. Recortar la pepa de la cara más grande
    (x, y, w, h) = max(caras, key=lambda rect: rect[2] * rect[3])
    rostro = cv2.resize(gris[y:y+h, x:x+w], (100, 100))
    
    return rostro, "OK"

# --- 🚀 ENDPOINTS ---

@app.get("/")
async def read_index():
    # Le decimos que se meta a la carpeta 'frontend' a buscar el archivo
    path = os.path.join(os.getcwd(), "frontend", "index.html")
    
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": f"No encontré el index.html en la ruta: {path}"}

@app.get("/api/estado")
async def obtener_estado(db: Session = Depends(get_db)):
    # Contamos cuántos rostros hay registrados de verdad
    conteo = db.query(UsuarioRostro).count()
    return {
        "status": "vacio" if conteo == 0 else "listo",
        "vectores": conteo
    }

@app.on_event("startup")
async def startup_event():
    # Al arrancar, intentamos cargar lo que haya en la DB
    db = SessionLocal()
    entrenar_motor_desde_db(db)
    db.close()

@app.post("/api/registrar")
async def registrar(nombre: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    rostro, status = cazar_cara(contents)
    
    if status != "OK":
        return {"message": f"Error: {status}"}
    
    # 🎯 GUARDAMOS LA CARA CRUDA (Los 10,000 píxeles)
    vector_crudo = rostro.flatten().tolist()

    nuevo_usuario = UsuarioRostro(
        nombre=nombre.upper(),
        vector_pesos=json.dumps(vector_crudo)
    )
    db.add(nuevo_usuario)
    db.commit()
    
    # Reentrenamos para que el motor asimile al nuevo compa
    entrenar_motor_desde_db(db)
    
    return {"message": f"{nombre} matriculado. Motor actualizado."}

@app.post("/api/identificar")
async def identificar(file: UploadFile = File(...), db: Session = Depends(get_db)):
    entrenar_motor_desde_db(db)
    
    # Si Render se reinició y hay menos de 2 personas, el motor no puede operar
    if pca.weights is None or pca.mean_face is None:
        return {"nombre": "FALTAN_DATOS_O_DB_VACIA", "confianza": 0}

    contents = await file.read()
    rostro, status = cazar_cara(contents)
    if status != "OK": return {"nombre": status, "confianza": 0}

    # Proceso de comparación (Similitud del Coseno)
    vector = rostro.flatten()
    centered = vector - pca.mean_face
    projection = np.dot(centered, pca.eigenfaces.T)
    
    norm_weights = np.linalg.norm(pca.weights, axis=1)
    norm_proj = np.linalg.norm(projection)
    similitudes = np.dot(pca.weights, projection) / (norm_weights * norm_proj)
    
    umbral = 0.85
    idx_max = np.argmax(similitudes)
    sim_max = similitudes[idx_max]
    
    conf_mapeada = round(((sim_max - umbral) / (1.0 - umbral)) * 100, 2) if sim_max > umbral else 0

    return {
        "nombre": nombres_db[idx_max] if conf_mapeada > 0 else "DESCONOCIDO",
        "confianza": conf_mapeada
    }

@app.delete("/api/borrar_todo")
async def borrar_db(clave: str, db: Session = Depends(get_db)):
    if clave == "RoroAdmin123": # Tu clave secreta
        db.query(UsuarioRostro).delete()
        db.commit()
        entrenar_motor_desde_db(db)
        return {"message": "Base de datos limpia, jefe."}
    return {"message": "No eres el admin xd"}
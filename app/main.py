import os
import cv2
import numpy as np
import base64
import json
from sqlalchemy import create_engine
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.pca import PCAEngine

# 🏗️ CONFIGURACIÓN DE LA BASE DE DATOS
DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://fJSazaxG3kcXfKw.root:TU_PASS@gateway01.../test")

engine = create_engine(DB_URL, connect_args={"ssl": {"ca": "/etc/ssl/certs/ca-certificates.crt"}})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UsuarioRostro(Base):
    __tablename__ = "usuarios_rostros"
    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    vector_pesos = Column(Text) 

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

pca = PCAEngine(n_components=20)
nombres_db = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 🧠 LÓGICA DE MOTOR ---
def entrenar_motor_desde_db(db: Session):
    global nombres_db
    registros = db.query(UsuarioRostro).all()
    
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
        all_raw_faces.append(json.loads(r.vector_pesos)) 
    
    X = np.array(all_raw_faces)
    pca.fit(X) 
    return True

# --- 📸 PROCESAMIENTO: VISIÓN NOCTURNA ---
def cazar_cara(imagen_bytes):
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, "FOTO_CORRUPTA"
    
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris) 
    
    if np.mean(gris) < 20: return None, "OSCURIDAD"
        
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
    if len(caras) == 0: return None, "NO_CARA"
        
    (x, y, w, h) = max(caras, key=lambda rect: rect[2] * rect[3])
    rostro = cv2.resize(gris[y:y+h, x:x+w], (100, 100))
    return rostro, "OK"

# --- 🚀 ENDPOINTS ---
@app.get("/")
async def read_index():
    path = os.path.join(os.getcwd(), "frontend", "index.html")
    if os.path.exists(path): return FileResponse(path)
    return {"error": f"No encontré el index.html en: {path}"}

@app.get("/api/estado")
async def obtener_estado(db: Session = Depends(get_db)):
    conteo = db.query(UsuarioRostro).count()
    if conteo < 2 or pca.mean_face is None:
        return {"status": "vacio", "total": conteo, "nombres": nombres_db}

    _, buffer_mean = cv2.imencode('.jpg', pca.mean_face.reshape(100, 100))
    mean_b64 = base64.b64encode(buffer_mean).decode('utf-8')

    eigen_b64_list = []
    for ef in pca.eigenfaces:
        ef_norm = cv2.normalize(ef.reshape(100, 100), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, buffer_ef = cv2.imencode('.jpg', ef_norm)
        eigen_b64_list.append(base64.b64encode(buffer_ef).decode('utf-8'))

    return {
        "status": "ready",
        "total": conteo,
        "nombres": nombres_db,
        "cara_promedio": mean_b64,
        "eigenfaces": eigen_b64_list
    }

@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    entrenar_motor_desde_db(db)
    db.close()

@app.post("/api/registrar")
async def registrar(nombre: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    rostro, status = cazar_cara(contents)
    if status != "OK": return {"message": f"Error: {status}"}
    
    vector_crudo = rostro.flatten().tolist()
    nuevo_usuario = UsuarioRostro(nombre=nombre.upper(), vector_pesos=json.dumps(vector_crudo))
    db.add(nuevo_usuario)
    db.commit()
    
    entrenar_motor_desde_db(db)
    return {"message": f"{nombre} matriculado. Motor actualizado."}

@app.post("/api/identificar")
async def identificar(file: UploadFile = File(...), db: Session = Depends(get_db)):
    entrenar_motor_desde_db(db)
    if pca.weights is None or pca.mean_face is None:
        return {"nombre": "FALTAN_DATOS", "confianza": 0}

    contents = await file.read()
    rostro, status = cazar_cara(contents)
    if status != "OK": return {"nombre": status, "confianza": 0}

    vector = rostro.flatten()
    centered = vector - pca.mean_face
    projection = np.dot(centered, pca.eigenfaces.T)
    
    norm_weights = np.linalg.norm(pca.weights, axis=1)
    norm_proj = np.linalg.norm(projection)
    similitudes = np.dot(pca.weights, projection) / (norm_weights * norm_proj)
    
    umbral = 0.85
    top_indices = np.argsort(similitudes)[::-1][:3]
    top_3 = []
    for i in top_indices:
        sim_val = similitudes[i]
        conf = round(((sim_val - umbral) / (1.0 - umbral)) * 100, 2) if sim_val > umbral else 0
        top_3.append({"nombre": nombres_db[i], "distancia": conf})
        
    _, buffer_cara = cv2.imencode('.jpg', rostro)
    cara_b64 = base64.b64encode(buffer_cara).decode('utf-8')

    pesos_reales = [round(float(w), 4) for w in projection]

    # 🌟 RECONSTRUCCIÓN DEL MONSTRUO
    reconstruccion_centrada = np.dot(projection, pca.eigenfaces)
    rostro_reconstruido = reconstruccion_centrada + pca.mean_face
    
    rostro_reconstruido_norm = cv2.normalize(rostro_reconstruido.reshape(100, 100), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, buffer_recon = cv2.imencode('.jpg', rostro_reconstruido_norm)
    recon_b64 = base64.b64encode(buffer_recon).decode('utf-8')

    return {
        "nombre": top_3[0]["nombre"] if top_3[0]["distancia"] > 0 else "DESCONOCIDO",
        "confianza": top_3[0]["distancia"],
        "top_3": top_3,
        "cara_procesada": cara_b64,
        "pesos_reales": pesos_reales,
        "rostro_reconstruido": recon_b64
    }

@app.delete("/api/borrar_todo")
async def borrar_db(clave: str, db: Session = Depends(get_db)):
    if clave == "RoroAdmin123":
        db.query(UsuarioRostro).delete()
        db.commit()
        entrenar_motor_desde_db(db)
        return {"message": "Base de datos limpia, jefe."}
    return {"message": "No eres el admin xd"}
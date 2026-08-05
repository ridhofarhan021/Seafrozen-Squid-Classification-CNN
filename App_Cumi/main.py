import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# Variabel global untuk menyimpan model
model = None
CLASSES = ['Cendol', 'CK', 'SMP DW', 'SMP UP']

# ==========================================
# 1. DEFINISIKAN LIFESPAN TERLEBIH DAHULU
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("⏳ Menyiapkan server... Membangun arsitektur EfficientNetB0...")
    
    # A. Bangun arsitektur
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # B. Load Bobot
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Pastikan nama file ini SAMA PERSIS dengan yang ada di dalam folder App_Cumi
    MODEL_PATH = os.path.join(BASE_DIR, "Model_Final_Juara.weights.h5") 
    
    try:
        model.load_weights(MODEL_PATH)
        print("✅ SUCCESS: Model AI Juara berhasil dimuat ke memori!")
    except Exception as e:
        print(f"❌ ERROR: Gagal memuat bobot model. Pastikan file .weights.h5 ada di direktori yang benar. Detail: {e}")
    
    yield 
    
    print("🛑 Server dimatikan, membersihkan memori GPU/CPU...")
    tf.keras.backend.clear_session()

# ==========================================
# 2. INISIALISASI APLIKASI FASTAPI (HANYA SEKALI)
# ==========================================
app = FastAPI(
    title="API Deteksi Varian Cumi VIPAC",
    description="Sistem Backend AI untuk menyortir kualitas cumi ekspor",
    version="1.0.0",
    lifespan=lifespan # Memanggil fungsi lifespan yang sudah didefinisikan di atas
)

# ==========================================
# 3. KONFIGURASI CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 4. FUNGSI PREPROCESSING GAMBAR
# ==========================================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# 5. ENDPOINT (JALUR DATA)
# ==========================================
@app.post("/predict/")
async def predict_cumi(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Format file tidak didukung. Harap unggah PNG, JPG, atau JPEG.")
    
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        detail_probs = {
            CLASSES[i]: round(float(predictions[0][i]) * 100, 2) 
            for i in range(len(CLASSES))
        }
        
        # --- LOGIKAN THRESHOLD BARU ---
        THRESHOLD = 0.75 # Ambang batas 75%
        
        if confidence < THRESHOLD:
            final_prediction = "Unidentified Object (Cek Manual)"
        else:
            final_prediction = CLASSES[predicted_idx]
        # ------------------------------
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": final_prediction, # Gunakan variabel final_prediction
            "confidence_percent": round(confidence * 100, 2),
            "detail_probabilities": detail_probs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server saat memproses gambar: {str(e)}")
  
  
@app.get("/")
def root():
    # Mengembalikan file index.html secara langsung saat root URL diakses
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "File index.html tidak ditemukan di folder App_Cumi"}    
# @app.get("/")
# def root():
#     return {
#         "message": "Selamat datang di API Deteksi Cumi VIPAC", 
#         "docs_url": "/docs",
#         "status": "Running"
#     }
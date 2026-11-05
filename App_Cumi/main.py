import os
import io
import time
import shutil
import tempfile
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
DEVICE = "CPU"
CLASSES = ['Cendol', 'CK', 'SMP DW', 'SMP UP']

# ==========================================
# 0. KONFIGURASI MODEL (REVISI PAPER ICIMCIS)
# ==========================================
# Bobot model frozen-backbone EfficientNet-B0 (skenario S3) yang dipilih lewat macro-F1 validasi.
# File ini dibuat oleh notebook analisis (Langkah 3, DEPLOY = "frozen"); salin ke folder App_Cumi.
WEIGHTS_FILE = "Model_Terpilih_Frozen.weights.h5"
# Ambang confidence gate, dipilih di validation set (selective risk <= 2%). Sebelumnya 0.75.
THRESHOLD = 0.78


def load_weights_any(net, path):
    """Memuat bobot HDF5. Mulai TF 2.13, nama yang berakhiran .weights.h5 dibaca sebagai format Keras
    baru, sehingga file format lama (disimpan dengan TF 2.10) gagal dibaca. Kalau itu terjadi, file
    disalin ke nama berakhiran .h5 supaya dibaca dengan pembaca format lama."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File bobot tidak ditemukan: {path}")
    try:
        net.load_weights(path)
        return "langsung"
    except Exception:
        tmp = os.path.join(tempfile.gettempdir(), "bobot_server_format_lama.h5")
        shutil.copyfile(path, tmp)
        net.load_weights(tmp)
        return "lewat salinan .h5 (format HDF5 lama)"


# ==========================================
# 1. DEFINISIKAN LIFESPAN TERLEBIH DAHULU
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, DEVICE
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            DEVICE = "GPU " + tf.config.experimental.get_device_details(gpus[0]).get("device_name", gpus[0].name)
        except Exception:
            DEVICE = "GPU " + gpus[0].name
    print(f"⏳ Menyiapkan server (TensorFlow {tf.__version__}, inferensi di {DEVICE})... "
          "Membangun arsitektur EfficientNetB0...")

    # A. Bangun arsitektur
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    net = models.Model(inputs=base_model.input, outputs=outputs)

    # B. Load Bobot
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, WEIGHTS_FILE)

    try:
        cara = load_weights_any(net, MODEL_PATH)
        model = net
        print(f"✅ SUCCESS: Bobot {WEIGHTS_FILE} berhasil dimuat ({cara}). Threshold = {THRESHOLD}")
    except Exception as e:
        # Jangan melayani prediksi dengan jaringan yang bobotnya belum dimuat (hasilnya acak).
        model = None
        print(f"❌ ERROR: Gagal memuat bobot model. Endpoint /predict/ akan mengembalikan HTTP 500 "
              f"sampai file bobot diperbaiki. Detail: {e}")

    yield

    print("🛑 Server dimatikan, membersihkan memori GPU/CPU...")
    tf.keras.backend.clear_session()

# ==========================================
# 2. INISIALISASI APLIKASI FASTAPI (HANYA SEKALI)
# ==========================================
app = FastAPI(
    title="API Deteksi Varian Cumi VIPAC",
    description="Sistem Backend AI untuk menyortir kualitas cumi ekspor",
    version="1.1.0",
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
# Catatan: gambar training diputar dan di-crop ke blok (rembg), sedangkan server
# hanya me-resize seluruh frame ke 224 x 224 (tidak berubah dari versi sebelumnya).
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

    if model is None:
        raise HTTPException(status_code=500, detail="Model AI gagal dimuat di server. Prediksi tidak tersedia, lakukan pengecekan manual.")

    try:
        contents = await file.read()
        t0 = time.perf_counter()   # waktu proses server: decode + resize + inferensi
        img_array = preprocess_image(contents)

        predictions = model.predict(img_array, verbose=0)
        processing_ms = (time.perf_counter() - t0) * 1000
        predicted_idx = np.argmax(predictions[0])
        top_prob = np.max(predictions[0])   # float32, dibandingkan dengan presisi yang sama seperti di notebook
        confidence = float(top_prob)

        detail_probs = {
            CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASSES))
        }

        # --- CONFIDENCE GATE ---
        if top_prob < np.float32(THRESHOLD):
            final_prediction = "Unidentified Object (Cek Manual)"
        else:
            final_prediction = CLASSES[predicted_idx]
        # ------------------------------

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": final_prediction, # Gunakan variabel final_prediction
            "confidence_percent": round(confidence * 100, 2),
            "detail_probabilities": detail_probs,
            "processing_ms": round(processing_ms, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server saat memproses gambar: {str(e)}")


@app.get("/health")
def health():
    # Dipakai skrip ukur_deploy.py untuk memastikan bobot baru dan threshold sudah aktif
    return {
        "model_loaded": model is not None,
        "weights_file": WEIGHTS_FILE,
        "threshold": THRESHOLD,
        "tensorflow": tf.__version__,
        "device": DEVICE
    }


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

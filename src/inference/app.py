from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

model = tf.keras.models.load_model("models/cnn_baseline.h5")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    start = time.time()
    img = Image.open(file.file).convert("RGB").resize((224,224))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(arr)[0][0]
    latency = time.time() - start
    logging.info(f"Prediction latency={latency:.3f}s")
    return {"class": "dog" if pred>0.5 else "cat", "probability": float(pred)}

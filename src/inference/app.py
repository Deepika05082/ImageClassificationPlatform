from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
import io
from cnn_baseline import CNNBaseline
import torchvision.transforms as T

app = FastAPI()

model = CNNBaseline()
model.load_state_dict(torch.load("models/cnn_baseline.h5"))
model.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    outputs = model(tensor)
    _, pred = torch.max(outputs, 1)
    label = "Cat" if pred.item() == 0 else "Dog"
    return {"class": label, "probability": torch.softmax(outputs,1).max().item()}

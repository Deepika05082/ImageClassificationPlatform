from src.inference.app import model, transform
from PIL import Image

def test_inference():
    img = Image.open("data/PetImages/Cat/1.jpg").convert("RGB")
    tensor = transform(img).unsqueeze(0)
    outputs = model(tensor)
    assert outputs.shape[1] == 2

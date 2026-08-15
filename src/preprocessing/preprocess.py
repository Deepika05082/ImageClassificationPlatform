import torchvision.transforms as T
from PIL import Image
import os

# Define preprocessing + augmentation
transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

def preprocess(img_path: str):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def preprocess_dataset(input_dir: str, output_dir: str):
    """Preprocess all images in dataset and save tensors."""
    os.makedirs(output_dir, exist_ok=True)
    for cls in ["Cat", "Dog"]:
        in_path = os.path.join(input_dir, cls)
        out_path = os.path.join(output_dir, cls)
        os.makedirs(out_path, exist_ok=True)
        for file in os.listdir(in_path):
            try:
                tensor = preprocess(os.path.join(in_path, file))
                torch.save(tensor, os.path.join(out_path, f"{file}.pt"))
            except Exception:
                continue

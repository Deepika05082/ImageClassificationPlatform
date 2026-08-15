import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
from cnn_baseline import CNNBaseline

def train_model():
    mlflow.start_run()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train_loader = DataLoader(
        datasets.ImageFolder("data/PetImages/train", transform=transform),
        batch_size=32, shuffle=True
    )

    model = CNNBaseline()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        mlflow.log_metric("loss", loss.item())

    torch.save(model.state_dict(), "models/cnn_baseline.h5")
    mlflow.log_artifact("models/cnn_baseline.h5")
    mlflow.end_run()

if __name__ == "__main__":
    train_model()

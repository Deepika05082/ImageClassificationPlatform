# Cats vs Dogs MLOps Pipeline 🐾

## Overview
End-to-end pipeline for binary image classification (Cats vs Dogs) using Docker, Minikube, and GitHub Actions.

## Steps
1. **Dataset**: Place Kaggle PetImages dataset under `data/PetImages/`.
2. **Preprocessing**: Run `src/preprocessing/preprocess.py` to prepare 224x224 RGB images.
3. **Training**: Run `python src/models/train.py` → saves `models/cnn_baseline.h5`.
4. **Inference Service**: Start FastAPI locally:
   ```bash
   uvicorn src.inference.app:app --reload

from src.preprocessing.preprocess import preprocess
import os

def test_preprocess():
    img = preprocess("data/PetImages/Cat/1.jpg")
    assert img.shape == (3,224,224)

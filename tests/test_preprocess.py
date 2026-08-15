from src.preprocessing.preprocess import preprocess_image
from PIL import Image
import numpy as np

def test_preprocess_shape(tmp_path):
    img = Image.new("RGB", (300,300))
    test_file = tmp_path / "test.jpg"
    img.save(test_file)
    arr = preprocess_image(str(test_file))
    assert arr.shape == (224,224,3)
    assert np.max(arr) <= 1.0

from PIL import Image
import numpy as np
import os
import tensorflow as tf

def preprocess_image(path, size=(224,224)):
    """Load and preprocess a single image file."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img) / 255.0
    return arr

def load_dataset(base_dir, size=(224,224)):
    """Load cats and dogs dataset from folder structure."""
    images, labels = [], []
    for label, folder in enumerate(["Cat", "Dog"]):
        folder_path = os.path.join(base_dir, folder)
        for file in os.listdir(folder_path):
            try:
                arr = preprocess_image(os.path.join(folder_path, file), size)
                images.append(arr)
                labels.append(label)
            except Exception:
                continue
    return np.array(images), np.array(labels)

def create_tf_dataset(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

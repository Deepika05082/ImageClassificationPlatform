import numpy as np
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from cnn_baseline import build_model
from preprocessing.preprocess import load_dataset, create_tf_dataset

# Load data
images, labels = load_dataset("data/PetImages")
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

train_ds = create_tf_dataset(X_train, y_train)
val_ds = create_tf_dataset(X_val, y_val)

# Build model
model = build_model()

# MLflow tracking
mlflow.start_run()
mlflow.log_param("epochs", 5)
mlflow.log_param("batch_size", 32)

history = model.fit(train_ds, validation_data=val_ds, epochs=5)

val_acc = history.history['val_accuracy'][-1]
mlflow.log_metric("val_accuracy", val_acc)

model.save("models/cnn_baseline.h5")
mlflow.log_artifact("models/cnn_baseline.h5")

mlflow.end_run()

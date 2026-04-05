from src.eda import run_eda
from src.train import train_models
from src.evaluate import evaluate
import tensorflow as tf


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

run_eda()
test_data = train_models()

model = tf.keras.models.load_model("artifacts/models/resnet.h5")
evaluate(model, test_data)

print("PIPELINE FINISHED SUCCESSFULLY")

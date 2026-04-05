import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src.gradcam import generate_gradcam

model = tf.keras.models.load_model("artifacts/models/resnet.h5")
labels = {0: "COVID", 1: "Normal", 2: "Viral Pneumonia"}

st.title("COVID-19 X-ray Diagnosis")

file = st.file_uploader("Upload X-ray", ["jpg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    img_resized = img.resize((224,224))
    arr = np.array(img_resized)/255.0
    arr = arr.reshape(1,224,224,3)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))

    st.success(f"Prediction: {labels[idx]}")
    st.write("Confidence:", float(preds[idx]))

    overlay = generate_gradcam(model, arr, np.array(img))
    st.image(overlay, caption="Grad-CAM Explanation", use_column_width=True)

# app.py

import mlflow
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
import yaml

from PIL import Image

# =====================================================
# CONFIG
# =====================================================

IMG_SIZE = 224

CLASS_NAMES = [
    'covid',
    'normal',
    'viral pneumonia'
]

# MODEL_PATH = "models/chest_xray_model.keras"

# =====================================================
# LOAD MODEL
# =====================================================

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_model():
    # MODEL_DIR = os.path.join("models", "mlruns")
    mlruns_path = os.path.abspath(os.path.join("models", "mlruns"))
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    with open("models/model_metrics.json", "r") as f:
        model_metrics = json.load(f)
    run_id = model_metrics["run_id"]
    experiment_dirs = os.listdir(mlruns_path)

    experiment_id = None

    for exp in experiment_dirs:
        possible_run = os.path.join(
                                        mlruns_path,
                                        exp,
                                        run_id
                                    )
        if os.path.exists(possible_run):
            experiment_id = exp
            break

    if experiment_id is None:
        raise Exception("Run ID not found")

    # -------------------------------------------------
    # Outputs directory
    # -------------------------------------------------

    outputs_dir = os.path.join(
                                    mlruns_path,
                                    experiment_id,
                                    run_id,
                                    "outputs"
                                )

    output_folders = os.listdir(outputs_dir)

    yaml_path = os.path.join(
                                outputs_dir,
                                output_folders[0],
                                "meta.yaml"
                            )

    # -------------------------------------------------
    # Read YAML
    # -------------------------------------------------

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    model_id = meta["destination_id"]

    model_path = os.path.join(
                                mlruns_path,
                                experiment_id,
                                "models",
                                model_id,
                                "artifacts",
                                "data",
                                "model.keras"
                            )
    # model = mlflow.sklearn.load_model(model_path)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("Chest X-ray Classification")

st.write(
    "Upload a Chest X-ray image to classify:\n"
    "- COVID\n"
    "- NORMAL\n"
    "- VIRAL PNEUMONIA"
)

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    # ================================================
    # PREPROCESS
    # ================================================

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.efficientnet.preprocess_input(
        img_array
    )

    # ================================================
    # PREDICT
    # ================================================

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(np.max(prediction))

    # ================================================
    # DISPLAY
    # ================================================

    st.subheader(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2%}")

    # ================================================
    # PROBABILITIES
    # ================================================

    st.subheader("Class Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):

        st.write(
            f"{class_name}: "
            f"{prediction[0][i]:.2%}"
        )
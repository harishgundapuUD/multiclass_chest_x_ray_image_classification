# train.py

import os
import json
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

TRAIN_DIR = "datasets/train"
TEST_DIR = "datasets/test"

# MODEL_SAVE_PATH = "models/chest_xray_model.keras"

NUM_CLASSES = 3

# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs("models", exist_ok=True)

# =========================================================
# GPU MEMORY GROWTH (OPTIONAL)
# =========================================================

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,

    rotation_range=15,
    zoom_range=0.15,

    width_shift_range=0.1,
    height_shift_range=0.1,

    brightness_range=[0.8, 1.2],

    horizontal_flip=True,

    validation_split=0.2,

    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# =========================================================
# DATA LOADERS
# =========================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================================================
# CLASS NAMES
# =========================================================

class_names = list(train_generator.class_indices.keys())

print("\nClasses:")
print(class_names)

# =========================================================
# LOAD PRETRAINED MODEL
# =========================================================

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# =========================================================
# FREEZE BASE MODEL
# =========================================================

base_model.trainable = False

# Freeze BatchNorm layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# =========================================================
# BUILD MODEL
# =========================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.3),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(NUM_CLASSES, activation='softmax')
])

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1
    )
]

# =========================================================
# START MLFLOW
# =========================================================
mlruns_path = os.path.abspath(os.path.join("models", "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
# mlflow.set_experiment(model_type)
mlflow.set_experiment("Chest_Xray_Classification")

model_metrics_path = os.path.join("models", "model_metrics.json")

with mlflow.start_run() as run:
    run_id = run.info.run_id

    # =====================================================
    # LOG PARAMETERS
    # =====================================================

    mlflow.log_param("model", "EfficientNetB0")
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("initial_epochs", INITIAL_EPOCHS)
    mlflow.log_param("fine_tune_epochs", FINE_TUNE_EPOCHS)

    # =====================================================
    # STAGE 1 TRAINING
    # =====================================================

    print("\n====================================")
    print("STAGE 1 TRAINING")
    print("====================================\n")

    history_stage1 = model.fit(
                                train_generator,
                                validation_data=val_generator,
                                epochs=INITIAL_EPOCHS,
                                callbacks=callbacks
                            )

    # =====================================================
    # FINE-TUNING
    # =====================================================

    print("\n====================================")
    print("STARTING FINE-TUNING")
    print("====================================\n")

    base_model.trainable = True

    # Freeze most layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Keep BatchNorm frozen
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Recompile with lower LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_stage2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    print("\n====================================")
    print("EVALUATION")
    print("====================================\n")

    predictions = model.predict(test_generator)

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
                                    y_true,
                                    y_pred,
                                    target_names=class_names,
                                    output_dict=True
                                )

    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    roc_auc = roc_auc_score(
                                tf.keras.utils.to_categorical(y_true, NUM_CLASSES),
                                predictions,
                                multi_class='ovr'
                            )

    # =====================================================
    # PRINT METRICS
    # =====================================================

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1_score:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    # =====================================================
    # LOG METRICS TO MLFLOW
    # =====================================================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("roc_auc", roc_auc)

    # =====================================================
    # PLOT CONFUSION MATRIX
    # =====================================================

    plt.figure(figsize=(8, 6))

    sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names
                )

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    # Save confusion matrix
    plt.savefig("models/confusion_matrix.png")

    # plt.show()

    mlflow.log_artifact("models/confusion_matrix.png")

    # =====================================================
    # SAVE MODEL
    # =====================================================

    # model.save(MODEL_SAVE_PATH)

    # =====================================================
    # LOG MODEL TO MLFLOW
    # =====================================================

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model"
    )

    results = {
                "run_id": run_id,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1_score),
                "roc-auc": float(roc_auc),
            }
    
    with open(model_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n====================================")
    print("MODEL SAVED")
    print("====================================")

    # print(f"\nSaved at: {MODEL_SAVE_PATH}")


print("\nTRAINING COMPLETED")
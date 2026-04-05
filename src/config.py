import os
import numpy as np
import tensorflow as tf

# ===============================
# Reproducibility
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===============================
# Training configuration
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 1   # sanity run; increase on GPU

# ===============================
# Base paths
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "datasets")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_DIR   = os.path.join(ARTIFACTS_DIR, "models")
METRIC_DIR  = os.path.join(ARTIFACTS_DIR, "metrics")
EDA_DIR     = os.path.join(ARTIFACTS_DIR, "eda")
GRADCAM_DIR = os.path.join(ARTIFACTS_DIR, "gradcam")

# ===============================
# Directory bootstrap (CRITICAL)
# ===============================
REQUIRED_DIRS = [
    DATA_DIR,
    TRAIN_DIR,
    TEST_DIR,
    ARTIFACTS_DIR,
    MODEL_DIR,
    METRIC_DIR,
    EDA_DIR,
    GRADCAM_DIR,
]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)

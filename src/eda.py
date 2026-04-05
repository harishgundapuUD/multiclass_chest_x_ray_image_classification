import os
import matplotlib.pyplot as plt
from src.config import TRAIN_DIR, EDA_DIR

def run_eda():
    class_counts = {}

    for cls in os.listdir(TRAIN_DIR):
        cls_path = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_path):
            class_counts[cls] = len(os.listdir(cls_path))

    plt.figure()
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Training Class Distribution")
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"))
    plt.close()

    print("EDA completed:", class_counts)

import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm

from src.config import TRAIN_DIR, ARTIFACTS_DIR, IMG_SIZE

BALANCED_TRAIN_DIR = os.path.join(ARTIFACTS_DIR, "balanced_train")

def augment_image(img):
    # rotation
    angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # brightness jitter
    factor = random.uniform(0.7, 1.3)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    return img


def build_balanced_dataset():
    if os.path.exists(BALANCED_TRAIN_DIR):
        shutil.rmtree(BALANCED_TRAIN_DIR)
    os.makedirs(BALANCED_TRAIN_DIR, exist_ok=True)

    class_counts = {
        cls: len(os.listdir(os.path.join(TRAIN_DIR, cls)))
        for cls in os.listdir(TRAIN_DIR)
    }

    max_count = max(class_counts.values())

    print("Original class distribution:", class_counts)

    for cls, count in class_counts.items():
        src_cls = os.path.join(TRAIN_DIR, cls)
        dst_cls = os.path.join(BALANCED_TRAIN_DIR, cls)
        os.makedirs(dst_cls, exist_ok=True)

        images = os.listdir(src_cls)

        # Copy originals
        for img_name in images:
            shutil.copy(
                os.path.join(src_cls, img_name),
                os.path.join(dst_cls, img_name)
            )

        # Augment minority classes
        needed = max_count - count
        if needed <= 0:
            continue

        for i in tqdm(range(needed), desc=f"Augmenting {cls}"):
            img_name = random.choice(images)
            img_path = os.path.join(src_cls, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)
            aug = augment_image(img)

            out_name = f"aug_{i}_{img_name}"
            cv2.imwrite(os.path.join(dst_cls, out_name), aug)

    print("Balanced dataset created at:", BALANCED_TRAIN_DIR)

    return BALANCED_TRAIN_DIR

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.config import *

def build_resnet(num_classes):
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(base.input, output)
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

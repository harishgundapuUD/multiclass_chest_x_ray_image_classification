# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from src.config import *

# def get_data():
#     train_gen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=20,
#         horizontal_flip=True
#     )

#     test_gen = ImageDataGenerator(rescale=1./255)

#     train_data = train_gen.flow_from_directory(
#         TRAIN_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="categorical"
#     )

#     test_data = test_gen.flow_from_directory(
#         TEST_DIR,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode="categorical",
#         shuffle=False
#     )

#     return train_data, test_data



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import *
from src.balanced_augmentation import build_balanced_dataset

def get_data():
    balanced_train_dir = build_balanced_dataset()

    train_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        balanced_train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, test_data

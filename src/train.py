from src.data_loader import get_data
from src.baseline_cnn import build_baseline
from src.transfer_model import build_resnet
from src.config import MODEL_DIR, EPOCHS

def train_models():
    train_data, test_data = get_data()
    num_classes = train_data.num_classes

    baseline = build_baseline(num_classes)
    baseline.fit(train_data, epochs=EPOCHS, validation_data=test_data)
    baseline.save(f"{MODEL_DIR}/baseline.h5")

    resnet = build_resnet(num_classes)
    resnet.fit(train_data, epochs=EPOCHS, validation_data=test_data)
    resnet.save(f"{MODEL_DIR}/resnet.h5")

    return test_data

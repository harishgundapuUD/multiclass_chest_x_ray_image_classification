import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import METRIC_DIR

def evaluate(model, test_data):
    y_true = test_data.classes
    y_prob = model.predict(test_data)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "roc_auc": roc_auc_score(y_true, y_prob, multi_class="ovr"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    with open(f"{METRIC_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed")

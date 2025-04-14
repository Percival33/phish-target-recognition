import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np


def calculate_metrics(cls_true, cls_pred, targets_true, targets_pred):
    cls_true = np.array(cls_true)
    cls_pred = np.array(cls_pred)

    class_metrics = {
        "f1_weighted": f1_score(cls_true, cls_pred, average="weighted"),
        "roc_auc": roc_auc_score(cls_true, cls_pred),
        "mcc": matthews_corrcoef(cls_true, cls_pred),
        "precision": precision_score(cls_true, cls_pred, average="macro"),
        "recall": recall_score(cls_true, cls_pred, average="macro"),
    }

    all_targets = list(targets_true) + list(targets_pred)
    le = LabelEncoder()
    le.fit(all_targets)

    targets_true_encoded = le.transform(targets_true)
    targets_pred_encoded = le.transform(targets_pred)

    # Repp_TP: Number of correctly reported true phishing webpages
    Repp_TP = np.sum((cls_true == 1) & (cls_pred == 1))

    # Idp: Number of correctly reported phishing webpages with brand reported correctly
    Idp = np.sum((cls_true == 1) & (cls_pred == 1) & (targets_true_encoded == targets_pred_encoded))

    target_metrics = {
        "target_f1_micro": f1_score(
            targets_true_encoded, targets_pred_encoded, average="micro"
        ),
        "target_f1_macro": f1_score(
            targets_true_encoded, targets_pred_encoded, average="macro"
        ),
        "target_f1_weighted": f1_score(
            targets_true_encoded, targets_pred_encoded, average="weighted"
        ),
        "target_mcc": matthews_corrcoef(targets_true_encoded, targets_pred_encoded),
        "precision": precision_score(
            targets_true_encoded, targets_pred_encoded, average="macro"
        ),
        "recall": recall_score(
            targets_true_encoded, targets_pred_encoded, average="macro"
        ),
        "identification_rate": (
            # Identification rate = Idp / Repp_TP
            Idp / Repp_TP if Repp_TP > 0 else 0
        ),
    }

    return class_metrics, target_metrics

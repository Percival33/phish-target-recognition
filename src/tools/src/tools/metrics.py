import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# Import PROJ_ROOT from the config module
from .config import PROJ_ROOT

TARGET_MAPPINGS_PATH = PROJ_ROOT / "target_mappings.json"


def load_and_prepare_mappings(file_path: Path) -> dict[str, str]:
    """Loads target mappings from JSON and prepares a lookup dictionary."""
    if not file_path.exists():
        print(
            f"Warning: Mapping file not found at {file_path}. Proceeding without normalization."
        )
        return {}

    try:
        with open(file_path, "r") as f:
            raw_mappings = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return {}

    normalized_map = {}
    for canonical_name, aliases in raw_mappings.items():
        # Ensure the canonical name maps to itself (case-insensitively)
        normalized_map[canonical_name.lower()] = canonical_name
        for alias in aliases:
            normalized_map[alias.lower()] = canonical_name
    return normalized_map


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

    normalized_targets_true = list(targets_true)
    normalized_targets_pred = list(targets_pred)

    mapping = None
    if TARGET_MAPPINGS_PATH.exists():
        mapping = load_and_prepare_mappings(TARGET_MAPPINGS_PATH)

    if mapping:
        # Only normalize if mapping was loaded successfully
        normalized_targets_true = [
            mapping.get(str(t).lower(), str(t)) for t in targets_true
        ]
        normalized_targets_pred = [
            mapping.get(str(t).lower(), str(t)) for t in targets_pred
        ]

    all_targets = list(normalized_targets_true) + list(normalized_targets_pred)
    le = LabelEncoder()
    le.fit(all_targets)

    targets_true_encoded = le.transform(normalized_targets_true)
    targets_pred_encoded = le.transform(normalized_targets_pred)

    # Repp_TP: Number of correctly reported true phishing webpages
    Repp_TP = np.sum((cls_true == 1) & (cls_pred == 1))

    # Idp: Number of correctly reported phishing webpages with brand reported correctly
    Idp = np.sum(
        (cls_true == 1)
        & (cls_pred == 1)
        & (targets_true_encoded == targets_pred_encoded)
    )

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

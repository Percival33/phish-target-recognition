from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


def calculate_metrics(cls_true, cls_pred, targets_true, targets_pred):
    class_metrics = {
        "f1_weighted": f1_score(cls_true, cls_pred, average='weighted'),
        "roc_auc": roc_auc_score(cls_true, cls_pred),
        "mcc": matthews_corrcoef(cls_true, cls_pred),
        "precision": precision_score(cls_true, cls_pred, average='macro'),
        "recall": recall_score(cls_true, cls_pred, average='macro'),
    }

    all_targets = list(set(targets_true + targets_pred))
    le = LabelEncoder()
    le.fit(all_targets)

    targets_true_encoded = le.transform(targets_true)
    targets_pred_encoded = le.transform(targets_pred)

    target_metrics = {
        "target_f1_micro": f1_score(targets_true_encoded, targets_pred_encoded, average='micro'),
        "target_f1_macro": f1_score(targets_true_encoded, targets_pred_encoded, average='macro'),
        "target_f1_weighted": f1_score(targets_true_encoded, targets_pred_encoded, average='weighted'),
        "target_mcc": matthews_corrcoef(targets_true_encoded, targets_pred_encoded),
        "precision": precision_score(cls_true, cls_pred, average='macro'),
        "recall": recall_score(cls_true, cls_pred, average='macro'),
    }

    return class_metrics, target_metrics

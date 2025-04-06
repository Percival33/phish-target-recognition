import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import argparse


def process_and_evaluate(csv1, csv2, plot=False):
    # Load CSV files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Concatenate dataframes
    df = pd.concat([df1, df2], ignore_index=True)

    # Fill NaN values in pp_target with 0
    df["pp_class"] = df["pp_class"].fillna(0)

    # Extract predictions and true values
    y_true = df["true_class"]
    y_pred = df["pp_class"]

    # Compute evaluation metrics
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"{20 * '='} Evaluation Metrics {20 * '='}")
    print(f"{20 * '='} benign / phish  {20 * '='}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")

    print(f"{20 * '='} target  {20 * '='}")
    le = LabelEncoder()
    df["pp_target"] = df["pp_target"].fillna("benign")

    le.fit([*list(df["true_target"]), *list(df["pp_target"])])

    y_true = le.transform(df["true_target"])
    y_pred = le.transform(df["pp_target"])

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)
    # roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"MCC: {mcc:.4f}")
    # print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Phishpedia CSVs to get metrics")
    parser.add_argument("csv1", help="Path to the Phishpedia CSV 1 file")
    parser.add_argument("csv2", help="Path to the Phishpedia CSV 2 file")
    parser.add_argument("--plot", action="store_true", help="Show ROC curve plot")
    args = parser.parse_args()

    process_and_evaluate(args.csv1, args.csv2, args.plot)

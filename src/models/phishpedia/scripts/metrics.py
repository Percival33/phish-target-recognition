import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tools.metrics import calculate_metrics
import argparse


def process_and_evaluate(path_to_csv, plot=False):
    # Load CSV files
    df = pd.read_csv(path_to_csv)
    # Fill NaN values in pp_target with 0
    df["pp_class"] = df["pp_class"].fillna(0)

    class_metrics, target_metrics = calculate_metrics(
        cls_true=df["true_class"],
        cls_pred=df["pp_class"],
        targets_true=df["true_target"],
        targets_pred=df["pp_target"],
    )

    print("Class metrics:")
    for metric, value in class_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nTarget metrics:")
    for metric, value in target_metrics.items():
        print(f"{metric}: {value:.4f}")

    if plot:

        fpr, tpr, _ = roc_curve(df["true_class"], df["pp_class"])
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(df["true_class"], df["pp_class"]):.4f})"
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Phishpedia CSVs to get metrics")
    parser.add_argument("csv", help="Path to the Phishpedia CSV 1 file")
    # parser.add_argument("csv2", help="Path to the Phishpedia CSV 2 file")
    parser.add_argument("--plot", action="store_true", help="Show ROC curve plot")
    args = parser.parse_args()

    # # Concatenate dataframes
    # df = pd.concat([df1, df2], ignore_index=True)
    # process_and_evaluate(args.csv1, args.csv2, args.plot)
    process_and_evaluate(args.csv, args.plot)

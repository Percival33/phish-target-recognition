import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from tools.config import setup_logging
from tools.metrics import calculate_metrics


def evaluate_from_csv(
    csv_path: str,
    plot: bool = False,
    out_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    df = pd.read_csv(csv_path)

    required = ["true_target", "true_class", "pp_class", "pp_target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    for col in required:
        if df[col].isna().any():
            logger.error(f"Found NaN values in required column: {col}")
            raise ValueError(f"Found NaN values in required column: {col}")

    df["pp_class"] = df["pp_class"].astype(int)
    df["true_class"] = df["true_class"].astype(int)
    df["true_class"] = pd.to_numeric(df["true_class"], errors="coerce")
    df.loc[df["true_class"] == 0, "true_target"] = "benign"

    class_metrics, target_metrics = calculate_metrics(
        cls_true=df["true_class"],
        cls_pred=df["pp_class"],
        targets_true=df["true_target"],
        targets_pred=df["pp_target"],
    )

    logging.info(f"Class metrics: {class_metrics}")
    logging.info(f"Target metrics: {target_metrics}")

    print("\n" + "=" * 50)
    print("VISUALPHISHNET MODEL EVALUATION RESULTS")
    print("=" * 50)

    print("\nClass Classification Metrics:")
    print("-" * 30)
    for metric_name, value in class_metrics.items():
        print(f"{metric_name:15}: {value:.4f}")

    print("\nTarget Identification Metrics:")
    print("-" * 30)
    for metric_name, value in target_metrics.items():
        print(f"{metric_name:20}: {value:.4f}")

    cm = confusion_matrix(df["true_class"], df["pp_class"])
    # cm = confusion_matrix(df["true_class"], df["pp_class"], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(out / "confusion_matrix.csv")

        def _latex(metrics: dict) -> str:
            items = [f"{k}={metrics[k]:.4f}" for k in sorted(metrics.keys())]
            return (
                "\\begin{tabular}{ll}\n"
                "\\toprule\n"
                "ID & Metrics \\ \\ \n"
                "\\midrule\n"
                f"Phishpedia & {', '.join(items)} \\ \n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
            )

        (out / "class_metrics.tex").write_text(_latex(class_metrics))
        (out / "target_metrics.tex").write_text(_latex(target_metrics))

    if plot:
        try:
            auc_score = roc_auc_score(df["true_class"], df["pp_class"])
            fpr, tpr, _ = roc_curve(df["true_class"], df["pp_class"])
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error("Failed to plot ROC: %s", e)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate from exported CSV (true_target,true_class,pp_class,pp_target)"
    )
    parser.add_argument("csv", type=str, help="Path to the exported CSV")
    parser.add_argument("--plot", action="store_true", help="Show ROC curve plot")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for confusion matrix and LaTeX",
    )
    args = parser.parse_args()

    evaluate_from_csv(args.csv, args.plot, args.out)

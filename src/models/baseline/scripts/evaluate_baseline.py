#!/usr/bin/env python3
"""
Evaluation script for baseline model results.

This script takes a CSV file (output from query.py) and calculates classification
and target identification metrics using the tools.metrics module.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from sklearn.metrics import roc_auc_score, roc_curve
from tools.config import setup_logging
from tools.metrics import calculate_metrics

import matplotlib.pyplot as plt
import pandas as pd

setup_logging()
logger = logging.getLogger(__name__)


def process_and_evaluate(
    csv_path: Path, plot_roc: bool = False, out_dir: Path = None
) -> None:
    """
    Process the CSV file and evaluate the baseline model performance.

    Args:
        csv_path: Path to the CSV file containing query results
        plot_roc: Whether to plot the ROC curve
        out_dir: Optional output directory to save results
    """
    try:
        logger.info(f"Loading results from {csv_path}")
        df = pd.read_csv(csv_path)

        required_columns = [
            "baseline_class",
            "baseline_target",
            "true_class",
            "true_target",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)

        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Columns: {df.columns.tolist()}")

        critical_columns = [
            "baseline_class",
            "true_class",
            "true_target",
            "baseline_target",
        ]
        for col in critical_columns:
            if df[col].isna().any():
                logger.error(f"Found NaN values in critical column: {col}")
                raise ValueError(f"Found NaN values in critical column: {col}")

        df["baseline_class"] = df["baseline_class"].astype(int)
        df["true_class"] = df["true_class"].astype(int)
        df.loc[df["true_class"] == 0, "true_target"] = "benign"

        logger.info("Calculating metrics...")
        class_metrics, target_metrics = calculate_metrics(
            cls_true=df["true_class"],
            cls_pred=df["baseline_class"],
            targets_true=df["true_target"],
            targets_pred=df["baseline_target"],
        )

        print("\n" + "=" * 50)
        print("BASELINE MODEL EVALUATION RESULTS")
        print("=" * 50)

        print("\nClass Classification Metrics:")
        print("-" * 30)
        for metric_name, value in class_metrics.items():
            print(f"{metric_name:15}: {value:.4f}")

        print("\nTarget Identification Metrics:")
        print("-" * 30)
        for metric_name, value in target_metrics.items():
            print(f"{metric_name:20}: {value:.4f}")

        if out_dir:
            save_results(out_dir, class_metrics, target_metrics, df, csv_path.stem)

        if plot_roc:
            logger.info("Generating ROC curve...")
            plot_roc_curve(df["true_class"], df["baseline_class"], out_dir)

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)


def save_results(
    out_dir: Path,
    class_metrics: dict,
    target_metrics: dict,
    df: pd.DataFrame,
    run_name: str,
) -> None:
    """
    Save evaluation results to files.

    Args:
        out_dir: Output directory path
        class_metrics: Classification metrics dictionary
        target_metrics: Target identification metrics dictionary
        df: DataFrame with results
        run_name: Name for the run (from input filename)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = {
        "run_name": run_name,
        "class_metrics": class_metrics,
        "target_metrics": target_metrics,
        "summary": {
            "total_samples": len(df),
            "phishing_samples": int(df["true_class"].sum()),
            "benign_samples": int((df["true_class"] == 0).sum()),
        },
    }

    metrics_file = out_dir / f"{run_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_file}")

    results_file = out_dir / f"{run_name}_detailed_results.csv"
    df.to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to: {results_file}")

    summary_data = []
    for metric_name, value in class_metrics.items():
        summary_data.append(
            {"metric_type": "class", "metric_name": metric_name, "value": value}
        )
    for metric_name, value in target_metrics.items():
        summary_data.append(
            {"metric_type": "target", "metric_name": metric_name, "value": value}
        )

    summary_df = pd.DataFrame(summary_data)
    summary_file = out_dir / f"{run_name}_summary_metrics.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary metrics to: {summary_file}")


def plot_roc_curve(y_true, y_pred, out_dir: Path = None):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels or scores
        out_dir: Optional output directory to save the plot
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {auc_score:.4f})",
        )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_file = out_dir / "roc_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curve to: {plot_file}")

        plt.show()

    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model results from CSV file"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file containing query results (output from query.py)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate ROC curve plot")
    parser.add_argument(
        "--out",
        type=str,
        help="Output directory to save evaluation results (metrics, plots, etc.)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    if not csv_path.suffix.lower() == ".csv":
        logger.error(f"File must be a CSV file: {csv_path}")
        sys.exit(1)

    out_dir = None
    if args.out:
        out_dir = Path(args.out)
        logger.info(f"Results will be saved to: {out_dir}")

    process_and_evaluate(csv_path, args.plot, out_dir)


if __name__ == "__main__":
    main()

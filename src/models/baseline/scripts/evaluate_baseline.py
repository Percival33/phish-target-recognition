#!/usr/bin/env python3
"""
Evaluation script for baseline model results.

This script takes a CSV file (output from query.py) and calculates classification
and target identification metrics using the tools.metrics module.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from tools.metrics import calculate_metrics
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def process_and_evaluate(csv_path: Path, plot_roc: bool = False) -> None:
    """
    Process the CSV file and evaluate the baseline model performance.

    Args:
        csv_path: Path to the CSV file containing query results
        plot_roc: Whether to plot the ROC curve
    """
    try:
        # Load the CSV file
        logger.info(f"Loading results from {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate required columns
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

        # Handle missing values in target columns
        # Fill NaN values with 'unknown' placeholder
        df["baseline_target"] = df["baseline_target"].fillna("unknown")
        df["true_target"] = df["true_target"].fillna("unknown")

        # Check for remaining NaN values in class columns
        if df["baseline_class"].isna().any():
            logger.warning("Found NaN values in baseline_class column")
            df["baseline_class"] = df["baseline_class"].fillna(0)

        if df["true_class"].isna().any():
            logger.warning("Found NaN values in true_class column")
            df["true_class"] = df["true_class"].fillna(0)

        # Calculate metrics using the tools.metrics function
        logger.info("Calculating metrics...")
        class_metrics, target_metrics = calculate_metrics(
            cls_true=df["true_class"],
            cls_pred=df["baseline_class"],
            targets_true=df["true_target"],
            targets_pred=df["baseline_target"],
        )

        # Output metrics in a user-friendly format
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

        # Plot ROC curve if requested
        if plot_roc:
            logger.info("Generating ROC curve...")
            plot_roc_curve(df["true_class"], df["baseline_class"])

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)


def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels or scores
    """
    try:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)

        # Create the plot
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

    args = parser.parse_args()

    # Convert to Path object and validate
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    if not csv_path.suffix.lower() == ".csv":
        logger.error(f"File must be a CSV file: {csv_path}")
        sys.exit(1)

    # Run the evaluation
    process_and_evaluate(csv_path, args.plot)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation script for VisualPhishNet model results.

This script takes a CSV file with VisualPhishNet results and calculates classification
and target identification metrics using the tools.metrics module.

Expected CSV columns:
- file: filename of the processed image
- vp_class: predicted class (0 for benign, 1 for phishing)
- vp_distance: distance to closest match
- vp_target: predicted target brand
- closest_file: closest matching file (optional)
- true_class: actual class (0 for benign, 1 for phishing)
- true_target: actual target brand
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tools.config import setup_logging
from tools.metrics import calculate_metrics

setup_logging()
logger = logging.getLogger(__name__)


def process_and_evaluate(csv_path: Path, plot_roc: bool = False, out_dir: Path = None) -> None:
    """
    Process the CSV file and evaluate the VisualPhishNet model performance.

    Args:
        csv_path: Path to the CSV file containing VisualPhishNet results
        plot_roc: Whether to plot the ROC curve
        out_dir: Optional output directory to save results
    """
    try:
        # Load the CSV file
        logger.info(f"Loading VisualPhishNet results from {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = [
            "vp_class",
            "vp_target",
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

        # Display column sample data for debugging
        logger.info("Sample data preview:")
        for col in required_columns:
            sample_values = df[col].head(5).tolist()
            logger.info(f"  {col}: {sample_values}")

        # Check for NaN values in critical class columns before processing
        critical_columns = ["vp_class", "true_class", "vp_target", "true_target"]
        for col in critical_columns:
            if df[col].isna().any():
                logger.error(f"Found NaN values in critical column: {col}")
                raise ValueError(f"Found NaN values in critical column: {col}")

        # Ensure benign samples have correct true_target assigned
        df["vp_class"] = df["vp_class"].astype(int)
        df["true_class"] = df["true_class"].astype(int)
        df["true_class"] = pd.to_numeric(df["true_class"], errors="coerce")
        df.loc[df["true_class"] == 0, "true_target"] = "benign"

        # Log class distribution for debugging
        logger.info("Class distribution:")
        logger.info(f"  True classes: {df['true_class'].value_counts().to_dict()}")
        logger.info(f"  Predicted classes: {df['vp_class'].value_counts().to_dict()}")

        # Log target distribution for debugging
        logger.info("Target distribution (top 10):")
        logger.info(f"  True targets: {df['true_target'].value_counts().head(10).to_dict()}")
        logger.info(f"  Predicted targets: {df['vp_target'].value_counts().head(10).to_dict()}")

        # Calculate metrics using the tools.metrics function
        logger.info("Calculating metrics...")
        class_metrics, target_metrics = calculate_metrics(
            cls_true=df["true_class"],
            cls_pred=df["vp_class"],
            targets_true=df["true_target"],
            targets_pred=df["vp_target"],
        )

        # Output metrics in a user-friendly format
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

        # Additional VisualPhishNet specific analysis
        print("\nDistance Analysis:")
        print("-" * 20)
        if "vp_distance" in df.columns:
            benign_distances = df[df["vp_class"] == 0]["vp_distance"]
            phishing_distances = df[df["vp_class"] == 1]["vp_distance"]

            if len(benign_distances) > 0:
                print(f"Avg distance (benign pred): {benign_distances.mean():.4f}")
                print(f"Std distance (benign pred): {benign_distances.std():.4f}")

            if len(phishing_distances) > 0:
                print(f"Avg distance (phish pred):  {phishing_distances.mean():.4f}")
                print(f"Std distance (phish pred):  {phishing_distances.std():.4f}")

        # Save results if output directory is specified
        if out_dir:
            save_results(out_dir, class_metrics, target_metrics, df, csv_path.stem)

        # Plot ROC curve if requested
        if plot_roc:
            logger.info("Generating ROC curve...")
            plot_roc_curve(df["true_class"], df["vp_class"], out_dir)

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

    # Save metrics as JSON
    metrics_data = {
        "run_name": run_name,
        "model": "VisualPhishNet",
        "class_metrics": class_metrics,
        "target_metrics": target_metrics,
        "summary": {
            "total_samples": len(df),
            "phishing_samples": int(df["true_class"].sum()),
            "benign_samples": int((df["true_class"] == 0).sum()),
            "predicted_phishing": int(df["vp_class"].sum()),
            "predicted_benign": int((df["vp_class"] == 0).sum()),
        },
    }

    # Add distance statistics if available
    if "vp_distance" in df.columns:
        benign_distances = df[df["vp_class"] == 0]["vp_distance"]
        phishing_distances = df[df["vp_class"] == 1]["vp_distance"]

        metrics_data["distance_stats"] = {
            "benign_pred_mean": float(benign_distances.mean()) if len(benign_distances) > 0 else None,
            "benign_pred_std": float(benign_distances.std()) if len(benign_distances) > 0 else None,
            "phishing_pred_mean": float(phishing_distances.mean()) if len(phishing_distances) > 0 else None,
            "phishing_pred_std": float(phishing_distances.std()) if len(phishing_distances) > 0 else None,
        }

    metrics_file = out_dir / f"{run_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_file}")

    # Save detailed results CSV
    results_file = out_dir / f"{run_name}_detailed_results.csv"
    df.to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to: {results_file}")

    # Save summary metrics as CSV for easy comparison
    summary_data = []
    for metric_name, value in class_metrics.items():
        summary_data.append({"metric_type": "class", "metric_name": metric_name, "value": value})
    for metric_name, value in target_metrics.items():
        summary_data.append({"metric_type": "target", "metric_name": metric_name, "value": value})

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
        plt.title("VisualPhishNet - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot if output directory is provided
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_file = out_dir / "visualphishnet_roc_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curve to: {plot_file}")

        plt.show()

    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate VisualPhishNet model results from CSV file")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file containing VisualPhishNet results",
    )
    parser.add_argument("--plot", action="store_true", help="Generate ROC curve plot")
    parser.add_argument(
        "--out",
        type=str,
        help="Output directory to save evaluation results (metrics, plots, etc.)",
    )

    args = parser.parse_args()

    # Convert to Path object and validate
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    if not csv_path.suffix.lower() == ".csv":
        logger.error(f"File must be a CSV file: {csv_path}")
        sys.exit(1)

    # Handle output directory
    out_dir = None
    if args.out:
        out_dir = Path(args.out)
        logger.info(f"Results will be saved to: {out_dir}")

    # Run the evaluation
    process_and_evaluate(csv_path, args.plot, out_dir)


if __name__ == "__main__":
    main()

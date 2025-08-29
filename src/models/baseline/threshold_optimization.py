#!/usr/bin/env python3
"""
Simple threshold optimization for baseline phishing detection.
Grid search over thresholds to find optimal F1 and MCC values.
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging

# Add parent directories to path for tools import
sys.path.append(str(Path(__file__).parent.parent.parent))
from tools.metrics import calculate_metrics
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_validation_data(val_csv_path: str) -> pd.DataFrame:
    """Load validation data from CSV file."""
    val_df = pd.read_csv(val_csv_path)
    logger.info(f"Loaded validation data: {len(val_df)} samples")
    logger.info(f"Columns: {val_df.columns.tolist()}")
    logger.info(f"Class distribution: {val_df['true_class'].value_counts().to_dict()}")
    return val_df


def match_results_with_validation(results_file: str, val_csv_path: str) -> pd.DataFrame:
    """Match query results with validation data to add true classes and targets."""
    # Load results and validation data
    results_df = pd.read_csv(results_file)
    val_df = load_validation_data(val_csv_path)

    # Create mapping from filename to validation info
    val_mapping = {}
    for _, row in val_df.iterrows():
        filename = Path(row["new_path"]).name
        val_mapping[filename] = {
            "true_class": row["true_class"],
            "true_target": row["true_target"],
        }

    # Add true classes and targets to results
    true_classes = []
    true_targets = []

    for _, row in results_df.iterrows():
        filename = row["file"]

        if filename in val_mapping:
            # Use validation data
            true_classes.append(val_mapping[filename]["true_class"])
            true_targets.append(val_mapping[filename]["true_target"])
        else:
            # Fallback: infer from directory structure based on baseline_target
            # If baseline_target is "benign", assume it's from trusted_list (class 0)
            # Otherwise assume it's from phishing (class 1)
            baseline_target = row.get("baseline_target", "unknown")
            if baseline_target == "benign":
                true_classes.append(0)
                true_targets.append("benign")
            else:
                true_classes.append(1)
                true_targets.append(
                    baseline_target
                )  # Use baseline prediction as fallback

    # Add true classes and targets to results dataframe
    results_df["true_class"] = true_classes
    results_df["true_target"] = true_targets

    logger.info(f"Matched {len(results_df)} results with validation data")
    return results_df


def run_query_with_threshold(
    threshold: float, val_csv_path: str, data_base_path: str, index_path: str
) -> bool:
    """Run query.py with specific threshold on validation data using --unknown flag."""
    val_data_base = data_base_path
    phish_data_path = f"{val_data_base}/phishing"
    benign_data_path = f"{val_data_base}/trusted_list"

    phish_output = f"phish_results_threshold_{threshold}.csv"
    benign_output = f"benign_results_threshold_{threshold}.csv"
    combined_output = f"results_threshold_{threshold}.csv"

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    try:
        print("Processing phishing samples...")
        # Run query on phishing samples with --unknown flag (blind evaluation)
        phish_cmd = [
            "uv",
            "run",
            "query.py",
            "--images",
            phish_data_path,
            "--index",
            index_path,
            "--output",
            phish_output,
            "--threshold",
            str(threshold),
            "--unknown",
            "--overwrite",
        ]

        result = subprocess.run(phish_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.error(
                f"Phishing query failed for threshold {threshold}: {result.stderr}"
            )
            return False

        print("  🔍 Processing benign samples...")
        # Run query on trusted_list samples with --unknown flag (blind evaluation)
        benign_cmd = [
            "uv",
            "run",
            "query.py",
            "--images",
            benign_data_path,
            "--index",
            index_path,
            "--output",
            benign_output,
            "--threshold",
            str(threshold),
            "--unknown",
            "--overwrite",
        ]

        result = subprocess.run(benign_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.error(
                f"Benign query failed for threshold {threshold}: {result.stderr}"
            )
            return False

        print("Combining results...")
        # Combine raw results (without true labels)
        phish_df = pd.read_csv(phish_output)
        benign_df = pd.read_csv(benign_output)
        combined_df = pd.concat([phish_df, benign_df], ignore_index=True)

        # Save raw combined results first
        combined_df.to_csv(combined_output, index=False)

        # Match results with validation data to add true labels
        matched_df = match_results_with_validation(combined_output, val_csv_path)
        matched_df.to_csv(combined_output, index=False)

        # Clean up temporary files
        Path(phish_output).unlink(missing_ok=True)
        Path(benign_output).unlink(missing_ok=True)

        logger.info(
            f"Successfully ran query for threshold {threshold} ({len(phish_df)} phish + {len(benign_df)} benign)"
        )
        return True

    except Exception as e:
        logger.error(f"Error running query for threshold {threshold}: {e}")
        return False


def calculate_metrics_from_results(results_file: str) -> tuple:
    """Calculate F1 and MCC from query results, treating benign as 'benign' target."""
    df = pd.read_csv(results_file)

    cls_true = df["true_class"].values
    cls_pred = df["baseline_class"].values
    targets_true = df["true_target"].values.copy()
    targets_pred = df["baseline_target"].values.copy()

    # Treat benign samples as 'benign' target
    targets_true = np.where(cls_true == 0, "benign", targets_true)
    targets_pred = np.where(cls_pred == 0, "benign", targets_pred)

    class_metrics, target_metrics = calculate_metrics(
        cls_true, cls_pred, targets_true, targets_pred
    )

    logger.info(
        f"Class F1: {class_metrics['f1_weighted']:.4f}, Class MCC: {class_metrics['mcc']:.4f}"
    )
    logger.info(f"Target MCC: {target_metrics['target_mcc']:.4f}")

    return (
        class_metrics["f1_weighted"],
        class_metrics["mcc"],
        target_metrics["target_mcc"],
    )


def main():
    """Main threshold optimization function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimize thresholds for baseline phishing detection"
    )
    parser.add_argument(
        "--val-csv",
        default="/home/phish-target-recognition/data_splits/visualphish/phishpedia/val.csv",
        help="Path to validation CSV file",
        dest="val_csv",
    )
    parser.add_argument(
        "--data-base",
        default="/home/phish-target-recognition/data_splits/visualphish/phishpedia/data/val",
        help="Base path to validation data directory",
        dest="data_base",
    )
    parser.add_argument(
        "--index-path",
        default="/home/phish-target-recognition/logs/vp/vp-for-baseline/index.faiss",
        help="Path to FAISS index file",
        dest="index_path",
    )
    args = parser.parse_args()

    thresholds = [30, 50, 70, 90, 110, 130, 150, 170, 200]
    results = []
    best_f1 = 0
    best_mcc = -1
    best_target_mcc = -1
    best_threshold_f1 = None
    best_threshold_mcc = None
    best_threshold_target_mcc = None

    # Use parsed arguments
    val_csv_path = args.val_csv

    print(
        f"Starting threshold optimization with {len(thresholds)} thresholds: {thresholds}"
    )
    logger.info(f"Starting threshold optimization with thresholds: {thresholds}")
    logger.info(f"Using validation CSV: {val_csv_path}")

    # Change to baseline directory
    baseline_dir = Path(__file__).parent
    os.chdir(baseline_dir)
    logger.info(f"Working directory: {baseline_dir}")
    print(f"Working directory: {baseline_dir}")

    for i, threshold in enumerate(thresholds, 1):
        print(f"\n[{i}/{len(thresholds)}] > Testing threshold: {threshold}")
        logger.info(f"Testing threshold: {threshold}")

        if run_query_with_threshold(
            threshold, val_csv_path, args.data_base, args.index_path
        ):
            output_file = f"results_threshold_{threshold}.csv"
            f1, mcc, target_mcc = calculate_metrics_from_results(output_file)

            results.append(
                {
                    "threshold": threshold,
                    "f1_weighted": f1,
                    "mcc": mcc,
                    "target_mcc": target_mcc,
                }
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = threshold

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold_mcc = threshold

            if target_mcc > best_target_mcc:
                best_target_mcc = target_mcc
                best_threshold_target_mcc = threshold

            print(f"Results: F1={f1:.4f}, MCC={mcc:.4f}, Target MCC={target_mcc:.4f}")
            logger.info(
                f"Threshold {threshold}: F1={f1:.4f}, MCC={mcc:.4f}, Target MCC={target_mcc:.4f}"
            )

            # Clean up temporary file
            Path(output_file).unlink(missing_ok=True)
        else:
            print(f"Failed to process threshold {threshold}")
            logger.error(f"Skipping threshold {threshold} due to query failure")

    # Save results
    if results:
        # Save results summary
        results_df = pd.DataFrame(results)
        results_df.to_csv("results_summary.csv", index=False)
        logger.info("Saved results_summary.csv")

        # Save best thresholds
        Path("best_threshold_f1.txt").write_text(str(best_threshold_f1))
        Path("best_threshold_mcc.txt").write_text(str(best_threshold_mcc))
        Path("best_threshold_target_mcc.txt").write_text(str(best_threshold_target_mcc))

        logger.info(f"Best F1 threshold: {best_threshold_f1}")
        logger.info(f"Best MCC threshold: {best_threshold_mcc}")
        logger.info(f"Best Target MCC threshold: {best_threshold_target_mcc}")

        print("\nOPTIMIZATION COMPLETE!")
        print(f"Best F1: {best_f1:.4f} at threshold {best_threshold_f1}")
        print(f"Best MCC: {best_mcc:.4f} at threshold {best_threshold_mcc}")
        print(
            f"Best Target MCC: {best_target_mcc:.4f} at threshold {best_threshold_target_mcc}"
        )
        print("Results saved to results_summary.csv")
    else:
        logger.error("No successful results obtained")
        sys.exit(1)


if __name__ == "__main__":
    main()

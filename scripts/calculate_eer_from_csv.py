#!/usr/bin/env python3
"""
Calculate EER thresholds from VisualPhish result CSV.

This script replicates the EER calculation logic from threshold_optimizer.py
using pre-computed distances stored in a CSV file.

Usage:
    python calculate_eer_from_csv.py /path/to/pp-result.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_distances_from_csv(csv_path: Path):
    """Load and separate benign/phishing distances from result CSV."""
    df = pd.read_csv(csv_path)

    # Filter valid rows (true_class should be 0 or 1)
    df = df[df["true_class"].isin([0, 1])]

    benign_mask = df["true_class"] == 0
    phish_mask = df["true_class"] == 1

    benign_distances = df.loc[benign_mask, "vp_distance"].values
    phish_distances = df.loc[phish_mask, "vp_distance"].values

    return benign_distances, phish_distances


def generate_thresholds(benign_distances, phish_distances):
    """Generate threshold range based on data statistics."""
    all_distances = np.concatenate([benign_distances, phish_distances])

    mean_val = np.mean(all_distances)
    std_val = np.std(all_distances)
    max_val = np.max(all_distances)

    print("Distance statistics:")
    print(f"  Mean: {mean_val:.3f}")
    print(f"  Std:  {std_val:.3f}")
    print(f"  Max:  {max_val:.3f}")
    print()

    # Statistical range (mean ± std)
    stat_start = int(mean_val - std_val)
    stat_end = int(mean_val + std_val)
    stat_range = np.arange(max(0, stat_start), stat_end + 1, 1)

    # Absolute range (0 to max in steps of 10)
    abs_range = np.arange(0, int(max_val) + 1, 10)

    # Fine-grained range around likely EER point
    fine_range = np.arange(0, min(10, max_val), 0.5)

    combined = np.unique(np.concatenate([stat_range, abs_range, fine_range]))
    combined = np.sort(combined)

    print(f"Generated {len(combined)} threshold values")
    print(f"  Range: [{combined[0]:.1f}, {combined[-1]:.1f}]")
    print()

    return combined


def find_eer_thresholds(benign_distances, phish_distances, thresholds):
    """
    Find EER thresholds where FPR ≈ FNR.

    Replicates logic from threshold_optimizer.py:find_eer_threshold
    """
    results = []

    for threshold in thresholds:
        # FPR: proportion of benign samples with distance ≤ threshold
        # (incorrectly classified as phishing)
        false_positives = np.sum(benign_distances <= threshold)
        fpr = false_positives / len(benign_distances)

        # FNR: proportion of phishing samples with distance > threshold
        # (incorrectly classified as benign)
        false_negatives = np.sum(phish_distances > threshold)
        fnr = false_negatives / len(phish_distances)

        # Other metrics
        true_positives = np.sum(phish_distances <= threshold)
        true_negatives = np.sum(benign_distances > threshold)

        tpr = true_positives / len(phish_distances)
        tnr = true_negatives / len(benign_distances)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / (
            len(benign_distances) + len(phish_distances)
        )
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        # EER: point where FPR ≈ FNR
        eer_diff = abs(fpr - fnr)
        eer_value = (fpr + fnr) / 2

        results.append(
            {
                "threshold": threshold,
                "fpr": fpr,
                "fnr": fnr,
                "tpr": tpr,
                "tnr": tnr,
                "precision": precision,
                "recall": tpr,
                "accuracy": accuracy,
                "f1": f1,
                "eer_diff": eer_diff,
                "eer_value": eer_value,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate EER thresholds from VisualPhish result CSV"
    )
    parser.add_argument(
        "csv_path", type=Path, help="Path to result CSV (e.g., pp-result.csv)"
    )
    parser.add_argument(
        "--save-sweep", type=Path, help="Optional: save full threshold sweep to CSV"
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: File not found: {args.csv_path}")
        sys.exit(1)

    print(f"Loading data from: {args.csv_path}")
    print()

    benign_distances, phish_distances = load_distances_from_csv(args.csv_path)

    print("Loaded distances:")
    print(f"  Benign samples:   {len(benign_distances)}")
    print(f"  Phishing samples: {len(phish_distances)}")
    print()

    print(
        f"Benign distances - min: {benign_distances.min():.4f}, max: {benign_distances.max():.4f}, mean: {benign_distances.mean():.4f}"
    )
    print(
        f"Phishing distances - min: {phish_distances.min():.4f}, max: {phish_distances.max():.4f}, mean: {phish_distances.mean():.4f}"
    )
    print()

    thresholds = generate_thresholds(benign_distances, phish_distances)
    results = find_eer_thresholds(benign_distances, phish_distances, thresholds)

    # Sort by eer_diff to find best EER points
    results_sorted = sorted(results, key=lambda x: x["eer_diff"])
    top_3 = results_sorted[:3]

    print("=" * 70)
    print("TOP 3 EER THRESHOLDS")
    print("=" * 70)

    for i, metrics in enumerate(top_3, 1):
        print(f"\n#{i}: Threshold = {metrics['threshold']:.3f}")
        print(f"     EER       = {metrics['eer_value']:.4f}")
        print(f"     FPR       = {metrics['fpr']:.4f}")
        print(f"     FNR       = {metrics['fnr']:.4f}")
        print(f"     |FPR-FNR| = {metrics['eer_diff']:.6f}")
        print(f"     Accuracy  = {metrics['accuracy']:.4f}")
        print(f"     F1        = {metrics['f1']:.4f}")

    print()
    print("=" * 70)

    if args.save_sweep:
        df = pd.DataFrame(results)
        df.to_csv(args.save_sweep, index=False)
        print(f"\nSaved full threshold sweep to: {args.save_sweep}")


if __name__ == "__main__":
    main()

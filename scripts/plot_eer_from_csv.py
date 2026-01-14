#!/usr/bin/env python3
"""
Plot EER (Equal Error Rate) curve from a threshold sweep CSV produced by VisualPhish threshold optimizer.

Inputs (CSV columns expected): threshold, fpr, fnr [others optional]
Outputs: A PNG image with FPR and FNR vs threshold and the EER point highlighted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from font_config import get_font_size, get_figure_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EER curve from threshold sweep CSV"
    )
    parser.add_argument(
        "--csv", type=Path, required=True, help="Path to threshold_sweep.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the output PNG (default: alongside CSV as *_eer_curve.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    # Ensure required columns exist
    for col in ("threshold", "fpr", "fnr"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV: {args.csv}")

    # Sort by threshold for nice plotting
    df = df.sort_values(by="threshold").reset_index(drop=True)

    # Compute EER diff if not provided
    if "eer_diff" not in df.columns:
        df["eer_diff"] = (df["fpr"] - df["fnr"]).abs()

    # Compute EER value if not provided
    if "eer_value" not in df.columns:
        df["eer_value"] = (df["fpr"] + df["fnr"]) / 2.0

    # Find optimal threshold where |FPR - FNR| is minimized (EER point)
    best_idx = int(df["eer_diff"].idxmin())
    best = df.loc[best_idx]

    # Prepare output path
    if args.output is None:
        args.output = args.csv.with_name(args.csv.stem + "_eer_curve.png")

    # Plot
    plt.figure(figsize=get_figure_size("single_plot"))

    # Setup formatters for axes
    formatter_2 = FuncFormatter(lambda x, pos: f"{x:.2f}")
    formatter_3 = FuncFormatter(lambda x, pos: f"{x:.3f}")

    def fmt_value(v, decimals=2):
        return f"{v:.{decimals}f}"

    plt.plot(
        df["threshold"], df["fpr"], "b-", linewidth=2, label="False Positive Rate (FPR)"
    )
    plt.plot(
        df["threshold"], df["fnr"], "r-", linewidth=2, label="False Negative Rate (FNR)"
    )

    # Plot EER point with values in legend
    plt.plot(
        best["threshold"],
        best["fpr"],
        "go",
        markersize=10,
        label=f"EER Point (FPR={best['fpr']:.3f}, FNR={best['fnr']:.3f})",
    )

    # Add green vertical line at optimal threshold (no label)
    plt.axvline(x=best["threshold"], color="green", linestyle="--", linewidth=2)

    # Add text label on the green line
    ax_tmp = plt.gca()
    ymax = max(df["fpr"].max(), df["fnr"].max())
    text_y = ymax * 0.75  # Position text at 75% height to avoid legend overlap

    ax_tmp.text(
        best["threshold"],
        text_y,
        f"EER Threshold = {fmt_value(float(best['threshold']), 2)}",
        color="green",
        fontsize=get_font_size("eer_text"),
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Apply formatters and styling
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter_2)
    ax.yaxis.set_major_formatter(formatter_3)
    ax.tick_params(axis="both", which="major", labelsize=32)

    plt.xlabel("Distance Threshold", fontsize=26)
    plt.ylabel("Error Rate", fontsize=26)
    plt.title(
        "EER Analysis: FPR and FNR vs Threshold",
        fontsize=get_font_size("title"),
        weight="bold",
    )
    plt.legend(fontsize=get_font_size("legend"))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved EER curve to: {args.output}")
    print(
        f"Optimal threshold={best['threshold']}, EER={best['eer_value']:.4f}, FPR={best['fpr']:.4f}, FNR={best['fnr']:.4f}"
    )


if __name__ == "__main__":
    main()

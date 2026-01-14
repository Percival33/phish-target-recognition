#!/usr/bin/env python3
"""
Plot grouped barplots for binary and multiclass classification metrics.

Binary: 3 panels (F1, ROC AUC, MCC)
Multiclass: 4 panels (F1 micro, F1 macro, MCC, Identification rate)
X-axis: datasets (CERT, VP, PP), grouped by methods (Phishpedia, Baseline, VisualPhish).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from font_config import get_font_size

# Binary classification data
BINARY_DATA = {
    "Phishpedia": {
        "CERT": {"F1": 0.1598, "ROC AUC": 0.5304, "MCC": 0.1301},
        "VP": {"F1": 0.4263, "ROC AUC": 0.4544, "MCC": -0.0955},
        "PP": {"F1": 0.9062, "ROC AUC": 0.6679, "MCC": 0.2729},
    },
    "Baseline": {
        "CERT": {"F1": 0.6229, "ROC AUC": 0.5682, "MCC": 0.1449},
        "VP": {"F1": 0.7673, "ROC AUC": 0.8201, "MCC": 0.6294},
        "PP": {"F1": 0.9539, "ROC AUC": 0.7759, "MCC": 0.5391},
    },
    "VisualPhish": {
        "CERT": {"F1": 0.3577, "ROC AUC": 0.4101, "MCC": -0.1812},
        "VP": {"F1": 0.4954, "ROC AUC": 0.5887, "MCC": 0.1693},
        "PP": {"F1": 0.1673, "ROC AUC": 0.1018, "MCC": -0.6160},
    },
}

# Multiclass classification data
MULTICLASS_DATA = {
    "Phishpedia": {
        "CERT": {
            "F1 micro": 0.5482,
            "F1 macro": 0.2621,
            "MCC": 0.2013,
            "Identification rate": 0.9845,
        },
        "VP": {
            "F1 micro": 0.3782,
            "F1 macro": 0.3073,
            "MCC": 0.2569,
            "Identification rate": 0.9270,
        },
        "PP": {
            "F1 micro": 0.7691,
            "F1 macro": 0.2894,
            "MCC": 0.7384,
            "Identification rate": 0.9154,
        },
    },
    "Baseline": {
        "CERT": {
            "F1 micro": 0.5823,
            "F1 macro": 0.1670,
            "MCC": 0.3668,
            "Identification rate": 0.2554,
        },
        "VP": {
            "F1 micro": 0.7009,
            "F1 macro": 0.4111,
            "MCC": 0.5089,
            "Identification rate": 0.5679,
        },
        "PP": {
            "F1 micro": 0.5467,
            "F1 macro": 0.3591,
            "MCC": 0.4939,
            "Identification rate": 0.5687,
        },
    },
    "VisualPhish": {
        "CERT": {
            "F1 micro": 0.3654,
            "F1 macro": 0.1481,
            "MCC": 0.0733,
            "Identification rate": 0.7093,
        },
        "VP": {
            "F1 micro": 0.3924,
            "F1 macro": 0.0047,
            "MCC": 0.0694,
            "Identification rate": 0.0037,
        },
        "PP": {
            "F1 micro": 0.1003,
            "F1 macro": 0.0334,
            "MCC": 0.0111,
            "Identification rate": 1.0000,
        },
    },
}

# Display order
DATASETS = ["CERT", "VP", "PP"]
METHODS = ["Phishpedia", "Baseline", "VisualPhish"]
BINARY_METRICS = ["F1", "ROC AUC", "MCC"]
MULTICLASS_METRICS = ["F1 micro", "F1 macro", "MCC", "Identification rate"]

# Method colors - distinctive palette
METHOD_COLORS = {
    "Phishpedia": "#2ecc71",  # green
    "Baseline": "#3498db",  # blue
    "VisualPhish": "#e74c3c",  # red
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot grouped barplots for classification metrics"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for plots (default: ../scripts-data relative to script)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["binary", "multiclass", "both"],
        default="binary",
        help="Type of classification to plot (default: binary)",
    )
    return parser.parse_args()


def build_dataframe(data: dict) -> pd.DataFrame:
    """Convert nested dict to long-form DataFrame for seaborn."""
    records = []
    for method, datasets in data.items():
        for dataset, metrics in datasets.items():
            for metric, value in metrics.items():
                if value is not None:
                    records.append(
                        {
                            "Method": method,
                            "Dataset": dataset,
                            "Metric": metric,
                            "Value": value,
                        }
                    )
    return pd.DataFrame(records)


def format_value_with_minus(value: float) -> str:
    """Format value with proper minus sign (U+2212) for negative numbers."""
    formatted = f"{value:.2f}"
    return formatted.replace("-", "\u2212")


def plot_metrics_barplot(
    df: pd.DataFrame,
    title_suffix: str,
    output_path: Path,
    metrics: list[str],
    grid_layout: Optional[Tuple[int, int]] = None,
) -> None:
    """Create a multi-panel grouped barplot for metrics."""
    sns.set_theme(style="whitegrid")

    n_panels = len(metrics)

    # Determine grid layout
    if grid_layout:
        nrows, ncols = grid_layout
        fig, axes = plt.subplots(nrows, ncols, figsize=(14 * ncols, 14 * nrows))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(14 * n_panels, 14))
        if n_panels == 1:
            axes = [axes]

    # Font size multiplier for larger text
    FONT_MULT = 1.4

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_df = df[df["Metric"] == metric]

        # Create barplot
        sns.barplot(
            data=metric_df,
            x="Dataset",
            y="Value",
            hue="Method",
            order=DATASETS,
            hue_order=METHODS,
            palette=METHOD_COLORS,
            ax=ax,
            edgecolor="black",
            linewidth=1.5,
        )

        # Set alpha for all bars
        for patch in ax.patches:
            patch.set_alpha(0.6)

        # Styling with larger fonts
        ax.set_title(
            metric, fontsize=int(get_font_size("title") * FONT_MULT), weight="bold"
        )
        ax.set_xlabel("Dataset", fontsize=int(get_font_size("xlabel") * FONT_MULT))
        ax.set_ylabel("Value", fontsize=int(get_font_size("ylabel") * FONT_MULT))
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=int(get_font_size("tick_labels") * FONT_MULT),
        )

        # Set y-axis limits based on metric
        if metric == "MCC":
            # ax.set_ylim(-0.7, 1.05)
            ax.set_ylim(-0.1, 0.8)
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.7)
        else:
            ax.set_ylim(0, 1.05)

        # Add value labels on bars with larger font
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt=format_value_with_minus,
                fontsize=int(get_font_size("title") * FONT_MULT),
                padding=4,
                fontweight="bold",
            )

        # Legend only on first panel
        if idx == 0:
            ax.legend(
                title="Method",
                fontsize=int(get_font_size("title") * FONT_MULT),
                title_fontsize=int(get_font_size("title") * FONT_MULT),
                loc="upper left",
            )
        else:
            ax.get_legend().remove()

    # Hide unused subplots if using grid layout
    if grid_layout:
        for idx in range(n_panels, len(axes)):
            axes[idx].set_visible(False)

    plt.suptitle(
        f"{title_suffix} Classification Metrics",
        fontsize=int((get_font_size("title") + 4) * FONT_MULT),
        weight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved plot: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(__file__).resolve().parents[1] / "scripts-data"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ("binary", "both"):
        df_binary = build_dataframe(BINARY_DATA)
        plot_metrics_barplot(
            df_binary,
            title_suffix="Binary",
            output_path=output_dir / "metrics-binary-barplot.png",
            metrics=BINARY_METRICS,
            grid_layout=(2, 2),
        )

    if args.type in ("multiclass", "both"):
        df_multiclass = build_dataframe(MULTICLASS_DATA)
        if not df_multiclass.empty:
            plot_metrics_barplot(
                df_multiclass,
                title_suffix="Multiclass",
                output_path=output_dir / "metrics-multiclass-barplot.png",
                metrics=MULTICLASS_METRICS,
                grid_layout=(2, 2),
            )
        else:
            print("No data for multiclass classification - skipping.")


if __name__ == "__main__":
    main()

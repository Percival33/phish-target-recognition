#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from font_config import get_font_size, get_figure_size

# Color palette matching plot_metrics_barplot.py
COLORS = {
    "blue": "#3498db",
    "red": "#e74c3c",
    "green": "#2ecc71",
}


def _get_scipy_stats():
    """Lazy import of scipy.stats to avoid dependency when not fitting."""
    try:
        from scipy import stats

        return stats
    except ImportError:
        raise ImportError(
            "scipy is required for distribution fitting. Install with: pip install scipy"
        )


def estimate_distribution_params_from_histogram(bins, hist_density):
    """
    Estimate distribution parameters (mu, sigma) from histogram data.

    Args:
        bins: Bin edges array
        hist_density: Histogram density values

    Returns:
        tuple: (mu, sigma) or (None, None) if estimation fails
    """
    try:
        stats = _get_scipy_stats()

        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_widths = np.diff(bins)
        approx_counts = hist_density * bin_widths * 10000
        approx_counts = np.maximum(approx_counts, 0).astype(int)

        sample_data = []
        for center, count in zip(bin_centers, approx_counts):
            sample_data.extend([center] * count)

        if len(sample_data) < 2:
            return None, None

        sample_array = np.array(sample_data)
        mu, sigma = stats.norm.fit(sample_array)

        return float(mu), float(sigma)

    except Exception:
        return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EER curve from JSON data")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("eer.json"),
        help="Path to input JSON file (default: eer.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save output PNG (default: eer_curve.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.input
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=get_figure_size("single_plot"))
    comma_formatter_2 = FuncFormatter(lambda x, pos: f"{x:.2f}")
    comma_formatter_3 = FuncFormatter(lambda x, pos: f"{x:.3f}")

    # Case 1: EER curve format
    if all(k in data for k in ("thresholds", "fprs", "fnrs")):
        thresholds = data["thresholds"]
        fprs = data["fprs"]
        fnrs = data["fnrs"]
        optimal_threshold = data.get("optimal_threshold")
        optimal_fpr = data.get("optimal_fpr")
        optimal_fnr = data.get("optimal_fnr")

        plt.plot(
            thresholds,
            fprs,
            color=COLORS["blue"],
            linewidth=2,
            label="False Positive Rate (FPR)",
        )
        plt.plot(
            thresholds,
            fnrs,
            color=COLORS["red"],
            linewidth=2,
            label="False Negative Rate (FNR)",
        )

        if (
            optimal_threshold is not None
            and optimal_fpr is not None
            and optimal_fnr is not None
        ):
            plt.plot(
                optimal_threshold,
                optimal_fpr,
                "o",
                color=COLORS["green"],
                markersize=10,
                label=f"EER Point (FPR={optimal_fpr:.3f}, FNR={optimal_fnr:.3f})",
            )
            plt.axvline(
                x=optimal_threshold, color=COLORS["green"], linestyle="--", linewidth=1
            )
            ax_tmp = plt.gca()
            ymax = max(max(fprs), max(fnrs))
            text_y = ymax * 0.75  # Lowered to avoid legend overlap

            def fmt_pl(v, decimals=2):
                # return (f"{v:.{decimals}f}").replace(".", ",")
                return f"{v:.{decimals}f}"

            ax_tmp.text(
                optimal_threshold,
                text_y,
                f"EER Threshold = {fmt_pl(float(optimal_threshold), 2)}",
                color=COLORS["green"],
                fontsize=get_font_size("eer_text"),
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax = plt.gca()
        # ax.xaxis.set_major_formatter(comma_formatter_2)
        # ax.yaxis.set_major_formatter(comma_formatter_3)
        ax.tick_params(
            axis="both", which="major", labelsize=get_font_size("tick_labels")
        )

        plt.xlabel("Distance Threshold", fontsize=get_font_size("xlabel"))
        plt.ylabel("Error Rate", fontsize=get_font_size("ylabel"))
        plt.title(
            "EER Analysis: FPR and FNR vs Threshold",
            fontsize=get_font_size("title"),
            weight="bold",
        )
        plt.legend(fontsize=get_font_size("legend"))
        plt.tight_layout()

    # Case 2: Density histogram format
    elif all(k in data for k in ("bins", "hist_benign_density", "hist_phish_density")):
        bins = np.asarray(data["bins"], dtype=float)
        hist_benign = np.asarray(data["hist_benign_density"], dtype=float)
        hist_phish = np.asarray(data["hist_phish_density"], dtype=float)
        x_range = (
            np.asarray(data.get("x_range", []), dtype=float)
            if data.get("x_range") is not None
            else None
        )
        benign_pdf = (
            np.asarray(data.get("benign_pdf", []), dtype=float)
            if data.get("benign_pdf") is not None
            else None
        )
        phish_pdf = (
            np.asarray(data.get("phish_pdf", []), dtype=float)
            if data.get("phish_pdf") is not None
            else None
        )
        optimal_threshold = data.get("optimal_threshold")

        widths = np.diff(bins)
        lefts = bins[:-1]

        plt.bar(
            lefts,
            hist_benign,
            width=widths,
            align="edge",
            alpha=0.6,
            label="Images from benign websites",
            color=COLORS["blue"],
            edgecolor="black",
        )
        plt.bar(
            lefts,
            hist_phish,
            width=widths,
            align="edge",
            alpha=0.6,
            label="Images from phishing websites",
            color=COLORS["red"],
            edgecolor="black",
        )

        # Get Gaussian fit parameters for legend
        benign_mu = data.get("benign_mu")
        benign_sigma = data.get("benign_sigma")
        phish_mu = data.get("phish_mu")
        phish_sigma = data.get("phish_sigma")

        # Estimate from histogram if not present in JSON
        if benign_mu is None or benign_sigma is None:
            benign_mu, benign_sigma = estimate_distribution_params_from_histogram(
                bins, hist_benign
            )
        if phish_mu is None or phish_sigma is None:
            phish_mu, phish_sigma = estimate_distribution_params_from_histogram(
                bins, hist_phish
            )

        def fmt_val(v, decimals=2):
            # return (f"{v:.{decimals}f}").replace(".", ",")
            return f"{v:.{decimals}f}"

        if (
            x_range is not None
            and x_range.size > 0
            and benign_pdf is not None
            and benign_pdf.size == x_range.size
        ):
            benign_label = "Benign fit"
            if benign_mu is not None and benign_sigma is not None:
                benign_label = (
                    f"Benign fit (μ={fmt_val(benign_mu)}, σ={fmt_val(benign_sigma)})"
                )
            plt.plot(
                x_range,
                benign_pdf,
                color=COLORS["blue"],
                linewidth=2,
                label=benign_label,
            )
        if (
            x_range is not None
            and x_range.size > 0
            and phish_pdf is not None
            and phish_pdf.size == x_range.size
        ):
            phish_label = "Phishing fit"
            if phish_mu is not None and phish_sigma is not None:
                phish_label = (
                    f"Phishing fit (μ={fmt_val(phish_mu)}, σ={fmt_val(phish_sigma)})"
                )
            plt.plot(
                x_range, phish_pdf, color=COLORS["red"], linewidth=2, label=phish_label
            )

        if optimal_threshold is not None:
            try:
                thr = float(optimal_threshold)
                plt.axvline(
                    x=thr,
                    color=COLORS["green"],
                    linestyle="--",
                    linewidth=2,
                    label="Optimal EER Threshold",
                )
                ymax = max(
                    float(np.max(hist_benign)) if hist_benign.size else 0.0,
                    float(np.max(hist_phish)) if hist_phish.size else 0.0,
                )
                text_y = (
                    ymax * 0.75 if ymax > 0 else 0.0
                )  # Lowered to avoid legend overlap

                def fmt_pl(v, decimals=2):
                    # return (f"{v:.{decimals}f}").replace(".", ",")
                    return f"{v:.{decimals}f}"

                plt.text(
                    thr,
                    text_y,
                    f"EER Threshold = {fmt_pl(thr, 2)}",
                    color=COLORS["green"],
                    fontsize=get_font_size("eer_text"),
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            except Exception:
                pass

        ax = plt.gca()
        ax.xaxis.set_major_formatter(comma_formatter_2)
        ax.yaxis.set_major_formatter(comma_formatter_3)
        ax.tick_params(
            axis="both", which="major", labelsize=get_font_size("tick_labels")
        )

        plt.xlabel("Distance to Nearest Target", fontsize=get_font_size("xlabel"))
        plt.ylabel("Density", fontsize=get_font_size("ylabel"))
        plt.title(
            "Distance Distribution with Gaussian Fit and EER Threshold",
            fontsize=get_font_size("title"),
            weight="bold",
        )
        plt.legend(fontsize=get_font_size("legend"))
        plt.tight_layout()

    else:
        raise ValueError(
            "Unknown file format: expected fields (thresholds,fprs,fnrs) or (bins,hist_*,pdfs)"
        )

    out_path = args.output if args.output is not None else Path("eer_curve.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()

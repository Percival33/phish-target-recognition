#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from font_config import get_font_size, get_figure_size


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
    parser = argparse.ArgumentParser(
        description="Plot density histograms for three datasets"
    )
    parser.add_argument(
        "--folder-root",
        "-f",
        type=Path,
        default=None,
        help="Path to folder containing data files (default: ../scripts-data relative to script)",
    )
    return parser.parse_args()


def load_hist_json(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
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
    benign_mu = data.get("benign_mu")
    benign_sigma = data.get("benign_sigma")
    phish_mu = data.get("phish_mu")
    phish_sigma = data.get("phish_sigma")

    return {
        "bins": bins,
        "hist_benign": hist_benign,
        "hist_phish": hist_phish,
        "x_range": x_range,
        "benign_pdf": benign_pdf,
        "phish_pdf": phish_pdf,
        "optimal_threshold": optimal_threshold,
        "benign_mu": benign_mu,
        "benign_sigma": benign_sigma,
        "phish_mu": phish_mu,
        "phish_sigma": phish_sigma,
    }


def plot_one(ax, payload: dict, title: str) -> None:
    bins = payload["bins"]
    hist_benign = payload["hist_benign"]
    hist_phish = payload["hist_phish"]
    x_range = payload["x_range"]
    benign_pdf = payload["benign_pdf"]
    phish_pdf = payload["phish_pdf"]
    optimal_threshold = payload["optimal_threshold"]
    benign_mu = payload["benign_mu"]
    benign_sigma = payload["benign_sigma"]
    phish_mu = payload["phish_mu"]
    phish_sigma = payload["phish_sigma"]

    if benign_mu is None or benign_sigma is None:
        benign_mu, benign_sigma = estimate_distribution_params_from_histogram(
            bins, hist_benign
        )
    if phish_mu is None or phish_sigma is None:
        phish_mu, phish_sigma = estimate_distribution_params_from_histogram(
            bins, hist_phish
        )

    widths = np.diff(bins)
    lefts = bins[:-1]

    ax.bar(
        lefts,
        hist_benign,
        width=widths,
        align="edge",
        alpha=0.6,
        label="Obrazy ze stron nieszkodliwych",
        color="blue",
        edgecolor="black",
    )
    ax.bar(
        lefts,
        hist_phish,
        width=widths,
        align="edge",
        alpha=0.6,
        label="Obrazy ze stron phishingowych",
        color="red",
        edgecolor="black",
    )

    def fmt_pl(value, decimals=2):
        """Format numbers with Polish decimal notation."""
        return (f"{value:.{decimals}f}").replace(".", ",")

    if (
        x_range is not None
        and x_range.size > 0
        and benign_pdf is not None
        and benign_pdf.size == x_range.size
    ):
        if benign_mu is not None and benign_sigma is not None:
            ax.plot(
                x_range,
                benign_pdf,
                "b-",
                linewidth=2,
                label=f"Dopasowanie nieszkodliwe (μ={fmt_pl(benign_mu, 2)}, σ={fmt_pl(benign_sigma, 2)})",
            )
        else:
            ax.plot(
                x_range, benign_pdf, "b-", linewidth=2, label="Dopasowanie nieszkodliwe"
            )

    if (
        x_range is not None
        and x_range.size > 0
        and phish_pdf is not None
        and phish_pdf.size == x_range.size
    ):
        if phish_mu is not None and phish_sigma is not None:
            ax.plot(
                x_range,
                phish_pdf,
                "r-",
                linewidth=2,
                label=f"Dopasowanie phishingowe (μ={fmt_pl(phish_mu, 2)}, σ={fmt_pl(phish_sigma, 2)})",
            )
        else:
            ax.plot(
                x_range, phish_pdf, "r-", linewidth=2, label="Dopasowanie phishingowe"
            )

    if optimal_threshold is not None:
        try:
            thr = float(optimal_threshold)
            ax.axvline(
                x=thr,
                color="green",
                linestyle="--",
                linewidth=2,
                label="Optymalny próg EER",
            )
            ymax = max(
                float(np.max(hist_benign)) if hist_benign.size else 0.0,
                float(np.max(hist_phish)) if hist_phish.size else 0.0,
            )
            text_y = ymax * 0.65 if ymax > 0 else 0.0

            def fmt_pl(v, decimals=2):
                return (f"{v:.{decimals}f}").replace(".", ",")

            ax.text(
                thr,
                text_y,
                f"Próg EER = {fmt_pl(thr, 2)}",
                color="green",
                fontsize=get_font_size("eer_text"),
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        except Exception:
            pass

    ax.set_title(title, fontsize=get_font_size("title"), weight="bold")
    ax.set_xlabel("Odległość do najbliższego celu", fontsize=get_font_size("xlabel"))
    ax.set_ylabel("Gęstość", fontsize=get_font_size("ylabel"))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=get_font_size("tick_labels"))


def main():
    args = parse_args()

    if args.folder_root is not None:
        repo_root = args.folder_root
    else:
        repo_root = Path(__file__).resolve().parents[1] / "scripts-data"

    cert_path = repo_root / "cert-hist.json"
    vp_path = repo_root / "vp-hist.json"
    pp_path = repo_root / "pp-hist.json"

    cert_data = load_hist_json(cert_path)
    vp_data = load_hist_json(vp_path)
    pp_data = load_hist_json(pp_path)

    comma_formatter_2 = FuncFormatter(lambda x, pos: f"{x:.2f}".replace(".", ","))
    comma_formatter_3 = FuncFormatter(lambda x, pos: f"{x:.3f}".replace(".", ","))

    datasets = [
        (cert_data, "Zbiór CERT", "cert-hist.png"),
        (vp_data, "Zbiór VP", "vp-hist.png"),
        (pp_data, "Zbiór PP", "pp-hist.png"),
    ]

    for payload, title, filename in datasets:
        fig, ax = plt.subplots(figsize=get_figure_size("single_plot"))

        plot_one(ax, payload, title)
        ax.xaxis.set_major_formatter(comma_formatter_2)
        ax.yaxis.set_major_formatter(comma_formatter_3)

        ax.legend(loc="upper right", fontsize=get_font_size("legend"))

        plt.tight_layout()
        out_path = repo_root / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Zapisano wykres: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

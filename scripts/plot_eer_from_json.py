#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from font_config import get_font_size, get_figure_size


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
        raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    plt.figure(figsize=get_figure_size("single_plot"))
    comma_formatter_2 = FuncFormatter(lambda x, pos: f"{x:.2f}".replace(".", ","))
    comma_formatter_3 = FuncFormatter(lambda x, pos: f"{x:.3f}".replace(".", ","))

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
            "b-",
            linewidth=2,
            label="Odsetek fałszywie pozytywnych (FPR)",
        )
        plt.plot(
            thresholds,
            fnrs,
            "r-",
            linewidth=2,
            label="Odsetek fałszywie negatywnych (FNR)",
        )

        if (
            optimal_threshold is not None
            and optimal_fpr is not None
            and optimal_fnr is not None
        ):
            plt.plot(
                optimal_threshold,
                optimal_fpr,
                "go",
                markersize=10,
                label=f"Punkt EER (FPR={optimal_fpr:.3f}, FNR={optimal_fnr:.3f})",
            )
            plt.axvline(x=optimal_threshold, color="green", linestyle="--", linewidth=1)
            ax_tmp = plt.gca()
            ymax = max(max(fprs), max(fnrs))
            text_y = ymax * 0.75  # Lowered to avoid legend overlap

            def fmt_pl(v, decimals=2):
                return (f"{v:.{decimals}f}").replace(".", ",")

            ax_tmp.text(
                optimal_threshold,
                text_y,
                f"Próg EER = {fmt_pl(float(optimal_threshold), 2)}",
                color="green",
                fontsize=get_font_size("eer_text"),
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax = plt.gca()
        ax.xaxis.set_major_formatter(comma_formatter_2)
        ax.yaxis.set_major_formatter(comma_formatter_3)
        ax.tick_params(
            axis="both", which="major", labelsize=get_font_size("tick_labels")
        )

        plt.xlabel("Próg odległości", fontsize=get_font_size("xlabel"))
        plt.ylabel("Współczynnik błędu", fontsize=get_font_size("ylabel"))
        plt.title(
            "Analiza EER: FPR i FNR względem progu",
            fontsize=get_font_size("title"),
            weight="bold",
        )
        plt.legend(fontsize=get_font_size("legend"))
        plt.grid(True, alpha=0.3)
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
            label="Obrazy ze stron nieszkodliwych",
            color="blue",
            edgecolor="black",
        )
        plt.bar(
            lefts,
            hist_phish,
            width=widths,
            align="edge",
            alpha=0.6,
            label="Obrazy ze stron phishingowych",
            color="red",
            edgecolor="black",
        )

        if (
            x_range is not None
            and x_range.size > 0
            and benign_pdf is not None
            and benign_pdf.size == x_range.size
        ):
            plt.plot(x_range, benign_pdf, "b-", linewidth=2)
        if (
            x_range is not None
            and x_range.size > 0
            and phish_pdf is not None
            and phish_pdf.size == x_range.size
        ):
            plt.plot(x_range, phish_pdf, "r-", linewidth=2)

        if optimal_threshold is not None:
            try:
                thr = float(optimal_threshold)
                plt.axvline(
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
                text_y = (
                    ymax * 0.75 if ymax > 0 else 0.0
                )  # Lowered to avoid legend overlap

                def fmt_pl(v, decimals=2):
                    return (f"{v:.{decimals}f}").replace(".", ",")

                plt.text(
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

        ax = plt.gca()
        ax.xaxis.set_major_formatter(comma_formatter_2)
        ax.yaxis.set_major_formatter(comma_formatter_3)
        ax.tick_params(
            axis="both", which="major", labelsize=get_font_size("tick_labels")
        )

        plt.xlabel("Odległość do najbliższego celu", fontsize=get_font_size("xlabel"))
        plt.ylabel("Gęstość", fontsize=get_font_size("ylabel"))
        plt.title(
            "Rozkład odległości z dopasowaniem Gaussa i progiem EER",
            fontsize=get_font_size("title"),
        )
        plt.legend(fontsize=get_font_size("legend"))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    else:
        raise ValueError(
            "Nieznany format pliku eer_curve_data.json: oczekiwano pól (thresholds,fprs,fnrs) lub (bins,hist_*,pdfs)"
        )

    out_path = Path("eer_curve.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Zapisano wykres do: {out_path}")


if __name__ == "__main__":
    main()

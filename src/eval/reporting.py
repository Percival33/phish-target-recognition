"""
Reporting module for generating and saving formatted tables.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tools.config import setup_logging
from config_loader import Config

setup_logging()
logger = logging.getLogger(__name__)


def create_formatted_results_table(
    scores_df: pd.DataFrame, rankings_df: pd.DataFrame, config: Config
) -> pd.DataFrame:
    dataset_names = config.dataset_names
    algorithm_names = config.algorithm_names

    formatted_data = []

    for dataset in dataset_names:
        row = {"Dataset": dataset}
        for algorithm in algorithm_names:
            score = scores_df.loc[dataset, algorithm]
            rank = int(rankings_df.loc[dataset, algorithm])
            row[algorithm] = f"{score:.2f} ({rank})"
        formatted_data.append(row)

    mean_ranks = rankings_df.mean()
    sum_ranks = rankings_df.sum()

    formatted_data.extend(
        [
            {
                "Dataset": "Mean Rank",
                **{alg: f"{mean_ranks[alg]:.2f}" for alg in algorithm_names},
            },
            {
                "Dataset": "Sum Ranks",
                **{alg: f"{sum_ranks[alg]:.0f}" for alg in algorithm_names},
            },
        ]
    )

    return pd.DataFrame(formatted_data)


def save_table_as_csv(formatted_df: pd.DataFrame, output_path: Path) -> None:
    formatted_df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV table to: {output_path}")


def save_table_as_image(
    formatted_df: pd.DataFrame, output_path: Path, config: Config
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    def style_row(row_idx, color, bold=True):
        for col_idx in range(len(formatted_df.columns)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(color)
            if bold:
                cell.set_text_props(
                    weight="bold", color="white" if color == "#4CAF50" else "black"
                )

    style_row(0, "#4CAF50")

    num_rows = len(formatted_df)
    style_row(num_rows - 1, "#E8F5E8")
    style_row(num_rows, "#E8F5E8")

    metric_name = config.metric_config.display_name
    plt.title(
        f"Algorithm Performance Comparison - {metric_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    logger.info(f"Saved table image to: {output_path}")


def create_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

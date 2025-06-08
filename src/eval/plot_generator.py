import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tools.config import setup_logging
from aeon.visualisation import plot_critical_difference
from config_loader import Config

setup_logging()
logger = logging.getLogger(__name__)


def create_and_save_cd_diagram(
    scores_matrix: np.ndarray, config: Config, output_path: Path
) -> None:
    """Create and save a critical difference diagram."""

    algorithm_names = config.algorithm_names
    metric_config = config.metric_config

    lower_better = not metric_config.higher_scores_are_better
    test = config.plot_config.get("cd_test", "nemenyi")
    correction = config.plot_config.get("cd_correction", "holm")

    logger.info(
        f"Creating CD diagram with test={test}, correction={correction}, lower_better={lower_better}"
    )

    fig, ax = plot_critical_difference(
        scores=scores_matrix,
        labels=algorithm_names,
        lower_better=lower_better,
        test=test,
        correction=correction,
    )

    ax.set_title(
        f"Critical Difference Diagram - {metric_config.display_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()

    logger.info(f"Saved critical difference diagram to: {output_path}")

"""
Data processing module for CSV reading and metric calculation.
"""

import logging
import pandas as pd
import numpy as np
import importlib
from pathlib import Path
from typing import Tuple

from config_loader import Config

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Exception raised for data processing errors."""

    pass


def process_all_data(config: Config) -> np.ndarray:
    """
    Process all CSV files and calculate metrics for each algorithm-dataset combination.

    Args:
        config: Configuration dictionary

    Returns:
        2D numpy array where rows are datasets and columns are algorithms

    Raises:
        DataProcessingError: If data processing fails
    """
    metric_config = config.metric_config

    try:
        module = importlib.import_module(metric_config.python_module_path)
        metric_function = getattr(module, metric_config.metric_function_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{metric_config.python_module_path}': {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Function '{metric_config.metric_function_name}' not found in module '{metric_config.python_module_path}': {e}"
        ) from e

    dataset_names = config.dataset_names
    algorithm_names = config.algorithm_names
    scores_matrix = np.zeros((len(dataset_names), len(algorithm_names)))

    for dataset_idx, dataset_name in enumerate(dataset_names):
        for algorithm_idx, algorithm_name in enumerate(algorithm_names):
            try:
                score = _calculate_score(
                    config, dataset_name, algorithm_name, metric_function
                )
                scores_matrix[dataset_idx, algorithm_idx] = score
            except Exception as e:
                raise DataProcessingError(
                    f"Processing failed for {algorithm_name} on {dataset_name}: {e}"
                )

    return scores_matrix


def _calculate_score(
    config: Config, dataset_name: str, algorithm_name: str, metric_function
) -> float:
    """
    Process a single algorithm-dataset combination.

    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset
        algorithm_name: Name of the algorithm
        metric_function: The metric calculation function

    Returns:
        Calculated metric score

    Raises:
        DataProcessingError: If processing fails
    """
    csv_path = config.data_input_config.paths_to_csv_files[dataset_name][algorithm_name]
    cls_true, cls_pred, targets_true, targets_pred = _read_csv_data(
        csv_path, config, algorithm_name
    )

    class_metrics, target_metrics = metric_function(
        cls_true, cls_pred, targets_true, targets_pred
    )

    metric_config = config.metric_config
    metrics = (
        class_metrics
        if metric_config.metric_type == "class_metrics"
        else target_metrics
    )
    score = metrics[metric_config.metric_name]

    return float(score * 100 if metric_config.scale_to_percentage else score)


def _read_csv_data(
    csv_path: str, config: Config, algorithm_name: str
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Read CSV data and extract the required columns.

    Args:
        csv_path: Path to the CSV file
        config: Configuration dictionary
        algorithm_name: Name of the algorithm

    Returns:
        Tuple of (cls_true, cls_pred, targets_true, targets_pred) pandas Series

    Raises:
        DataProcessingError: If CSV reading or column extraction fails
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise DataProcessingError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise DataProcessingError(f"Failed to read CSV file {csv_path}: {e}")

    prefix = config.data_input_config.csv_column_prefixes[algorithm_name]
    suffixes = config.data_input_config.csv_column_suffixes

    columns = {
        "cls_true": suffixes["true_class"],
        "cls_pred": f"{prefix}{suffixes['pred_class']}",
        "targets_true": suffixes["true_target"],
        "targets_pred": f"{prefix}{suffixes['pred_target']}",
    }

    try:
        return tuple(df[col] for col in columns.values())
    except KeyError as e:
        raise DataProcessingError(
            f"Missing column in {csv_path}: {e}. Available: {list(df.columns)}"
        )

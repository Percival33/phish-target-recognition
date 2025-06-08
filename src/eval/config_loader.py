"""
Configuration loading and validation module using Pydantic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, field_validator
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class MetricConfig(BaseModel):
    display_name: str
    python_module_path: str
    metric_function_name: str
    metric_type: str = Field(..., pattern="^(class_metrics|target_metrics)$")
    metric_name: str
    scale_to_percentage: bool
    higher_scores_are_better: bool


class DataInputConfig(BaseModel):
    paths_to_csv_files: Dict[str, Dict[str, str]]
    csv_column_prefixes: Dict[str, str]
    csv_column_suffixes: Dict[str, str]


class OutputConfig(BaseModel):
    results_directory: str = "./results/eval_output"
    ranked_table_image_filename: str = "ranked_results_table.png"
    critical_difference_plot_filename: str = "critical_difference_diagram.png"
    summary_csv_filename: str = "summary_results.csv"


class Config(BaseModel):
    # Dataset and algorithm configuration
    dataset_names: List[str] = Field(..., min_items=1)
    algorithm_names: List[str] = Field(..., min_items=1)

    # Nested configurations
    metric_config: MetricConfig
    data_input_config: DataInputConfig
    output_config: OutputConfig
    plot_config: Optional[Dict[str, Any]] = None

    @field_validator("data_input_config", mode="after")
    @classmethod
    def validate_data_input_config(cls, v, info):
        """Validate data input configuration against dataset and algorithm names."""
        if not info.data:
            return v

        datasets = info.data.get("dataset_names", [])
        algorithms = info.data.get("algorithm_names", [])

        if not datasets or not algorithms:
            pass

        missing_paths = []
        for dataset in datasets:
            if dataset not in v.paths_to_csv_files:
                missing_paths.append(f"dataset: {dataset}")
                continue
            if not isinstance(v.paths_to_csv_files.get(dataset), dict):
                missing_paths.append(
                    f"CSV path entry for dataset '{dataset}' is not a valid structure."
                )
                continue
            for algorithm in algorithms:
                if algorithm not in v.paths_to_csv_files[dataset]:
                    missing_paths.append(f"dataset {dataset}, algorithm {algorithm}")

        if missing_paths:
            raise ValueError(f"Missing CSV paths for: {', '.join(missing_paths)}")

        missing_prefixes = [
            alg for alg in algorithms if alg not in v.csv_column_prefixes
        ]
        if missing_prefixes:
            raise ValueError(
                f"Missing column prefixes for algorithms: {', '.join(missing_prefixes)}"
            )

        return v


def load_config(config_path: str) -> Config:
    """Load and validate configuration from JSON file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file) as f:
            config_data = json.load(f)

        config = Config(**config_data)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ConfigError(f"Configuration error: {e}")

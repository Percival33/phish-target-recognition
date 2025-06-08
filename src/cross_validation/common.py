#!/usr/bin/env python3
"""
Common utilities and constants for cross-validation modules.
Extracted to follow DRY principles.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Iterator, Tuple
import argparse
from tools.config import PROJ_ROOT


# Constants
class CVConstants:
    """Cross-validation constants"""

    # File names
    TRAIN_CSV = "train.csv"
    VAL_CSV = "val.csv"
    CONFIG_JSON = "config.json"
    LABELS_TXT = "labels.txt"

    # Directory patterns
    SPLIT_DIR_PATTERN = "split_{}"
    IMAGES_DIR = "images"

    # CSV columns
    TRAIN_COLUMNS = ["file", "true_target", "true_class"]
    VAL_COLUMNS = [
        "file",
        "true_target",
        "true_class",
        "pp_target",
        "pp_class",
        "vp_target",
        "vp_class",
        "baseline_target",
        "baseline_class",
    ]

    # Image extensions
    IMAGE_EXTENSIONS = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]


@dataclass
class DatasetConfig:
    """Dataset configuration data"""

    name: str
    path: str
    label_strategy: str
    target_mapping: Dict[str, str]


@dataclass
class CrossValidationConfig:
    """Cross-validation configuration data"""

    enabled: bool
    n_splits: int
    shuffle: bool
    random_state: int
    output_splits_directory: str
    dataset_configs: Dict[str, DatasetConfig]


@dataclass
class ValidationResult:
    """Result of validation check"""

    passed: bool
    message: str
    details: Dict = None


class ConfigLoader:
    """Universal configuration loader for cross-validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_config(
        self, config_path: str = CVConstants.CONFIG_JSON
    ) -> CrossValidationConfig:
        """Load and parse configuration from JSON file"""
        try:
            if config_path == CVConstants.CONFIG_JSON:
                config_path = str(Path(PROJ_ROOT) / config_path)

            with open(config_path, "r") as f:
                config = json.load(f)

            cv_config = config.get("cross_validation_config", {})

            if not cv_config.get("enabled", False):
                raise ValueError("Cross-validation is not enabled in configuration")

            dataset_configs = {}

            for name, info in cv_config.get("dataset_image_paths", {}).items():
                dataset_path = info["path"]
                if not Path(dataset_path).is_absolute():
                    dataset_path = str(Path(PROJ_ROOT) / dataset_path)

                dataset_configs[name] = DatasetConfig(
                    name=name,
                    path=dataset_path,
                    label_strategy=info["label_strategy"],
                    target_mapping=info["target_mapping"],
                )

            output_dir = cv_config.get("output_splits_directory", "data_splits")
            if not Path(output_dir).is_absolute():
                output_dir = str(Path(PROJ_ROOT) / output_dir)

            return CrossValidationConfig(
                enabled=cv_config["enabled"],
                n_splits=cv_config.get("n_splits", 3),
                shuffle=cv_config.get("shuffle", True),
                random_state=cv_config.get("random_state", 42),
                output_splits_directory=output_dir,
                dataset_configs=dataset_configs,
            )

        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise


class DirectoryIterator:
    """Utility for iterating over CV directory structures"""

    @staticmethod
    def iter_splits_and_datasets(
        splits_dir: Path, dataset_names: List[str], n_splits: int
    ) -> Iterator[Tuple[int, str, Path]]:
        """
        Iterate over all split/dataset combinations

        Yields:
            Tuple of (split_idx, dataset_name, dataset_dir)
        """
        for split_idx in range(n_splits):
            for dataset_name in dataset_names:
                dataset_dir = (
                    splits_dir
                    / CVConstants.SPLIT_DIR_PATTERN.format(split_idx)
                    / dataset_name
                )
                yield split_idx, dataset_name, dataset_dir

    @staticmethod
    def get_split_dir(splits_dir: Path, split_idx: int) -> Path:
        """Get directory for a specific split"""
        return splits_dir / CVConstants.SPLIT_DIR_PATTERN.format(split_idx)

    @staticmethod
    def get_dataset_dir(splits_dir: Path, split_idx: int, dataset_name: str) -> Path:
        """Get directory for a specific dataset in a split"""
        return DirectoryIterator.get_split_dir(splits_dir, split_idx) / dataset_name


class CVArgumentParser:
    """Common argument parser for CV modules"""

    @staticmethod
    def create_base_parser(description: str) -> argparse.ArgumentParser:
        """Create base argument parser with common CV arguments"""
        default_config = str(Path(PROJ_ROOT) / CVConstants.CONFIG_JSON)

        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--config", default=default_config, help="Config file path")
        parser.add_argument("--splits-dir", help="Splits directory (overrides config)")
        return parser


class PathUtils:
    """Common path utilities"""

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if needed"""
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_train_csv_path(dataset_dir: Path) -> Path:
        """Get path to train.csv file"""
        return dataset_dir / CVConstants.TRAIN_CSV

    @staticmethod
    def get_val_csv_path(dataset_dir: Path) -> Path:
        """Get path to val.csv file"""
        return dataset_dir / CVConstants.VAL_CSV

    @staticmethod
    def get_images_dir_path(dataset_dir: Path) -> Path:
        """Get path to images directory"""
        return dataset_dir / CVConstants.IMAGES_DIR


# Protocol for shared interfaces
class Validator(Protocol):
    """Interface for validation checks"""

    def validate(
        self, splits_dir: Path, config: CrossValidationConfig
    ) -> ValidationResult: ...

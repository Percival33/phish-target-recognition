#!/usr/bin/env python3
"""Simple dataset configuration loader."""

import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Simple dataset configuration handler."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.datasets = self.config.get("cross_validation_config", {}).get(
            "dataset_image_paths", {}
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file or return default."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Config file {config_path} not found, using Phishpedia default"
            )
            return {
                "cross_validation_config": {
                    "dataset_image_paths": {
                        "Phishpedia": {"label_strategy": "subfolders"}
                    }
                }
            }

    def get_available_datasets(self) -> List[str]:
        """Get available dataset names."""
        return list(self.datasets.keys())

    def get_split_structure(self, dataset_name: str) -> Dict[str, str]:
        """Get directory structure for dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        return {
            "train_benign": f"{dataset_name}/images/train/benign",
            "train_phish": f"{dataset_name}/images/train/phish",
            "val": f"{dataset_name}/images/val",
        }

#!/usr/bin/env python3
"""
Single script for dataset splitting into 60:30:10 train/val/test splits
with stratified sampling and optional symlink creation.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

from tools.config import setup_logging, PROJ_ROOT

# Add parent directory to sys.path to access cross_validation
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_validation.cv_splits import SimpleDataProcessor
from cross_validation.common import DatasetConfig


class DataSplitter:
    """Main class for splitting datasets into train/val/test splits"""

    def __init__(self, config_path: str):
        # Setup logging using tools
        setup_logging()
        self.logger = logging.getLogger(__name__)

        self.config = self.load_config(config_path)
        self.data_processor = SimpleDataProcessor()

    def load_config(self, config_path: str) -> Dict:
        """Load JSON configuration file"""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract data_split configuration
        if "data_split" not in config:
            raise ValueError("Configuration must contain 'data_split' key")

        return config["data_split"]

    def _resolve_output_path(self, output_dir: str) -> Path:
        """Resolve output directory path - prefix with PROJECT_ROOT if relative"""
        output_path = Path(output_dir)

        if output_path.is_absolute():
            return output_path
        else:
            return PROJ_ROOT / output_path

    def load_dataset(self, dataset_name: str, dataset_config: Dict) -> pd.DataFrame:
        """Load images and labels using existing components"""
        self.logger.info(f"Loading dataset: {dataset_name}")

        # Create DatasetConfig object
        ds_config = DatasetConfig(
            name=dataset_name,
            path=dataset_config["path"],
            label_strategy=dataset_config["label_strategy"],
            target_mapping=dataset_config["target_mapping"],
        )

        # Process dataset using existing processor
        data = self.data_processor.process_dataset(ds_config)

        self.logger.info(f"Loaded {len(data)} samples for {dataset_name}")
        return data

    def perform_stratified_split(
        self, X: np.ndarray, y: np.ndarray, y_stratify: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Two-stage stratified split for 60:30:10 proportions

        Algorithm:
        1. First split: train+val (90%) vs test (10%)
        2. Second split: train (60/90 = 66.67%) vs val (30/90 = 33.33%)

        Args:
            X: Array of indices
            y: Array of true_class values (for returning)
            y_stratify: Array of combined stratification keys (true_class + true_target)
        """
        random_state = self.config.get("random_state", 42)

        # Check if we have enough samples for stratification
        class_counts = Counter(y_stratify)

        # Identify and filter out classes with insufficient samples
        classes_to_drop = [cls for cls, count in class_counts.items() if count < 3]

        if classes_to_drop:
            # Log warnings for each dropped class combination
            self.logger.warning(
                f"Dropping {len(classes_to_drop)} class-target combinations with insufficient samples (<3):"
            )
            for cls in classes_to_drop:
                count = class_counts[cls]
                self.logger.warning(
                    f"  - Dropping combination '{cls}': only {count} sample{'s' if count != 1 else ''}"
                )

            # Create a mask to keep only valid samples
            mask = np.array([cls not in classes_to_drop for cls in y_stratify])

            # Filter the data
            X = X[mask]
            y = y[mask]
            y_stratify = y_stratify[mask]

            # Update class counts after filtering
            class_counts = Counter(y_stratify)

            self.logger.info(
                f"After filtering: {len(X)} samples remaining with {len(class_counts)} unique class-target combinations"
            )

        # Check if we have any samples left after filtering
        if len(X) == 0:
            raise ValueError(
                "No samples remaining after filtering classes with insufficient samples"
            )

        # Check if we still have enough unique classes for stratification
        if len(class_counts) < 2:
            raise ValueError(
                f"Insufficient unique class combinations for stratification. Only {len(class_counts)} combination(s) remaining."
            )

        self.logger.info(
            f"Splitting {len(X)} samples with {len(class_counts)} unique class-target combinations"
        )

        # First split: separate test set (10%)
        X_temp, X_test, y_temp, y_test, y_stratify_temp, y_stratify_test = (
            train_test_split(
                X,
                y,
                y_stratify,
                test_size=0.1,
                stratify=y_stratify,
                random_state=random_state,
            )
        )

        # Second split: separate train (60%) and val (30%) from remaining 90%
        val_ratio = 0.3 / 0.9  # 30% of total / 90% remaining = 33.33% of remaining
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            stratify=y_stratify_temp,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(self, data: pd.DataFrame, splits: Tuple, dataset_name: str):
        """Save train/val/test CSV files"""
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        # Create output directory using resolved path
        output_base = self._resolve_output_path(self.config["output_directory"])
        output_dir = output_base / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create split DataFrames
        train_df = data.iloc[X_train].copy()
        val_df = data.iloc[X_val].copy()
        test_df = data.iloc[X_test].copy()

        # Save CSV files
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        self.logger.info(f"Saved splits for {dataset_name} to {output_dir}")

        return train_df, val_df, test_df

    def create_symlinks(
        self, splits_data: Tuple, dataset_name: str, dataset_config: Dict
    ):
        """Optional: Create symlink structure for easier access"""
        if not dataset_config.get("create_symlinks", False):
            return

        train_df, val_df, test_df = splits_data

        # Create symlink directory structure using resolved path
        output_base = self._resolve_output_path(self.config["output_directory"])
        output_dir = output_base / dataset_name / "images"

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for _, row in split_df.iterrows():
                src_path = Path(row["file"])
                dst_path = split_dir / src_path.name

                # Create symlink if it doesn't exist
                if not dst_path.exists():
                    try:
                        dst_path.symlink_to(src_path.absolute())
                    except OSError as e:
                        self.logger.warning(f"Failed to create symlink {dst_path}: {e}")

        self.logger.info(f"Created symlinks for {dataset_name} in {output_dir}")

    def print_split_summary(
        self, splits: Tuple, dataset_name: str, data: pd.DataFrame = None
    ):
        """Print split statistics for visual verification"""
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        total = len(X_train) + len(X_val) + len(X_test)

        print(f"\n=== Split Summary for {dataset_name} ===")
        print(f"Total samples: {total}")
        print(f"Train: {len(X_train)} ({len(X_train)/total*100:.1f}%)")
        print(f"Val:   {len(X_val)} ({len(X_val)/total*100:.1f}%)")
        print(f"Test:  {len(X_test)} ({len(X_test)/total*100:.1f}%)")

        print("\n=== Class Distribution ===")
        print(f"Train - Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
        print(f"Val   - Class 0: {sum(y_val==0)}, Class 1: {sum(y_val==1)}")
        print(f"Test  - Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)}")

        # If data is provided, show target distribution statistics
        if data is not None:
            print("\n=== Target Distribution (sample counts per unique target) ===")
            train_data = data.iloc[X_train]
            val_data = data.iloc[X_val]
            test_data = data.iloc[X_test]

            # Count unique targets in each split
            train_targets = train_data["true_target"].nunique()
            val_targets = val_data["true_target"].nunique()
            test_targets = test_data["true_target"].nunique()
            total_targets = data["true_target"].nunique()

            print(f"Unique targets - Total: {total_targets}")
            print(f"Train: {train_targets} ({train_targets/total_targets*100:.1f}%)")
            print(f"Val:   {val_targets} ({val_targets/total_targets*100:.1f}%)")
            print(f"Test:  {test_targets} ({test_targets/total_targets*100:.1f}%)")

            # Show some examples of target preservation across splits
            print("\n=== Stratification Verification (5 sample targets) ===")
            sample_targets = data["true_target"].value_counts().head(5).index
            for target in sample_targets:
                orig_count = len(data[data["true_target"] == target])
                train_count = len(train_data[train_data["true_target"] == target])
                val_count = len(val_data[val_data["true_target"] == target])
                test_count = len(test_data[test_data["true_target"] == target])
                print(
                    f"Target '{target}': Total={orig_count}, Train={train_count}, Val={val_count}, Test={test_count}"
                )

    def process_all_datasets(self):
        """Main processing function for all datasets"""
        self.logger.info("Starting dataset splitting process")

        for dataset_name, dataset_config in self.config["datasets"].items():
            try:
                self.logger.info(f"Processing dataset: {dataset_name}")

                # Load dataset
                data = self.load_dataset(dataset_name, dataset_config)

                # Prepare data for splitting
                X = np.arange(len(data))  # indices
                y = data["true_class"].values

                # Create combined stratification key from true_class and true_target
                # This ensures both class and target distributions are preserved
                y_stratify = (
                    data["true_class"].astype(str)
                    + "_"
                    + data["true_target"].astype(str)
                )
                y_stratify = y_stratify.values

                # Perform stratified split
                splits = self.perform_stratified_split(X, y, y_stratify)

                # Save splits to CSV
                splits_data = self.save_splits(data, splits, dataset_name)

                # Create symlinks if requested
                self.create_symlinks(splits_data, dataset_name, dataset_config)

                # Print summary
                self.print_split_summary(splits, dataset_name, data)

            except Exception as e:
                self.logger.error(f"Failed to process dataset {dataset_name}: {e}")
                raise

        self.logger.info("\nDataset splitting completed successfully!")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Split datasets into stratified 60:30:10 train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json
  %(prog)s --config /path/to/config.json
        """,
    )

    parser.add_argument("config", nargs="?", help="Path to configuration JSON file")

    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        help="Path to configuration JSON file (alternative to positional argument)",
    )

    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Determine config path
    config_path = args.config or args.config_path

    if not config_path:
        parser.print_help()
        print("\nError: Configuration file is required")
        sys.exit(1)

    # Validate config file exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file '{config_path}' does not exist")
        sys.exit(1)

    try:
        splitter = DataSplitter(config_path)
        splitter.process_all_datasets()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

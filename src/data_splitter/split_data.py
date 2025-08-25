#!/usr/bin/env python3
"""
Single script for dataset splitting into 60:30:10 train/val/test splits
with stratified sampling and optional symlink creation.
Creates both visualphish and phishpedia formatted outputs from VisualPhish dataset.
"""

import json
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple, Set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import logging
import re

from tools.config import setup_logging, PROJ_ROOT

# Add parent directory to sys.path to access cross_validation
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_validation.cv_splits import SimpleDataProcessor
from cross_validation.common import DatasetConfig

# Import domain mapping from phishpedia organize script
sys.path.insert(
    0, str(Path(__file__).parent.parent / "models" / "phishpedia" / "scripts")
)
from organize import get_special_domain_mapping


class DataSplitter:
    """Main class for splitting datasets into train/val/test splits"""

    def __init__(self, config_path: str):
        # Setup logging using tools
        setup_logging()
        self.logger = logging.getLogger(__name__)

        self.config = self.load_config(config_path)
        self.data_processor = SimpleDataProcessor()

        # Get domain mapping for visualphish format
        self.domain_to_label = self._build_inverse_mapping(get_special_domain_mapping())
        # Add local overrides for domains not in the original mapping
        self.domain_to_label["miamidade.gov"] = (
            "mdpd"  # Map miamidade.gov to mdpd label
        )

    def _build_inverse_mapping(self, forward_map: Dict[str, str]) -> Dict[str, str]:
        """Invert mapping, ensuring no domain collisions."""
        inverse = {}
        for label, domain in forward_map.items():
            if domain in inverse and inverse[domain] != label:
                raise ValueError(f"Domain '{domain}' maps to multiple labels")
            inverse[domain] = label
        return inverse

    def _parse_true_target(self, true_target: str) -> str:
        """Extract domain from true_target (no longer assumes -- separator)."""
        if not true_target:
            return "unknown"

        # If it contains '--', take the part before it as domain
        if "--" in true_target:
            return true_target.split("--", 1)[0]

        # Otherwise, use the entire string as domain
        return true_target

    def _get_domain_suffix(self, target: str) -> str:
        """Extract domain suffix from target using special_domain_mapping."""
        domain_mapping = get_special_domain_mapping()

        if target in domain_mapping:
            full_domain = domain_mapping[target]
            # Split on first dot and take everything after
            if "." in full_domain:
                return full_domain.split(".", 1)[1]  # e.g., "absa.co.za" -> "co.za"

        # Default to .com for unknown domains
        return "com"

    def _check_duplicate_before_write(self, dest_file: Path, source_file: Path) -> None:
        """Raise error if file already exists with different content."""
        if dest_file.exists():
            if not dest_file.samefile(source_file):
                raise ValueError(
                    f"Duplicate filename collision: {dest_file} already exists "
                    f"and would be overwritten by {source_file}"
                )

    def _handle_empty_target(self, row: pd.Series) -> str:
        """Handle cases where true_target is None or empty."""
        if not row.get("true_target") or pd.isna(row["true_target"]):
            self.logger.warning(
                f"Empty or None true_target found for file: {row.get('file', 'unknown')}. Using 'unknown' as fallback."
            )
            return "unknown"
        return str(row["true_target"])

    def _validate_url_format(self, url: str, target: str, sample_num: str) -> bool:
        """Validate URL format before writing to info.txt."""
        expected_pattern = f"https://{target}+{sample_num}."
        if not url.startswith(expected_pattern):
            self.logger.error(
                f"Invalid URL format generated: {url}. Expected pattern: {expected_pattern}*"
            )
            return False
        return True

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

    def _create_visualphish_format(
        self, split_df: pd.DataFrame, output_dir: Path, split_name: str
    ) -> pd.DataFrame:
        """Create VisualPhish format: organized by target with phishing/trusted_list folders"""
        data_dir = output_dir / "data" / split_name
        phishing_dir = data_dir / "phishing"
        trusted_dir = data_dir / "trusted_list"
        phishing_dir.mkdir(parents=True, exist_ok=True)
        trusted_dir.mkdir(parents=True, exist_ok=True)

        new_paths = []
        phishing_labels = set()
        trusted_labels = set()

        # Track used filenames per directory to prevent collisions
        used_filenames: Dict[Path, Set[str]] = defaultdict(set)

        for idx, row in split_df.iterrows():
            source_path = Path(row["file"]).resolve()

            # Handle empty/None true_target with proper logging
            true_target = self._handle_empty_target(row)

            # Get domain (use full domain name, not abbreviation)
            domain = self._parse_true_target(true_target)

            # Use domain directly as folder name (full name, not abbreviation)
            label = domain

            self.logger.debug(
                f"Processing file {source_path} with target '{true_target}' -> domain '{domain}'"
            )

            # Choose destination directory
            is_phishing = int(row["true_class"]) == 1
            dest_base = phishing_dir if is_phishing else trusted_dir
            dest_dir = dest_base / label
            dest_dir.mkdir(exist_ok=True)

            # Preserve original filename structure
            original_stem = source_path.stem  # e.g., "T7_41" or "T0_1"

            # Parse T{X}_{Y} format if present, otherwise use original stem
            match = re.match(r"T(\d+)_(\d+)", original_stem)
            if match:
                x_num = match.group(1)
                y_num = int(match.group(2))
                base_identifier = f"T{x_num}"
            else:
                # Fallback if not in expected format - use original stem
                base_identifier = original_stem
                y_num = 0

            # Ensure uniqueness - generate unique Y value if collision detected
            identifier = f"{base_identifier}_{y_num}"
            while identifier in used_filenames[dest_dir]:
                y_num += 1
                identifier = f"{base_identifier}_{y_num}"

            used_filenames[dest_dir].add(identifier)

            # Create symlink or copy with collision detection
            dest_file = dest_dir / f"{identifier}{source_path.suffix}"

            # Check for file existence to prevent overwrites
            self._check_duplicate_before_write(dest_file, source_path)

            self.logger.debug(
                f"Creating {'symlink' if self.config.get('create_symlinks', True) else 'copy'}: {source_path} -> {dest_file}"
            )

            relative_path = dest_file.relative_to(data_dir)
            new_paths.append(str(relative_path))

            if self.config.get("create_symlinks", True):
                if not dest_file.exists():
                    dest_file.symlink_to(source_path)
            else:
                if not dest_file.exists():
                    shutil.copy2(source_path, dest_file)

            # Track labels
            (phishing_labels if is_phishing else trusted_labels).add(label)

        # Write target files
        if phishing_labels:
            (phishing_dir / "targets2.txt").write_text(
                "\n".join(sorted(phishing_labels)) + "\n"
            )
        if trusted_labels:
            (trusted_dir / "targets.txt").write_text(
                "\n".join(sorted(trusted_labels)) + "\n"
            )

        # Create DataFrame with new paths
        result_df = pd.DataFrame(
            {
                "original_path": split_df["file"].values,
                "new_path": new_paths,
                "true_target": split_df["true_target"].values,
                "true_class": split_df["true_class"].values,
            }
        )

        return result_df

    def _create_phishpedia_format(
        self, split_df: pd.DataFrame, output_dir: Path, split_name: str
    ) -> pd.DataFrame:
        """Create Phishpedia format: organized by sample with shot.png and info.txt"""
        data_dir = output_dir / "data" / split_name
        phishing_dir = data_dir / "phishing"
        trusted_dir = data_dir / "trusted_list"
        phishing_dir.mkdir(parents=True, exist_ok=True)
        trusted_dir.mkdir(parents=True, exist_ok=True)

        new_paths = []
        phishing_targets = set()
        trusted_targets = set()

        for idx, row in split_df.iterrows():
            source_path = Path(row["file"]).resolve()

            # Handle empty/None true_target with proper logging
            true_target = self._handle_empty_target(row)

            # Get target name (use domain from true_target)
            domain = self._parse_true_target(true_target)
            target = domain.lower() if domain else "unknown"

            self.logger.debug(
                f"Processing Phishpedia format for file {source_path} with target '{true_target}' -> '{target}'"
            )

            # Extract sample number from original filename
            original_stem = source_path.stem  # e.g., "T1_14"
            match = re.match(r"T\d+_(\d+)", original_stem)
            if match:
                sample_num = match.group(1)  # Extract Y number (e.g., "14")
            else:
                # Fallback: use index if filename doesn't match pattern
                sample_num = str(idx + 1)

            # Determine destination
            is_phishing = int(row["true_class"]) == 1
            if is_phishing:
                phishing_targets.add(target)
                parent_dir = phishing_dir
            else:
                trusted_targets.add(target)
                parent_dir = trusted_dir

            # Create sample directory
            sample_dir = parent_dir / f"{target}+sample{sample_num}"
            sample_dir.mkdir(exist_ok=True)

            # Create shot.png (symlink or copy)
            dest_file = sample_dir / "shot.png"
            relative_path = dest_file.relative_to(data_dir)
            new_paths.append(str(relative_path))

            if self.config.get("create_symlinks", True):
                if not dest_file.exists():
                    dest_file.symlink_to(source_path)
            else:
                if not dest_file.exists():
                    shutil.copy2(source_path, dest_file)

            # Create info.txt with proper URL
            info_file = sample_dir / "info.txt"
            if not info_file.exists():
                # Check if original has info.txt (for subfolders strategy)
                original_info = source_path.parent / "info.txt"
                if original_info.exists():
                    if self.config.get("create_symlinks", True):
                        info_file.symlink_to(original_info.resolve())
                    else:
                        shutil.copy2(original_info, info_file)
                else:
                    # Generate URL with proper domain suffix
                    domain_suffix = self._get_domain_suffix(target)
                    url = f"https://{target}+{sample_num}.{domain_suffix}"

                    # Validate URL format before writing
                    if self._validate_url_format(url, target, sample_num):
                        info_file.write_text(url)
                        self.logger.debug(
                            f"Generated URL for {target}+sample{sample_num}: {url}"
                        )
                    else:
                        # Write a fallback URL if validation fails
                        fallback_url = f"https://{target}+{sample_num}.com"
                        info_file.write_text(fallback_url)
                        self.logger.warning(
                            f"Used fallback URL due to validation failure: {fallback_url}"
                        )

        # Create DataFrame with new paths
        result_df = pd.DataFrame(
            {
                "original_path": split_df["file"].values,
                "new_path": new_paths,
                "true_target": split_df["true_target"].values,
                "true_class": split_df["true_class"].values,
            }
        )

        return result_df

    def save_splits(self, data: pd.DataFrame, splits: Tuple, dataset_name: str):
        """Save train/val/test CSV files and create visualphish/phishpedia formats"""
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        # Create output directory using resolved path
        output_base = self._resolve_output_path(self.config["output_directory"])
        output_dir = output_base / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create split DataFrames
        train_df = data.iloc[X_train].copy()
        val_df = data.iloc[X_val].copy()
        test_df = data.iloc[X_test].copy()

        # Save main CSV files (for backward compatibility)
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        # Create visualphish format
        vp_dir = output_dir / "visualphish"
        vp_dir.mkdir(exist_ok=True)

        vp_train_df = self._create_visualphish_format(train_df, vp_dir, "train")
        vp_val_df = self._create_visualphish_format(val_df, vp_dir, "val")
        vp_test_df = self._create_visualphish_format(test_df, vp_dir, "test")

        # Save visualphish CSVs
        vp_train_df.to_csv(vp_dir / "train.csv", index=False)
        vp_val_df.to_csv(vp_dir / "val.csv", index=False)
        vp_test_df.to_csv(vp_dir / "test.csv", index=False)

        self.logger.info(f"Created VisualPhish format for {dataset_name}")

        # Create phishpedia format
        pp_dir = output_dir / "phishpedia"
        pp_dir.mkdir(exist_ok=True)

        pp_train_df = self._create_phishpedia_format(train_df, pp_dir, "train")
        pp_val_df = self._create_phishpedia_format(val_df, pp_dir, "val")
        pp_test_df = self._create_phishpedia_format(test_df, pp_dir, "test")

        # Save phishpedia CSVs
        pp_train_df.to_csv(pp_dir / "train.csv", index=False)
        pp_val_df.to_csv(pp_dir / "val.csv", index=False)
        pp_test_df.to_csv(pp_dir / "test.csv", index=False)

        self.logger.info(f"Created Phishpedia format for {dataset_name}")

        self.logger.info(f"Saved all splits for {dataset_name} to {output_dir}")

        return train_df, val_df, test_df

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

                # Save splits to CSV and create formatted outputs
                _ = self.save_splits(data, splits, dataset_name)

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

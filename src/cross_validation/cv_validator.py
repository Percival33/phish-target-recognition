#!/usr/bin/env python3
"""
Cross-Validation Splits Validator - SOLID Principles Implementation

A simple validator that follows SOLID principles to validate generated CV splits.
"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
from collections import Counter
from tools.config import setup_logging
from common import (
    ValidationResult,
    Validator,
    ConfigLoader,
    DirectoryIterator,
    CVConstants,
    PathUtils,
    CVArgumentParser,
    CrossValidationConfig,
)


# Concrete Validators
class DirectoryStructureValidator:
    """Validates directory structure exists"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(
        self, splits_dir: Path, config: CrossValidationConfig
    ) -> ValidationResult:
        """Validate directory structure"""
        if not splits_dir.exists():
            return ValidationResult(
                False, f"Splits directory does not exist: {splits_dir}"
            )

        missing_splits = []
        missing_datasets = []

        for (
            split_idx,
            dataset_name,
            dataset_dir,
        ) in DirectoryIterator.iter_splits_and_datasets(
            splits_dir, list(config.dataset_configs.keys()), config.n_splits
        ):
            split_dir = DirectoryIterator.get_split_dir(splits_dir, split_idx)
            if not split_dir.exists():
                if f"split_{split_idx}" not in missing_splits:
                    missing_splits.append(f"split_{split_idx}")
                continue

            if not dataset_dir.exists():
                missing_datasets.append(f"{split_idx}/{dataset_name}")

        if missing_splits or missing_datasets:
            details = {
                "missing_splits": missing_splits,
                "missing_datasets": missing_datasets,
            }
            return ValidationResult(False, "Missing directories", details)

        return ValidationResult(True, "Directory structure is valid")


class CSVFilesValidator:
    """Validates CSV files exist and have correct structure"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(
        self, splits_dir: Path, config: CrossValidationConfig
    ) -> ValidationResult:
        """Validate CSV files"""
        missing_files = []
        invalid_files = []

        for (
            split_idx,
            dataset_name,
            dataset_dir,
        ) in DirectoryIterator.iter_splits_and_datasets(
            splits_dir, list(config.dataset_configs.keys()), config.n_splits
        ):
            # Check train.csv
            train_csv = PathUtils.get_train_csv_path(dataset_dir)
            if not train_csv.exists():
                missing_files.append(str(train_csv))
            else:
                train_validation = self._validate_train_csv(train_csv)
                if not train_validation.passed:
                    invalid_files.append(f"{train_csv}: {train_validation.message}")

            # Check val.csv
            val_csv = PathUtils.get_val_csv_path(dataset_dir)
            if not val_csv.exists():
                missing_files.append(str(val_csv))
            else:
                val_validation = self._validate_val_csv(val_csv, config)
                if not val_validation.passed:
                    invalid_files.append(f"{val_csv}: {val_validation.message}")

        if missing_files or invalid_files:
            details = {"missing_files": missing_files, "invalid_files": invalid_files}
            return ValidationResult(False, "CSV validation failed", details)

        return ValidationResult(True, "All CSV files are valid")

    def _validate_train_csv(self, csv_path: Path) -> ValidationResult:
        """Validate training CSV structure"""
        try:
            df = pd.read_csv(csv_path)
            required_columns = CVConstants.TRAIN_COLUMNS

            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return ValidationResult(False, f"Missing columns: {missing_columns}")

            if df.empty:
                return ValidationResult(False, "Empty CSV file")

            return ValidationResult(True, "Valid training CSV")

        except Exception as e:
            return ValidationResult(False, f"Error reading CSV: {e}")

    def _validate_val_csv(
        self, csv_path: Path, config: CrossValidationConfig
    ) -> ValidationResult:
        """Validate validation CSV structure"""
        try:
            df = pd.read_csv(csv_path)
            required_columns = CVConstants.get_val_columns(config.csv_column_prefixes)

            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return ValidationResult(False, f"Missing columns: {missing_columns}")

            if df.empty:
                return ValidationResult(False, "Empty CSV file")

            return ValidationResult(True, "Valid validation CSV")

        except Exception as e:
            return ValidationResult(False, f"Error reading CSV: {e}")


class ClassBalanceValidator:
    """Validates class balance across splits"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(
        self, splits_dir: Path, config: CrossValidationConfig
    ) -> ValidationResult:
        """Validate class balance"""
        balance_issues = []

        for dataset_name in config.dataset_configs.keys():
            dataset_balance = self._check_dataset_balance(
                splits_dir, dataset_name, config.n_splits
            )
            if not dataset_balance.passed:
                balance_issues.append(f"{dataset_name}: {dataset_balance.message}")

        if balance_issues:
            return ValidationResult(
                False, "Class balance issues", {"issues": balance_issues}
            )

        return ValidationResult(True, "Class balance is acceptable")

    def _check_dataset_balance(
        self, splits_dir: Path, dataset_name: str, n_splits: int
    ) -> ValidationResult:
        """Check balance for a single dataset"""
        split_distributions = []

        for split_idx in range(n_splits):
            dataset_dir = DirectoryIterator.get_dataset_dir(
                splits_dir, split_idx, dataset_name
            )
            val_csv = PathUtils.get_val_csv_path(dataset_dir)

            if val_csv.exists():
                try:
                    df = pd.read_csv(val_csv)
                    class_dist = Counter(df["true_class"])
                    split_distributions.append(class_dist)
                except Exception as e:
                    return ValidationResult(False, f"Error reading {val_csv}: {e}")

        if not split_distributions:
            return ValidationResult(False, "No validation files found")

        # Check if distributions are reasonably similar
        total_samples = [sum(dist.values()) for dist in split_distributions]
        if (
            max(total_samples) - min(total_samples) > max(total_samples) * 0.2
        ):  # 20% tolerance
            return ValidationResult(False, "Uneven split sizes")

        return ValidationResult(True, "Balanced splits")


class SymlinkValidator:
    """Validates symlinks if they exist"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(
        self, splits_dir: Path, config: CrossValidationConfig
    ) -> ValidationResult:
        """Validate symlinks"""
        broken_symlinks = []

        for (
            split_idx,
            dataset_name,
            dataset_dir,
        ) in DirectoryIterator.iter_splits_and_datasets(
            splits_dir, list(config.dataset_configs.keys()), config.n_splits
        ):
            images_dir = PathUtils.get_images_dir_path(dataset_dir)

            if images_dir.exists():
                broken = self._check_symlinks_in_dir(images_dir)
                if broken:
                    broken_symlinks.extend(
                        [f"{split_idx}/{dataset_name}: {link}" for link in broken]
                    )

        if broken_symlinks:
            return ValidationResult(
                False, "Broken symlinks found", {"broken_symlinks": broken_symlinks}
            )

        return ValidationResult(True, "All symlinks are valid")

    def _check_symlinks_in_dir(self, images_dir: Path) -> List[str]:
        """Check symlinks in a directory"""
        broken = []

        for item in images_dir.rglob("*"):
            if item.is_symlink() and not item.exists():
                broken.append(str(item.relative_to(images_dir)))

        return broken


class CrossValidationValidator:
    """Main validator orchestrator"""

    def __init__(self, validators: List[Validator]):
        self.validators = validators
        self.logger = logging.getLogger(__name__)

    def validate_splits(
        self, config_path: str = CVConstants.CONFIG_JSON, splits_dir: str = None
    ) -> bool:
        """Validate all splits"""
        try:
            # Load config
            config_loader = ConfigLoader()
            config = config_loader.load_config(config_path)

            # Determine splits directory
            if splits_dir is None:
                splits_dir = config.output_splits_directory

            splits_path = Path(splits_dir)

            self.logger.info(f"Validating splits in: {splits_path}")

            # Run all validators
            all_passed = True
            for validator in self.validators:
                result = validator.validate(splits_path, config)

                if result.passed:
                    self.logger.info(f"✓ {result.message}")
                else:
                    self.logger.error(f"✗ {result.message}")
                    if result.details:
                        for key, value in result.details.items():
                            self.logger.error(f"  {key}: {value}")
                    all_passed = False

            if all_passed:
                self.logger.info("All validation checks passed!")
            else:
                self.logger.error("Some validation checks failed!")

            return all_passed

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False


def main():
    """Main entry point"""
    parser = CVArgumentParser.create_base_parser(
        "Validate cross-validation data splits"
    )
    args = parser.parse_args()
    setup_logging()

    # Assemble validators
    validators = [
        DirectoryStructureValidator(),
        CSVFilesValidator(),
        ClassBalanceValidator(),
        SymlinkValidator(),
    ]

    # Create main validator
    validator = CrossValidationValidator(validators)

    # Execute validation
    success = validator.validate_splits(args.config, args.splits_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

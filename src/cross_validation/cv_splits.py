import logging
from pathlib import Path
from typing import Dict, List, Tuple, Protocol, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import glob
from tools.config import setup_logging
from common import (
    DatasetConfig,
    CrossValidationConfig,
    ConfigLoader,
    CVConstants,
    PathUtils,
    CVArgumentParser,
)


class DataProcessor(Protocol):
    """Interface for data processing"""

    def process_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame: ...


class SplitGenerator(Protocol):
    """Interface for split generation"""

    def generate_splits(
        self, data: pd.DataFrame, n_splits: int, random_state: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]: ...


class FileWriter(Protocol):
    """Interface for file writing"""

    def write_split_files(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        split_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        output_dir: Path,
    ): ...


class SymlinkManager(Protocol):
    """Interface for symlink management"""

    def create_symlinks(
        self,
        data: pd.DataFrame,
        dataset_config: DatasetConfig,
        split_dir: Path,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ): ...


# Concrete Implementations
class ImageFileDiscoverer:
    """Utility for discovering image files"""

    @classmethod
    def get_image_files(cls, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        files = []
        for ext in CVConstants.IMAGE_EXTENSIONS:
            pattern_files = glob.glob(str(directory / ext))
            pattern_files.extend(glob.glob(str(directory / ext.upper())))
            files.extend(pattern_files)
        return [Path(f) for f in files]


class LabelExtractor:
    """Extracts labels from labels.txt files"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_labels(self, labels_file: Path) -> List[str]:
        """Extract labels from labels.txt file"""
        if not labels_file.exists():
            return []

        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Extracted {len(labels)} labels from {labels_file}")
            return labels
        except Exception as e:
            self.logger.warning(f"Failed to read labels from {labels_file}: {e}")
            return []


class SimpleDataProcessor:
    """Data processor implementing multiple label strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_discoverer = ImageFileDiscoverer()
        self.label_extractor = LabelExtractor()

    def process_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        base_path = Path(dataset_config.path)
        samples = []

        for target_type, subdirectory in dataset_config.target_mapping.items():
            target_path = base_path / subdirectory
            class_value = 1 if target_type == "phishing" else 0

            if dataset_config.label_strategy == "directory":
                samples.extend(
                    self._process_directory_strategy(
                        target_path, target_type, class_value
                    )
                )
            elif dataset_config.label_strategy == "labels_file":
                samples.extend(
                    self._process_labels_file_strategy(
                        target_path, target_type, class_value
                    )
                )
            elif dataset_config.label_strategy == "subfolders":
                samples.extend(
                    self._process_subfolders_strategy(
                        target_path, target_type, class_value
                    )
                )
            else:
                self.logger.warning(
                    f"Unknown label strategy: {dataset_config.label_strategy}"
                )

        df = pd.DataFrame(samples)

        # Validate no duplicate targets across classes
        # self._validate_no_duplicate_targets(df, dataset_config.name)

        self.logger.info(
            f"{dataset_config.name}: {len(df)} samples, classes: {Counter(df['true_class'])}"
        )
        return df

    def _process_directory_strategy(
        self, target_path: Path, target_type: str, class_value: int
    ) -> List[Dict[str, Any]]:
        """Process using directory strategy: images organized in class subdirectories"""
        samples = []

        if not target_path.exists():
            self.logger.warning(f"Target path does not exist: {target_path}")
            return samples

        # For directory strategy, expect subdirectories named by target/brand
        for subdir in target_path.iterdir():
            if subdir.is_dir():
                brand_name = subdir.name
                image_files = self.file_discoverer.get_image_files(subdir)

                for img_file in image_files:
                    samples.append(
                        {
                            "file": str(img_file),
                            "true_target": brand_name,
                            "true_class": class_value,
                        }
                    )

        return samples

    def _process_labels_file_strategy(
        self, target_path: Path, target_type: str, class_value: int
    ) -> List[Dict[str, Any]]:
        """Process using labels_file strategy: flat structure with labels.txt"""
        samples = []

        if not target_path.exists():
            self.logger.warning(f"Target path does not exist: {target_path}")
            return samples

        # Get all image files in the directory
        image_files = sorted(self.file_discoverer.get_image_files(target_path))

        # Read labels from labels.txt
        labels_file = target_path / CVConstants.LABELS_TXT
        labels = self.label_extractor.extract_labels(labels_file)

        # Match images with labels
        for i, img_file in enumerate(image_files):
            if target_type == "benign":
                # For benign, use label from labels.txt if available, otherwise "benign"
                target_label = labels[i] if i < len(labels) and labels[i] else "benign"
            else:
                # For phishing, use label from labels.txt or default
                target_label = labels[i] if i < len(labels) else "phishing"

            samples.append(
                {
                    "file": str(img_file),
                    "true_target": target_label,
                    "true_class": class_value,
                }
            )

        return samples

    def _process_subfolders_strategy(
        self, target_path: Path, target_type: str, class_value: int
    ) -> List[Dict[str, Any]]:
        """Process using subfolders strategy: nested subdirs each containing shot.png and info.txt"""
        samples = []

        if not target_path.exists():
            self.logger.warning(f"Target path does not exist: {target_path}")
            return samples

        # For subfolders strategy: iterate through subdirectories, each containing shot.png
        for subdir in target_path.iterdir():
            if subdir.is_dir():
                shot_file = subdir / "shot.png"

                if shot_file.exists():
                    if target_type == "benign":
                        if subdir.name and subdir.name != "benign":
                            target_label = subdir.name
                        else:
                            target_label = "benign"
                    else:
                        # We can not remove timestamp (split on "+") as we lose all samples but one from the same target
                        # Example:
                        # 1&1 Ionos+2020-08-13-19`10`20/shot.png
                        # 1&1 Ionos+2020-08-16-16`00`55/shot.png
                        # Only one of them will be kept. Due to duplication of target name `1&1 Ionos`
                        # in PerSampleSymlinkManager._create_sample_symlinks() method
                        target_label = subdir.name

                    samples.append(
                        {
                            "file": str(shot_file),
                            "true_target": target_label,
                            "true_class": class_value,
                        }
                    )

        return samples

    def _validate_no_duplicate_targets(self, df: pd.DataFrame, dataset_name: str):
        """Validate that no target appears in both phishing and benign classes"""
        if df.empty:
            return

        phishing_targets = set(df[df["true_class"] == 1]["true_target"].unique())
        benign_targets = set(df[df["true_class"] == 0]["true_target"].unique())

        duplicate_targets = phishing_targets.intersection(benign_targets)

        if duplicate_targets:
            duplicate_list = sorted(list(duplicate_targets))
            raise ValueError(
                f"Duplicate target found in dataset {dataset_name}: {duplicate_list}"
            )


class StratifiedSplitGenerator:
    """Generates stratified K-fold splits"""

    def generate_splits(
        self, data: pd.DataFrame, n_splits: int, random_state: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified splits"""
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        return list(skf.split(data, data["true_class"]))


class CSVFileWriter:
    """Writes CSV files for training and evaluation"""

    def __init__(self, csv_column_prefixes: Dict[str, str]):
        self.logger = logging.getLogger(__name__)
        self.csv_column_prefixes = csv_column_prefixes

    def write_split_files(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        split_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        output_dir: Path,
    ):
        """Write train and validation CSV files"""
        split_dir = output_dir / f"split_{split_idx}" / dataset_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Training CSV - simple format
        train_df = data.iloc[train_idx][CVConstants.TRAIN_COLUMNS].copy()
        train_df.to_csv(split_dir / CVConstants.TRAIN_CSV, index=False)

        # Validation CSV - with empty prediction columns
        val_df = self._create_evaluation_dataframe(data.iloc[val_idx])
        val_df.to_csv(split_dir / CVConstants.VAL_CSV, index=False)

        self.logger.info(
            f"Saved {dataset_name} split {split_idx}: {len(train_df)} train, {len(val_df)} val"
        )

    def _create_evaluation_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create evaluation DataFrame with empty prediction columns"""
        eval_df = data[CVConstants.TRAIN_COLUMNS].copy()

        # Add empty prediction columns for each algorithm using configurable prefixes
        for algorithm, prefix in self.csv_column_prefixes.items():
            eval_df[f"{prefix}_target"] = ""
            eval_df[f"{prefix}_class"] = ""

        return eval_df


class PerSampleSymlinkManager:
    """Creates symlinks for each sample individually"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_symlinks(
        self,
        data: pd.DataFrame,
        dataset_config: DatasetConfig,
        split_dir: Path,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ):
        """Create symlinks for each sample in the split"""
        images_dir = PathUtils.get_images_dir_path(split_dir)
        PathUtils.ensure_directory(images_dir)

        # Create symlinks for training data
        train_data = data.iloc[train_idx]
        self._create_sample_symlinks(
            train_data,
            images_dir,
            "train",
            dataset_config,
        )

        # Create symlinks for validation data
        val_data = data.iloc[val_idx]
        self._create_sample_symlinks(val_data, images_dir, "val", dataset_config)

        self.logger.info(f"Created symlinks in {images_dir}")

    def _create_sample_symlinks(
        self,
        data: pd.DataFrame,
        images_dir: Path,
        split_type: str,
        dataset_config: DatasetConfig,
    ):
        """Create symlinks for samples in a split"""
        split_images_dir = images_dir / split_type
        split_images_dir.mkdir(parents=True, exist_ok=True)

        base_path = Path(".").resolve()  # Absolute path for symlinks

        for _, row in data.iterrows():
            original_file = base_path / row["file"]
            normalized_parent = str(original_file.parent.name).split("+")[
                0
            ]  # Normalize parent for symlink

            if split_type == "train":
                target_dir = (
                    split_images_dir
                    / ("phish" if row["true_class"] == 1 else "benign")
                    / row["true_target"]
                )
            else:
                target_dir = split_images_dir / row["true_target"]
            target_dir.mkdir(parents=True, exist_ok=True)

            # Create symlink for the main file (shot.png)
            symlink_path = target_dir / original_file.name
            try:
                if not symlink_path.exists():
                    symlink_path.symlink_to(original_file)
            except OSError as e:
                self.logger.warning(f"Failed to create symlink {symlink_path}: {e}")
                continue

            info_file = original_file.parent / "info.txt"
            info_symlink_path = target_dir / "info.txt"
            if info_file.exists():
                try:
                    if not info_symlink_path.exists():
                        info_symlink_path.symlink_to(info_file)
                except OSError as e:
                    self.logger.warning(
                        f"Failed to create info.txt symlink {info_symlink_path}: {e}"
                    )
            else:
                self.logger.warning(
                    f"info.txt not found for {original_file}, creating symlink {info_symlink_path}"
                )
                with info_symlink_path.open("w", encoding="utf-8") as f:
                    f.write(normalized_parent)


class CrossValidationSplitter:
    def __init__(
        self,
        config_loader: ConfigLoader,
        data_processor: DataProcessor,
        split_generator: SplitGenerator,
        file_writer: FileWriter,
        symlink_manager: SymlinkManager = None,
    ):
        self.config_loader = config_loader
        self.data_processor = data_processor
        self.split_generator = split_generator
        self.file_writer = file_writer
        self.symlink_manager = symlink_manager
        self.logger = logging.getLogger(__name__)

    def create_splits(
        self, config_path: str = CVConstants.CONFIG_JSON, create_symlinks: bool = False
    ) -> bool:
        try:
            config = self.config_loader.load_config(config_path)
            output_dir = Path(config.output_splits_directory)
            PathUtils.ensure_directory(output_dir)

            for dataset_config in config.dataset_configs.values():
                self._process_single_dataset(
                    dataset_config, config, output_dir, create_symlinks
                )

            self.logger.info(f"Cross-validation splits created in {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create splits: {e}")
            return False

    def _process_single_dataset(
        self,
        dataset_config: DatasetConfig,
        config: CrossValidationConfig,
        output_dir: Path,
        create_symlinks: bool,
    ):
        self.logger.info(f"Processing {dataset_config.name}")

        # Load dataset
        data = self.data_processor.process_dataset(dataset_config)

        if data.empty:
            self.logger.warning(f"No samples found for {dataset_config.name}")
            return

        # Generate splits
        splits = self.split_generator.generate_splits(
            data, config.n_splits, config.random_state
        )

        # Create split files
        for split_idx, (train_idx, val_idx) in enumerate(splits):
            split_dir = output_dir / f"split_{split_idx}" / dataset_config.name

            # Write CSV files
            self.file_writer.write_split_files(
                data, dataset_config.name, split_idx, train_idx, val_idx, output_dir
            )

            # Create symlinks if requested
            if create_symlinks and self.symlink_manager:
                self.symlink_manager.create_symlinks(
                    data, dataset_config, split_dir, train_idx, val_idx
                )


def main():
    parser = CVArgumentParser.create_base_parser("Cross-validation data splits")
    parser.add_argument(
        "--create-symlinks", action="store_true", help="Create symlinks for each sample"
    )

    args = parser.parse_args()
    setup_logging()

    # Load config first to get CSV column prefixes
    config = ConfigLoader().load_config(args.config)

    splitter = CrossValidationSplitter(
        config_loader=(ConfigLoader()),
        data_processor=(SimpleDataProcessor()),
        split_generator=(StratifiedSplitGenerator()),
        file_writer=(CSVFileWriter(config.csv_column_prefixes)),
        symlink_manager=(PerSampleSymlinkManager() if args.create_symlinks else None),
    )

    success = splitter.create_splits(args.config, args.create_symlinks)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

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
    """Simple data processor following Single Responsibility Principle"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_discoverer = ImageFileDiscoverer()
        self.label_extractor = LabelExtractor()

    def process_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        """Process dataset and return DataFrame with all samples"""
        base_path = Path(dataset_config.path)
        samples = []

        for target_type, subdirectory in dataset_config.target_mapping.items():
            target_path = base_path / subdirectory

            if target_type == "phishing":
                samples.extend(
                    self._process_phishing_directory(
                        target_path, dataset_config, base_path
                    )
                )
            else:  # benign
                samples.extend(
                    self._process_benign_directory(
                        target_path, base_path, dataset_config
                    )
                )

        df = pd.DataFrame(samples)
        self.logger.info(
            f"{dataset_config.name}: {len(df)} samples, classes: {Counter(df['true_class'])}"
        )
        return df

    def _process_phishing_directory(
        self, target_path: Path, dataset_config: DatasetConfig, base_path: Path
    ) -> List[Dict[str, Any]]:
        """Process phishing directory based on dataset structure"""
        samples = []

        if dataset_config.name == "Phishpedia":
            # Phishpedia structure: subdirs with shot.png
            for subdir in target_path.iterdir():
                if subdir.is_dir():
                    shot_file = subdir / "shot.png"
                    if shot_file.exists():
                        target = self._get_target_for_phishpedia(subdir, target_path)
                        samples.append(
                            {
                                "file": str(shot_file),
                                "true_target": target,
                                "true_class": 1,
                            }
                        )
        else:
            # Other datasets: direct image files
            image_files = self.file_discoverer.get_image_files(target_path)
            labels = self.label_extractor.extract_labels(
                target_path / CVConstants.LABELS_TXT
            )

            for i, img_file in enumerate(image_files):
                target = labels[i] if i < len(labels) else "phishing"
                samples.append(
                    {"file": str(img_file), "true_target": target, "true_class": 1}
                )

        return samples

    def _process_benign_directory(
        self, target_path: Path, base_path: Path, dataset_config: DatasetConfig = None
    ) -> List[Dict[str, Any]]:
        """Process benign directory"""
        samples = []

        # Special handling for Phishpedia benign structure (subdirs with shot.png)
        if dataset_config and dataset_config.name == "Phishpedia":
            for subdir in target_path.iterdir():
                if subdir.is_dir():
                    shot_file = subdir / "shot.png"
                    if shot_file.exists():
                        samples.append(
                            {
                                "file": str(shot_file),
                                "true_target": "benign",
                                "true_class": 0,
                            }
                        )
        else:
            # Other datasets: direct image files
            image_files = self.file_discoverer.get_image_files(target_path)
            for img_file in image_files:
                samples.append(
                    {"file": str(img_file), "true_target": "benign", "true_class": 0}
                )

        return samples

    def _get_target_for_phishpedia(self, subdir: Path, target_path: Path) -> str:
        """Get target label for Phishpedia subdirectory"""
        labels = self.label_extractor.extract_labels(
            target_path / CVConstants.LABELS_TXT
        )
        return labels[0] if labels else subdir.name


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

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

        # Add empty prediction columns for each algorithm
        for alg in ["pp", "vp", "baseline"]:
            eval_df[f"{alg}_target"] = ""
            eval_df[f"{alg}_class"] = ""

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
        self._create_sample_symlinks(train_data, images_dir, "train", dataset_config)

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

            if original_file.exists():
                # Create meaningful symlink name
                target_dir = split_images_dir / row["true_target"]
                target_dir.mkdir(parents=True, exist_ok=True)

                symlink_path = target_dir / original_file.name

                try:
                    # Create symlink
                    if not symlink_path.exists():
                        symlink_path.symlink_to(original_file)
                except OSError as e:
                    self.logger.warning(f"Failed to create symlink {symlink_path}: {e}")


class CrossValidationSplitter:
    """Main orchestrator following Dependency Inversion Principle"""

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
        """Create cross-validation splits"""
        try:
            # Load configuration
            config = self.config_loader.load_config(config_path)
            output_dir = Path(config.output_splits_directory)
            PathUtils.ensure_directory(output_dir)

            # Process each dataset
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
        """Process a single dataset"""
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
    """Main entry point"""
    parser = CVArgumentParser.create_base_parser(
        "Cross-validation data splits with SOLID principles"
    )
    parser.add_argument(
        "--create-symlinks", action="store_true", help="Create symlinks for each sample"
    )

    args = parser.parse_args()
    setup_logging()

    # Dependency injection - assemble components
    config_loader = ConfigLoader()
    data_processor = SimpleDataProcessor()
    split_generator = StratifiedSplitGenerator()
    file_writer = CSVFileWriter()
    symlink_manager = PerSampleSymlinkManager() if args.create_symlinks else None

    # Create main splitter
    splitter = CrossValidationSplitter(
        config_loader=config_loader,
        data_processor=data_processor,
        split_generator=split_generator,
        file_writer=file_writer,
        symlink_manager=symlink_manager,
    )

    # Execute
    success = splitter.create_splits(args.config, args.create_symlinks)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

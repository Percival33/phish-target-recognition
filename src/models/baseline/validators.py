import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_images_dir(images_dir: Path) -> None:
    """Validate that images directory exists."""
    if not images_dir.exists():
        logger.error(f"Images directory {images_dir} does not exist")
        sys.exit(1)
    if not images_dir.is_dir():
        logger.error(f"Path {images_dir} is not a directory")
        sys.exit(1)


def validate_index_path(
    index_path: Path, must_exist: bool = False, overwrite: bool = False
) -> None:
    """Validate index path based on requirements."""
    if must_exist:
        if not index_path.exists():
            logger.error(f"Index file {index_path} does not exist")
            sys.exit(1)
    elif index_path.exists() and not overwrite:
        logger.error(
            f"Index file {index_path} already exists. Use --overwrite to replace"
        )
        sys.exit(1)


def validate_metadata_path(
    metadata_path: Path, must_exist: bool = False, overwrite: bool = False
) -> None:
    """Validate metadata CSV path based on requirements."""
    if must_exist:
        if not metadata_path.exists():
            logger.error(f"Metadata file {metadata_path} does not exist")
            sys.exit(1)
    elif metadata_path.exists() and not overwrite:
        logger.error(
            f"Metadata file {metadata_path} already exists. Use --overwrite to replace"
        )
        sys.exit(1)


def validate_output_path(output_path: Path, overwrite: bool = False) -> None:
    """Validate output path."""
    if output_path.exists() and not overwrite:
        logger.error(
            f"Output file {output_path} already exists. Use --overwrite to replace"
        )
        sys.exit(1)


def validate_labels_path(labels_path: Optional[Path]) -> None:
    """Validate labels path if provided."""
    if labels_path and not labels_path.exists():
        logger.error(f"Labels file {labels_path} does not exist")
        sys.exit(1)


def validate_threshold(threshold: Optional[float]) -> None:
    """Validate threshold value if provided."""
    if threshold is not None and not (0 <= threshold <= 100):
        logger.error(f"Threshold must be between 0 and 100, got {threshold}")
        sys.exit(1)


def validate_top_k(top_k: int) -> None:
    """Validate top-k parameter value."""
    if not isinstance(top_k, int):
        logger.error(f"Top-k must be an integer, got {type(top_k)}")
        sys.exit(1)
    if top_k <= 0:
        logger.error(f"Top-k must be positive, got {top_k}")
        sys.exit(1)

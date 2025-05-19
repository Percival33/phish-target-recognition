import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_image_paths(images_dir: Path) -> List[Path]:
    """Get list of image paths from directory."""
    image_paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(images_dir.glob(ext))

    try:
        image_paths.sort(key=lambda p: int(p.stem))
        logger.debug("Successfully sorted images by numeric prefix")
    except ValueError:
        logger.debug("Images were not numbered, skipping sorting")

    if not image_paths:
        logger.error(f"No images found in directory {images_dir}")
        sys.exit(1)

    return image_paths


def load_labels(labels_path: Optional[Path]) -> Optional[List[str]]:
    """Load target labels from file if provided."""
    if not labels_path:
        return None

    try:
        with open(labels_path) as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        logger.error(f"Failed to load labels from {labels_path}: {e}")
        return None

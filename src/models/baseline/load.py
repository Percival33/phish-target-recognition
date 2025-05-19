import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

from .BaselineEmbedder import BaselineEmbedder
from .validators import validate_images_dir, validate_index_path, validate_labels_path
from .common import get_image_paths, load_labels
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def validate_inputs(
    images_dir: Path, index_path: Path, labels_path: Optional[Path], overwrite: bool
) -> List[Path]:
    """Validate all input parameters and return list of valid image paths."""
    validate_images_dir(images_dir)
    validate_index_path(index_path, must_exist=False, overwrite=overwrite)
    validate_labels_path(labels_path)
    return get_image_paths(images_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Load and index images for baseline perceptual hash service"
    )
    parser.add_argument(
        "--images", type=str, required=True, help="Path to images directory"
    )
    parser.add_argument(
        "--index", type=str, required=True, help="Path to save/update FAISS index"
    )
    parser.add_argument(
        "--labels", type=str, help="Path to labels.txt file containing target labels"
    )
    parser.add_argument(
        "--is-phish", action="store_true", help="Mark all images as phishing"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for processing"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing index"
    )

    args = parser.parse_args()

    # Convert paths
    images_dir = Path(args.images)
    index_path = Path(args.index)
    metadata_path = index_path.with_suffix(".csv")
    labels_path = Path(args.labels) if args.labels else None

    # Validate inputs and get image paths
    image_paths = validate_inputs(images_dir, index_path, labels_path, args.overwrite)

    # Load target labels if provided
    target_labels = load_labels(labels_path)
    if target_labels and len(target_labels) != len(image_paths):
        logger.error(
            f"Number of labels ({len(target_labels)}) does not match number of images ({len(image_paths)})"
        )
        sys.exit(1)

    embedder = BaselineEmbedder(
        index_path=index_path if not args.overwrite else None,
        metadata_path=metadata_path if not args.overwrite else None,
    )

    try:
        # Compute embeddings
        logger.info(
            f"Processing {len(image_paths)} images with batch size {args.batch_size}"
        )
        _, metadata = embedder.compute_embeddings(
            image_paths=image_paths, is_phish=args.is_phish, batch_size=args.batch_size
        )

        if not embedder.save_index(index_path, overwrite=args.overwrite):
            logger.error("Failed to save index")
            sys.exit(1)

        if not embedder.save_metadata_csv(metadata_path, overwrite=args.overwrite):
            logger.error("Failed to save metadata")
            sys.exit(1)

        logger.info(
            f"Successfully processed {len(metadata)} images and saved index to {index_path} and metadata to {metadata_path}"
        )

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

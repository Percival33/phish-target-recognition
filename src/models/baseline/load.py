import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import wandb

from BaselineEmbedder import BaselineEmbedder
from checkers import (
    validate_images_dir,
    validate_index_path,
    validate_labels_path,
    validate_metadata_path,
)
from common import get_image_paths
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def validate_inputs(images_dir: Path, labels_path: Optional[Path]) -> None:
    """Validate input parameters."""
    validate_images_dir(images_dir)
    validate_labels_path(labels_path)


def main():
    parser = argparse.ArgumentParser(
        description="Load and index images for baseline perceptual hash service",
        epilog="""
Operation modes:
  Default (no flags): Create new index/metadata files. Fails if files already exist.
  --overwrite: Replace existing index/metadata files completely.
  --append: Add new images to existing index/metadata files.

Examples:
  python load.py --images /path/to/images --index model.faiss  # Create new
  python load.py --images /path/to/images --index model.faiss --overwrite  # Replace
  python load.py --images /path/to/images --index model.faiss --append  # Append
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # Mutually exclusive group for operation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index and metadata files",
    )
    mode_group.add_argument(
        "--append",
        action="store_true",
        help="Append new images to existing index and metadata files",
    )

    parser.add_argument("--log", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    if args.append:
        operation_mode = "append"
    elif args.overwrite:
        operation_mode = "overwrite"
    else:
        operation_mode = "create"

    logger.info(f"Operation mode: {operation_mode}")

    if args.log:
        run = wandb.init(
            project="baseline-load",
            group="baseline",
            config=args,
        )

    images_dir = Path(args.images)
    index_path = Path(args.index)
    metadata_path = index_path.with_suffix(".csv")
    labels_path = Path(args.labels) if args.labels else None

    if operation_mode == "append":
        validate_index_path(index_path, must_exist=True)
        validate_metadata_path(metadata_path, must_exist=True)
        logger.info(
            f"Appending to existing index: {index_path} and metadata: {metadata_path}"
        )
    elif operation_mode == "create":
        validate_index_path(index_path, must_exist=False, overwrite=False)
        validate_metadata_path(metadata_path, must_exist=False, overwrite=False)
        logger.info(f"Creating new index: {index_path} and metadata: {metadata_path}")
    else:
        logger.info(f"Overwriting index: {index_path} and metadata: {metadata_path}")

    validate_inputs(images_dir, labels_path)
    image_paths = get_image_paths(images_dir)

    num_images = len(image_paths)
    true_classes_list = [1] * num_images if args.is_phish else [0] * num_images

    embedder = BaselineEmbedder(
        index_path=index_path if operation_mode == "append" else None,
        metadata_path=metadata_path if operation_mode == "append" else None,
    )

    try:
        logger.info(
            f"Processing {len(image_paths)} images with batch size {args.batch_size}"
        )
        _, metadata = embedder.compute_embeddings(
            image_paths=image_paths,
            true_classes=true_classes_list,
            batch_size=args.batch_size,
        )

        if not embedder.save_index(index_path, overwrite=True):
            logger.error("Failed to save index")
            sys.exit(1)

        if not embedder.save_metadata_csv(metadata_path, overwrite=True):
            logger.error("Failed to save metadata")
            sys.exit(1)

        if args.log:
            try:
                logger.info(f"Load results for {len(metadata)} images")
                run.save(str(index_path))
                run.save(str(metadata_path))
            except Exception as e:
                logger.warning(f"Failed to log wandb artifacts: {e}")

        if operation_mode == "append":
            total_images = embedder.index.ntotal if embedder.index else len(metadata)
            logger.info(
                f"Successfully appended {len(metadata)} new images. "
                f"Index now contains {total_images} total images. "
                f"Saved to {index_path} and {metadata_path}"
            )
        else:
            logger.info(
                f"Successfully processed {len(metadata)} images and saved index to {index_path} and metadata to {metadata_path}"
            )

        if args.log:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

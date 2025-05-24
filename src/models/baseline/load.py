import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

import wandb

from BaselineEmbedder import BaselineEmbedder
from checkers import validate_images_dir, validate_index_path, validate_labels_path
from common import get_image_paths
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
    parser.add_argument("--log", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()
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

    validate_inputs(images_dir, index_path, labels_path, args.overwrite)
    image_paths = get_image_paths(images_dir)

    # Prepare true_classes_list based on args.is_phish
    num_images = len(image_paths)
    true_classes_list = [1] * num_images if args.is_phish else [0] * num_images

    # target_labels = load_labels(labels_path)
    # if target_labels and len(target_labels) != len(image_paths):
    #     logger.error(
    #         f"Number of labels ({len(target_labels)}) does not match number of images ({len(image_paths)})"
    #     )
    #     sys.exit(1)

    embedder = BaselineEmbedder(
        index_path=index_path if not args.overwrite else None,
        metadata_path=metadata_path if not args.overwrite else None,
    )

    try:
        # Compute embeddings
        logger.info(
            # 9363
            f"Processing {len(image_paths)} images with batch size {args.batch_size}"
        )
        _, metadata = embedder.compute_embeddings(
            image_paths=image_paths,
            true_classes=true_classes_list,
            batch_size=args.batch_size,
        )

        if not embedder.save_index(index_path, overwrite=args.overwrite):
            logger.error("Failed to save index")
            sys.exit(1)

        if not embedder.save_metadata_csv(metadata_path, overwrite=args.overwrite):
            logger.error("Failed to save metadata")
            sys.exit(1)

        # Log artifacts as wandb artifacts if logging enabled
        if args.log:
            try:
                logger.info(f"Load results for {len(metadata)} images")
                run.save(str(index_path))
                run.save(str(metadata_path))
            except Exception as e:
                logger.warning(f"Failed to log wandb artifacts: {e}")

        logger.info(
            f"Successfully processed {len(metadata)} images and saved index to {index_path} and metadata to {metadata_path}"
        )

        # Finish wandb run if logging enabled
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

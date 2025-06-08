import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import wandb

from BaselineEmbedder import BaselineEmbedder
from checkers import (
    validate_images_dir,
    validate_index_path,
    validate_output_path,
    validate_threshold,
    validate_metadata_path,
    validate_labels_path,
    validate_top_k,
)
from common import get_image_paths, load_labels
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def validate_inputs(
    images_dir: Path,
    index_path: Path,
    output_path: Path,
    labels_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    overwrite: bool = False,
    top_k: int = 1,
    is_phish_cli_arg: bool = False,
) -> Tuple[List[Path], Optional[List[str]], List[int]]:
    """Validate input parameters and return list of valid image paths, labels, and true classes.

    Args:
        images_dir: Directory containing query images
        index_path: Path to existing FAISS index
        output_path: Path to save results CSV
        labels_path: Optional path to labels.txt file
        threshold: Optional distance threshold for classification
        overwrite: Whether to overwrite existing output file
        top_k: Number of similar images to return
        is_phish_cli_arg: Boolean flag indicating if all query images should be marked as phishing

    Returns:
        Tuple of (list of valid image paths, optional list of target labels, list of true classes)
    """
    validate_images_dir(images_dir)
    validate_index_path(index_path, must_exist=True)
    validate_metadata_path(index_path.with_suffix(".csv"), must_exist=True)
    validate_output_path(output_path, overwrite)
    validate_threshold(threshold)
    validate_top_k(top_k)

    image_paths = get_image_paths(images_dir)
    num_images = len(image_paths)

    # Create true_classes_list based on is_phish_cli_arg
    true_classes_list = [1] * num_images if is_phish_cli_arg else [0] * num_images

    # Handle true_targets_list (labels)
    true_targets_list = [None] * num_images

    if labels_path:
        validate_labels_path(labels_path)
        loaded_labels = load_labels(labels_path)
        if len(loaded_labels) == num_images:
            true_targets_list = loaded_labels
        else:
            logger.warning(
                f"Found {len(loaded_labels)} labels but {num_images} images. True targets will not be fully populated for all query images."
            )
            true_targets_list = [p.parent.name.split("+")[0] for p in image_paths]
    else:
        # Default: derive from directory structure
        true_targets_list = [p.parent.name.split("+")[0] for p in image_paths]

    return image_paths, true_targets_list, true_classes_list


def main():
    parser = argparse.ArgumentParser(
        description="Query images against baseline perceptual hash index"
    )
    parser.add_argument(
        "--images", type=str, required=True, help="Path to query images directory"
    )
    parser.add_argument(
        "--index", type=str, required=True, help="Path to existing FAISS index"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--labels", type=str, help="Path to labels.txt file containing target labels"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold for classification",
    )
    parser.add_argument(
        "--top-k", type=int, default=1, help="Number of similar images to return"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for processing"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )
    parser.add_argument(
        "--is-phish",
        action="store_true",
        help="Mark all query images as phishing (true_class = 1)",
    )
    parser.add_argument("--log", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()
    if args.log:
        run = wandb.init(
            project="baseline-query",
            group="baseline",
            config=args,
        )

    # Convert paths
    images_dir = Path(args.images)
    index_path = Path(args.index)
    metadata_path = index_path.with_suffix(".csv")
    output_path = Path(args.output)
    labels_path = Path(args.labels) if args.labels else None

    # Validate inputs and get image paths and labels
    image_paths, labels, true_classes = validate_inputs(
        images_dir,
        index_path,
        output_path,
        labels_path,
        args.threshold,
        args.overwrite,
        args.top_k,
        args.is_phish,
    )

    # Initialize embedder with existing index
    embedder = BaselineEmbedder(index_path=index_path, metadata_path=metadata_path)

    try:
        # Process queries and get results
        results = embedder.search_similar(
            query_paths=image_paths,
            k=args.top_k,
            threshold=args.threshold,
            output_path=output_path if not args.overwrite else None,
            batch_size=args.batch_size,
            true_targets=labels,
            true_classes=true_classes,
        )

        if args.overwrite:
            results.to_csv(output_path, index=False)

        # Log CSV as wandb artifact if logging enabled
        if args.log:
            try:
                logger.info(f"Query results for {len(image_paths)} images")
                run.save(str(output_path))
            except Exception as e:
                logger.warning(f"Failed to log wandb artifact: {e}")

        logger.info(
            f"Successfully processed {len(image_paths)} queries and saved results to {output_path}"
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

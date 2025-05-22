import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import wandb

from BaselineEmbedder import BaselineEmbedder
from validators import (
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
) -> Tuple[List[Path], Optional[List[str]], List[int]]:
    """Validate input parameters and return list of valid image paths and labels.

    Args:
        images_dir: Directory containing query images
        index_path: Path to existing FAISS index
        output_path: Path to save results CSV
        labels_path: Optional path to labels.txt file
        threshold: Optional distance threshold for classification
        overwrite: Whether to overwrite existing output file
        top_k: Number of similar images to return

    Returns:
        Tuple of (list of valid image paths, optional list of labels, list of true_classes)
    """
    validate_images_dir(images_dir)
    validate_index_path(index_path, must_exist=True)
    validate_metadata_path(index_path.with_suffix(".csv"), must_exist=True)
    validate_output_path(output_path, overwrite)
    validate_threshold(threshold)
    validate_top_k(top_k)

    image_paths = get_image_paths(images_dir)

    # Handle labels if provided
    labels = None
    if labels_path:
        validate_labels_path(labels_path)
        labels = load_labels(labels_path)
        if len(labels) != len(image_paths):
            logger.error(
                f"Number of labels ({len(labels)}) does not match number of images ({len(image_paths)})"
            )
            sys.exit(1)

    # Initialize true_classes with placeholder value 7
    true_classes = [7] * len(image_paths)

    return image_paths, labels, true_classes


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

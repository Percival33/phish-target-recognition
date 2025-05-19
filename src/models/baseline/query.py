import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

from .baseline import BaselineEmbedder
from .validators import (
    validate_images_dir,
    validate_index_path,
    validate_output_path,
    validate_threshold,
    validate_metadata_path,
)
from .common import get_image_paths
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def validate_inputs(
    images_dir: Path,
    index_path: Path,
    output_path: Path,
    threshold: Optional[float] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Validate input parameters and return list of image paths.

    Args:
        images_dir: Directory containing query images
        index_path: Path to existing FAISS index
        output_path: Path to save results CSV
        threshold: Optional distance threshold for classification
        overwrite: Whether to overwrite existing output file

    Returns:
        List of valid image paths
    """
    validate_images_dir(images_dir)
    validate_index_path(index_path, must_exist=True)
    validate_metadata_path(index_path.with_suffix(".csv"), must_exist=True)
    validate_output_path(output_path, overwrite)
    validate_threshold(threshold)
    return get_image_paths(images_dir)


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

    args = parser.parse_args()

    # Convert paths
    images_dir = Path(args.images)
    index_path = Path(args.index)
    metadata_path = index_path.with_suffix(".csv")
    output_path = Path(args.output)

    # Validate inputs and get image paths
    image_paths = validate_inputs(
        images_dir, index_path, output_path, args.threshold, args.overwrite
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
        )

        if args.overwrite:
            results.to_csv(output_path, index=False)

        logger.info(
            f"Successfully processed {len(image_paths)} queries and saved results to {output_path}"
        )

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

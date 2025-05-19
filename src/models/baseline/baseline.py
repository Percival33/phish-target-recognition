import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import json

import pandas as pd
from PIL import Image
from .baseline import BaselineEmbedder
from tools.config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Baseline Perceptual Hash Service")
    parser.add_argument(
        "--images", type=str, required=True, help="Path to images directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument("--mapping", type=str, help="Path to target mappings JSON file")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file"
    )
    parser.add_argument(
        "--query", type=str, help="Path to query image for similarity search"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of similar images to return"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold for classifying as phishing. Applied only if --query is used. If distance < threshold, baseline_class for matched items in results becomes 1, else 0.",
    )
    args = parser.parse_args()

    # Input validation
    image_dir = Path(args.images)
    output_file = Path(args.output)

    if not image_dir.exists():
        logger.error(f"Input directory {image_dir} does not exist")
        sys.exit(1)

    image_paths: List[Path] = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpeg"))
    )
    if not image_paths:
        logger.error(f"No images found in input directory {image_dir}")
        sys.exit(1)

    if output_file.exists() and not args.overwrite:
        logger.error(
            f"Output file {output_file} already exists. Use --overwrite to replace"
        )
        sys.exit(1)

    # Load target mapping if provided
    target_mapping_data: Optional[Dict[str, str]] = None
    if args.mapping:
        mapping_path = Path(args.mapping)
        if mapping_path.exists():
            try:
                with open(mapping_path, "r") as f:
                    target_mapping_data = json.load(f)
                logger.info(f"Successfully loaded target mapping from {mapping_path}")
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding JSON from mapping file {mapping_path}: {e}"
                )
                # As per rule: log error, but don't halt if mapping is bad/missing
                # target_mapping_data remains None
            except Exception as e:
                logger.error(f"Error loading mapping file {mapping_path}: {e}")
                # target_mapping_data remains None
        else:
            logger.warning(
                f"Mapping file {mapping_path} not found. Proceeding without mappings."
            )
            # target_mapping_data remains None

    # Initialize embedder and compute hashes
    embedder = BaselineEmbedder(target_mapping=target_mapping_data)
    # The 'results' variable will now be self.image_metadata from the embedder instance
    results = embedder.compute_embeddings(image_paths, args.batch_size)

    # If query image is provided, perform similarity search
    if args.query:
        query_path = Path(args.query)
        if not query_path.exists():
            logger.error(f"Query image {query_path} does not exist")
            sys.exit(1)

        try:
            query_image = Image.open(query_path)
            similar_files, distances, targets = embedder.search_similar(
                query_image, args.top_k
            )

            # Keys for maps should be strings to match result_item['file']
            file_to_distance_map = {
                str(sf): dist for sf, dist in zip(similar_files, distances)
            }
            file_to_retrieved_target_map = {
                str(sf): target for sf, target in zip(similar_files, targets)
            }

            for result_item in results:
                file_key = result_item["file"]
                if file_key in file_to_distance_map:
                    distance = file_to_distance_map[file_key]
                    result_item["baseline_distance"] = distance
                    result_item["baseline_target"] = file_to_retrieved_target_map[
                        file_key
                    ]

                    # Apply threshold-based classification for baseline_class if threshold is provided
                    if args.threshold is not None:
                        if distance < args.threshold:
                            result_item["baseline_class"] = 1
                        else:
                            result_item["baseline_class"] = 0

        except Exception as e:
            logger.error(f"Error processing query image or updating results: {e}")
            sys.exit(1)

    df = pd.DataFrame(results)

    output_columns = [
        "file",
        "baseline_class",
        "baseline_distance",
        "baseline_target",
        "true_class",
        "true_target",
    ]
    df = df.reindex(
        columns=output_columns
    )  # Ensure column order and existence (fills with NaN if new)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

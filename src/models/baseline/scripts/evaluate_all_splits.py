import argparse
import logging
import shutil
import sys
from pathlib import Path

from dataset_config import DatasetConfig
from split_processor import SplitProcessor
from results_aggregator import ResultsAggregator

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def discover_splits(
    data_splits_dir: Path, dataset_name: str, split_structure: dict
) -> list:
    """Find and validate all splits for a specific dataset."""
    dataset_splits_dir = data_splits_dir / dataset_name
    if not dataset_splits_dir.exists():
        logger.warning(f"Dataset splits directory not found: {dataset_splits_dir}")
        return []

    # Find split directories
    split_dirs = [
        d
        for d in dataset_splits_dir.iterdir()
        if d.is_dir() and d.name.startswith("split_")
    ]

    if not split_dirs:
        logger.warning(f"No split directories found in {dataset_splits_dir}")
        return []

    # Validate structure
    valid_splits = []
    for split_dir in sorted(split_dirs):
        split_name = split_dir.name

        # Check required paths exist
        required_paths = [
            split_dir / split_structure["train_benign"],
            split_dir / split_structure["train_phish"],
            split_dir / split_structure["val"],
        ]

        if all(p.exists() for p in required_paths):
            valid_splits.append(split_name)
        else:
            logger.warning(
                f"Skipping {dataset_name}/{split_name} - missing required directories"
            )

    logger.info(
        f"Found {len(valid_splits)} valid splits for {dataset_name}: {valid_splits}"
    )
    return valid_splits


def get_split_paths(
    data_splits_dir: Path, dataset_name: str, split_name: str, split_structure: dict
) -> dict:
    """Get paths for a split."""
    split_dir = data_splits_dir / dataset_name / split_name
    return {
        "train_benign": split_dir / split_structure["train_benign"],
        "train_phish": split_dir / split_structure["train_phish"],
        "val": split_dir / split_structure["val"],
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model on all splits and datasets"
    )

    # Basic arguments only
    parser.add_argument(
        "--data-splits-dir", default="./data_splits", help="Data splits directory"
    )
    parser.add_argument("--output", default="results.json", help="Output JSON file")
    parser.add_argument(
        "--datasets",
        help="Specific datasets to process (comma-separated). If not provided, processes all datasets from config.",
    )
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k results")
    parser.add_argument("--threshold", type=float, help="Distance threshold")
    parser.add_argument("--splits", help="Specific splits to process (comma-separated)")
    parser.add_argument("--temp-dir", default="./temp", help="Temporary directory")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    try:
        # Load configuration
        config = DatasetConfig(args.config)
        available_datasets = config.get_available_datasets()

        if not available_datasets:
            raise ValueError("No datasets found in configuration")

        # Determine which datasets to process
        if args.datasets:
            requested_datasets = args.datasets.split(",")
            datasets_to_process = [
                d for d in available_datasets if d in requested_datasets
            ]
            if not datasets_to_process:
                raise ValueError(
                    f"None of the requested datasets found. Available: {available_datasets}"
                )
        else:
            datasets_to_process = available_datasets

        logger.info(f"Processing datasets: {datasets_to_process}")

        # Setup temp directory
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Process each dataset
        all_results = {}
        all_failed_splits = {}
        data_splits_dir = Path(args.data_splits_dir)

        for dataset_name in datasets_to_process:
            logger.info(f"Processing dataset: {dataset_name}")

            try:
                split_structure = config.get_split_structure(dataset_name)

                # Discover splits for this dataset
                all_splits = discover_splits(
                    data_splits_dir, dataset_name, split_structure
                )

                if not all_splits:
                    logger.warning(f"No valid splits found for dataset {dataset_name}")
                    continue

                # Filter splits if specified
                if args.splits:
                    requested = args.splits.split(",")
                    splits_to_process = [s for s in all_splits if s in requested]
                else:
                    splits_to_process = all_splits

                if not splits_to_process:
                    logger.warning(f"No splits to process for dataset {dataset_name}")
                    continue

                logger.info(
                    f"Processing {len(splits_to_process)} splits for {dataset_name}"
                )

                # Setup processor and aggregator for this dataset
                processor = SplitProcessor(
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                    threshold=args.threshold,
                )

                aggregator = ResultsAggregator(
                    dataset_name=dataset_name,
                    config={
                        "batch_size": args.batch_size,
                        "top_k": args.top_k,
                        "threshold": args.threshold,
                        "data_splits_dir": str(data_splits_dir),
                    },
                )

                # Process splits for this dataset
                dataset_temp_dir = temp_dir / dataset_name
                dataset_temp_dir.mkdir(parents=True, exist_ok=True)

                split_results = {}
                failed_splits = []

                for i, split_name in enumerate(splits_to_process):
                    logger.info(
                        f"Processing {dataset_name}/{split_name} ({i + 1}/{len(splits_to_process)})"
                    )

                    try:
                        split_paths = get_split_paths(
                            data_splits_dir, dataset_name, split_name, split_structure
                        )
                        split_temp_dir = dataset_temp_dir / split_name
                        split_temp_dir.mkdir(parents=True, exist_ok=True)

                        result = processor.process_split(split_paths, split_temp_dir)
                        split_results[split_name] = result

                        logger.info(f"Completed {dataset_name}/{split_name}")

                    except Exception as e:
                        logger.error(
                            f"Failed to process {dataset_name}/{split_name}: {e}"
                        )
                        failed_splits.append(split_name)

                # Store results for this dataset
                if split_results:
                    dataset_results = aggregator.aggregate_results(
                        split_results, failed_splits
                    )
                    all_results[dataset_name] = dataset_results
                    all_failed_splits[dataset_name] = failed_splits
                    logger.info(
                        f"Completed dataset {dataset_name}: {len(split_results)}/{len(splits_to_process)} splits successful"
                    )
                else:
                    logger.warning(f"No successful splits for dataset {dataset_name}")

            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {e}")
                all_failed_splits[dataset_name] = ["all_splits"]

        # Save combined results
        if not all_results:
            raise RuntimeError("No datasets processed successfully")

        # Create final combined results structure
        final_results = {
            "datasets": all_results,
            "failed_splits": all_failed_splits,
            "summary": {
                "total_datasets": len(datasets_to_process),
                "successful_datasets": len(all_results),
                "failed_datasets": len(datasets_to_process) - len(all_results),
                "config": {
                    "batch_size": args.batch_size,
                    "top_k": args.top_k,
                    "threshold": args.threshold,
                    "data_splits_dir": str(data_splits_dir),
                },
            },
        }

        # Save results
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            import json

            json.dump(final_results, f, indent=2)

        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(
            f"Processed {len(all_results)}/{len(datasets_to_process)} datasets successfully"
        )

        for dataset_name, dataset_results in all_results.items():
            if "summary" in dataset_results:
                summary = dataset_results["summary"]
                logger.info(
                    f"{dataset_name}: {summary.get('successful_splits', 0)} splits successful"
                )

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        if not args.keep_temp and Path(args.temp_dir).exists():
            try:
                shutil.rmtree(args.temp_dir)
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Failed to cleanup: {e}")


if __name__ == "__main__":
    main()

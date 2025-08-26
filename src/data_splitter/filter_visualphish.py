#!/usr/bin/env python3
"""
Simple script to filter VisualPhish format data based on sample thresholds.

Takes a folder with VisualPhish format data and creates symlinks to individual files
for targets that have at least the threshold number of samples in BOTH phish and benign folders.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Set, List


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def count_files_in_directory(directory: Path) -> int:
    """Count number of files (not directories) in a directory"""
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.is_file()])


def get_targets_in_directory(directory: Path) -> Set[str]:
    """Get set of target names (subdirectory names) in a directory"""
    if not directory.exists():
        return set()
    return {d.name for d in directory.iterdir() if d.is_dir()}


def create_symlinks_for_target(
    source_dir: Path, output_dir: Path, target: str, logger: logging.Logger
) -> int:
    """
    Create symlinks for all files in source_dir/target to output_dir/target
    Returns number of symlinks created
    """
    source_target_dir = source_dir / target
    output_target_dir = output_dir / target

    output_target_dir.mkdir(parents=True, exist_ok=True)

    symlinks_created = 0

    for source_file in source_target_dir.iterdir():
        if source_file.is_file():
            real_source = source_file.resolve()
            output_file = output_target_dir / source_file.name

            if not output_file.exists():
                try:
                    output_file.symlink_to(real_source)
                    symlinks_created += 1
                    logger.debug(f"Created symlink: {output_file} -> {real_source}")
                except OSError as e:
                    logger.error(f"Failed to create symlink {output_file}: {e}")
            else:
                logger.warning(f"Symlink already exists, skipping: {output_file}")

    return symlinks_created


def write_targets_file(
    output_dir: Path, filename: str, targets: List[str], logger: logging.Logger
):
    """Write targets to a file in the output directory"""
    targets_file = output_dir / filename

    with open(targets_file, "w") as f:
        for target in sorted(targets):
            f.write(f"{target}\n")

    logger.info(f"Created {filename} with {len(targets)} targets: {targets_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Filter VisualPhish format data based on sample thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/input --threshold 5 --output /path/to/output
  %(prog)s /path/to/input --phish-folder phishing --benign-folder trusted_list --threshold 10 --output /path/to/output
        """,
    )

    parser.add_argument(
        "input_folder",
        type=Path,
        help="Path to input folder containing VisualPhish format data",
    )

    parser.add_argument(
        "--phish-folder",
        type=str,
        default="phishing",
        help="Name of phishing subfolder under input folder (default: phishing)",
    )

    parser.add_argument(
        "--benign-folder",
        type=str,
        default="trusted_list",
        help="Name of benign subfolder under input folder (default: trusted_list)",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Minimum number of samples required in BOTH phish and benign folders",
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output folder"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate input folder
    if not args.input_folder.exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    if not args.input_folder.is_dir():
        logger.error(f"Input path is not a directory: {args.input_folder}")
        sys.exit(1)

    input_phish_dir = args.input_folder / args.phish_folder
    input_benign_dir = args.input_folder / args.benign_folder
    if not input_phish_dir.exists():
        logger.error(f"Phish folder does not exist: {input_phish_dir}")
        sys.exit(1)

    if not input_benign_dir.exists():
        logger.error(f"Benign folder does not exist: {input_benign_dir}")
        sys.exit(1)

    output_phish_dir = args.output / args.phish_folder
    output_benign_dir = args.output / args.benign_folder

    output_phish_dir.mkdir(parents=True, exist_ok=True)
    output_benign_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing VisualPhish data from: {args.input_folder}")
    logger.info(f"Phish folder: {args.phish_folder}")
    logger.info(f"Benign folder: {args.benign_folder}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Output folder: {args.output}")

    phish_targets = get_targets_in_directory(input_phish_dir)
    benign_targets = get_targets_in_directory(input_benign_dir)

    common_targets = phish_targets & benign_targets

    logger.info(f"Found {len(phish_targets)} targets in phish folder")
    logger.info(f"Found {len(benign_targets)} targets in benign folder")
    logger.info(f"Found {len(common_targets)} targets in both folders")

    valid_targets = []
    total_phish_samples = 0
    total_benign_samples = 0
    skipped_targets = []

    for target in sorted(common_targets):
        phish_count = count_files_in_directory(input_phish_dir / target)
        benign_count = count_files_in_directory(input_benign_dir / target)

        logger.debug(f"Target '{target}': phish={phish_count}, benign={benign_count}")

        if phish_count >= args.threshold and benign_count >= args.threshold:
            logger.info(
                f"Processing target '{target}' (phish={phish_count}, benign={benign_count})"
            )

            phish_symlinks = create_symlinks_for_target(
                input_phish_dir, output_phish_dir, target, logger
            )
            benign_symlinks = create_symlinks_for_target(
                input_benign_dir, output_benign_dir, target, logger
            )

            if phish_symlinks > 0 and benign_symlinks > 0:
                valid_targets.append(target)
                total_phish_samples += phish_symlinks
                total_benign_samples += benign_symlinks
                logger.info(
                    f"Created {phish_symlinks} phish + {benign_symlinks} benign symlinks for '{target}'"
                )
            else:
                logger.warning(f"No symlinks created for target '{target}', skipping")
                skipped_targets.append(target)
        else:
            logger.warning(
                f"Skipping target '{target}': phish={phish_count}, benign={benign_count} "
                f"(threshold={args.threshold})"
            )
            skipped_targets.append(target)

    if valid_targets:
        write_targets_file(output_benign_dir, "targets.txt", valid_targets, logger)
        write_targets_file(output_phish_dir, "targets2.txt", valid_targets, logger)

    logger.info("\n" + "=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Total targets found: {len(common_targets)}")
    logger.info(f"Valid targets (passed threshold): {len(valid_targets)}")
    logger.info(f"Skipped targets: {len(skipped_targets)}")
    logger.info(f"Total phish samples: {total_phish_samples}")
    logger.info(f"Total benign samples: {total_benign_samples}")
    logger.info(f"Total samples: {total_phish_samples + total_benign_samples}")

    if skipped_targets:
        logger.info(f"Skipped targets: {', '.join(skipped_targets)}")

    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()

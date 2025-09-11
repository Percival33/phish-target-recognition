#!/usr/bin/env python3
"""
Generate whitelist_file_names.npy based on train_idx.npy and the data folder structure.
This script reads the file names from the data folder and uses train_idx to select
which files should be included in the whitelist.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tools.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, setup_logging


def read_all_file_names(data_path, targets_file="targets.txt"):
    """
    Read all file names from the data directory structure.
    Returns file names in the format: "target_dir/filename"
    """
    targets_path = data_path / targets_file

    if not targets_path.exists():
        raise FileNotFoundError(f"targets.txt not found at {targets_path}")

    with open(targets_path, "r") as f:
        targets = f.read().strip().splitlines()

    all_file_names = []

    targets_list = sorted(targets)

    for target_dir in targets_list:
        target_path = data_path / target_dir

        if not target_path.exists():
            logging.warning(f"Target directory {target_path} does not exist, skipping...")
            continue

        files = sorted([f for f in target_path.iterdir() if f.is_file()])

        for file_path in files:
            file_name = f"{target_dir}/{file_path.name}"
            all_file_names.append(file_name)

    return np.array(all_file_names)


def generate_whitelist_filenames(trusted_list_path):
    """Generate file names for ALL trusted images (the whitelist)"""

    with open(trusted_list_path / "targets.txt", "r") as f:
        targets = f.read().strip().splitlines()

    all_file_names = []

    targets_list = sorted(targets)

    for target_dir in targets_list:
        target_path = trusted_list_path / target_dir

        files = sorted(target_path.iterdir())

        for file_path in files:
            if file_path.is_file():
                file_name = f"{file_path.parent}/{file_path.name}"
                all_file_names.append(file_name)

    return np.array(all_file_names)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(description="Generate whitelist_file_names.npy from train indices and data folder")

    parser.add_argument(
        "--data-folder",
        type=Path,
        default=RAW_DATA_DIR / "VisualPhish",
        help="Path to the data folder containing trusted_list directory",
    )

    parser.add_argument(
        "--train-idx",
        type=Path,
        default=INTERIM_DATA_DIR / "VisualPhish" / "train_idx.npy",
        help="Path to train_idx.npy file",
    )

    parser.add_argument(
        "--test-idx",
        type=Path,
        default=INTERIM_DATA_DIR / "VisualPhish" / "test_idx.npy",
        help="Path to test_idx.npy file (optional, for verification)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "VisualPhish" / "whitelist_file_names.npy",
        help="Path to save the whitelist_file_names.npy",
    )

    parser.add_argument(
        "--use-phishing", action="store_true", help="If set, also include phishing data and use train_idx to split it"
    )

    args = parser.parse_args()

    logger.info(f"Data folder: {args.data_folder}")
    logger.info(f"Train index file: {args.train_idx}")
    logger.info(f"Output file: {args.output}")

    if args.use_phishing:
        logger.info("Using phishing data mode - reading from phishing folder")

        phishing_path = args.data_folder / "phishing"
        if not phishing_path.exists():
            raise FileNotFoundError(f"phishing directory not found at {phishing_path}")

        phishing_file_names = read_all_file_names(phishing_path)
        logger.info(f"Found {len(phishing_file_names)} phishing files")

        train_idx = np.load(args.train_idx)

        if max(train_idx) < len(phishing_file_names):
            whitelist_file_names = phishing_file_names[train_idx]
        else:
            logger.error(f"Train indices ({max(train_idx)}) exceed phishing files ({len(phishing_file_names)})")
            raise ValueError("Invalid train indices for phishing data")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, whitelist_file_names)
        logger.info(f"Saved {len(whitelist_file_names)} whitelist file names from phishing data to {args.output}")
    else:
        whitelist_file_names = generate_whitelist_filenames(args.data_folder)

    if args.test_idx.exists():
        test_idx = np.load(args.test_idx)
        logger.info(f"Test indices: {len(test_idx)} items")

        if len(set(train_idx) & set(test_idx)) > 0:
            logger.warning("Warning: Train and test indices have overlapping values!")

    logger.info("Whitelist file names generation completed successfully")

    logger.info("\nSample of generated whitelist file names:")
    for i, fname in enumerate(whitelist_file_names[:5]):
        logger.info(f"  {i}: {fname}")
    if len(whitelist_file_names) > 5:
        logger.info(f"  ... ({len(whitelist_file_names) - 5} more)")


if __name__ == "__main__":
    main()

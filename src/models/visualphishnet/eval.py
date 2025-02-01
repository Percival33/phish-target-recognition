import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tools.config import PROCESSED_DATA_DIR, setup_logging

from DataHelper import TrainResults, get_phish_file_names
from Evaluate import Evaluate
from TargetHelper import TargetHelper

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    """
    file_handler = logging.FileHandler('result.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    """

    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=PROCESSED_DATA_DIR / "smallerSampleDataset")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR / "VP-original")
    parser.add_argument("--version", type=str, default="2")
    args = parser.parse_args()
    logger.info("Evaluating VisualPhishNet")
    logger.info(f"phishing_emb{args.version}.npy")
    # TODO: enable using wandb artifacts
    VPTrainResults = TrainResults(
        X_legit_train=np.load(args.output_dir / f"whitelist_emb{args.version}.npy"),
        y_legit_train=np.load(args.output_dir / f"whitelist_labels{args.version}.npy"),
        X_phish=np.load(args.output_dir / f"phishing_emb{args.version}.npy"),
        y_phish=np.load(args.output_dir / f"phishing_labels{args.version}.npy"),
        phish_test_idx=np.load(args.output_dir / "test_idx.npy"),
        phish_train_idx=np.load(args.output_dir / "train_idx.npy"),
    )

    targetHelper = TargetHelper(args.dataset_path / "phishing")

    legit_file_names = targetHelper.read_file_names(args.dataset_path / "trusted_list", "targets.txt")
    phish_file_names = targetHelper.read_file_names(args.dataset_path / "phishing", "targets.txt")

    phish_train_files, phish_test_files = get_phish_file_names(
        phish_file_names,
        VPTrainResults.phish_train_idx,
        VPTrainResults.phish_test_idx,
    )

    evaluate = Evaluate(VPTrainResults, legit_file_names, phish_train_files)

    n = 1  # Top-1 match
    correct = 0

    for i, test_file in enumerate(phish_test_files):
        filename = str(test_file.name) if isinstance(test_file, Path) else test_file
        print(f"FILENAME: {type(filename)} {filename} <> {test_file}")
        distances_to_train = evaluate.pairwise_distance[i, :]
        names_min_distance, only_names, min_distances = evaluate.find_names_min_distances(
            *evaluate.find_min_distances(np.ravel(distances_to_train), n)
        )
        found, found_idx = targetHelper.check_if_target_in_top(filename, only_names)
        logger.info(names_min_distance)

        if found == 1:
            correct += 1

    accuracy = correct / len(phish_test_files)
    logger.info(f"Correct match percentage: {accuracy:.2%}")

import logging
from argparse import ArgumentParser

import numpy as np
from tools.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, setup_logging
from pathlib import Path
from Evaluate import Evaluate
from TargetHelper import TargetHelper

from DataHelper import TrainResults, get_phish_file_names


def get_label_from_name(name):
    first_half = name.split("_", 1)[0]
    number = int(first_half.replace("T", ""))
    return number


# class TargetHelper:
#     # parents_targets = ['microsoft','apple','google','alibaba']
#     # sub_targets = [['ms_outlook','ms_office','ms_bing','ms_onedrive','ms_skype'],['itunes','icloud'],['google_drive'],['aliexpress']]
#
#     parents_targets_idx = [90, 12, 65, 4]
#     sub_targets = [[150, 152, 151, 149, 148], [153, 154], [147], [5]]
#
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#
#     # this function maps sub targets if they were split
#     def check_if_same_category(self, img_label1, img_label2):
#         if_same = 0
#         if img_label1 in self.parents_targets_idx:
#             if img_label2 in self.sub_targets[self.parents_targets_idx.index(img_label1)]:
#                 if_same = 1
#         elif img_label1 in self.sub_targets[0]:
#             if img_label2 in self.sub_targets[0] or img_label2 == self.parents_targets_idx[0]:
#                 if_same = 1
#         elif img_label1 in self.sub_targets[1]:
#             if img_label2 in self.sub_targets[1] or img_label2 == self.parents_targets_idx[1]:
#                 if_same = 1
#         elif img_label1 in self.sub_targets[2]:
#             if img_label2 in self.sub_targets[2] or img_label2 == self.parents_targets_idx[2]:
#                 if_same = 1
#         return if_same
#
#     # Find if target is in the top closest n distances
#     def check_if_target_in_top(self, test_file_name, only_names):
#         found = 0
#         idx = 0
#         test_label = get_label_from_name(test_file_name)
#         self.logger.info('***')
#         self.logger.info('Test example: %s', test_file_name)
#         for i in range(0, len(only_names)):
#             label_distance = get_label_from_name(only_names[i])
#             if label_distance == test_label or self.check_if_same_category(test_label, label_distance) == 1:
#                 found = 1
#                 idx = i + 1
#                 self.logger.info('found')
#                 break
#         return found, idx
#
#     # Get file names of each example
#     @staticmethod
#     def read_file_names(data_path, file_name):
#         targets_file = open(data_path / file_name, "r")
#         targets = targets_file.read()
#
#         file_names_list = []
#         targets_list = targets.splitlines()
#         for i in range(0, len(targets_list)):
#             target_path = data_path / targets_list[i]
#             file_names = sorted(os.listdir(target_path))
#             for j in range(0, len(file_names)):
#                 file_names_list.append(file_names[j])
#         return file_names_list
#
#     @staticmethod
#     def get_label_from_name(name):
#         first_half = name.split('_', 1)[0]
#         number = int(first_half.replace('T', ''))
#         return number


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger()

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
    args = parser.parse_args()
    logger.info("Evaluating VisualPhishNet")

    # TODO: enable using wandb artifacts
    VPTrainResults = TrainResults(
        X_legit_train=np.load(args.output_dir / "whitelist_emb3.npy"),
        y_legit_train=np.load(args.output_dir / "whitelist_labels3.npy"),
        X_phish=np.load(args.output_dir / "phishing_emb3.npy"),
        y_phish=np.load(args.output_dir / "phishing_labels3.npy"),
        phish_test_idx=np.load(args.output_dir / "test_idx.npy"),
        phish_train_idx=np.load(args.output_dir / "train_idx.npy"),
    )

    targetHelper = TargetHelper(args.dataset_path / "phishing")

    legit_file_names_targets = targetHelper.read_file_names(args.dataset_path / "trusted_list", "targets.txt")
    phish_file_names_targets = targetHelper.read_file_names(args.dataset_path / "phishing", "targets.txt")
    phish_train_file_names, phish_test_file_names = get_phish_file_names(
        phish_file_names_targets,
        VPTrainResults.phish_train_idx,
        VPTrainResults.phish_test_idx,
    )

    evaluate = Evaluate(VPTrainResults, legit_file_names_targets, phish_train_file_names)

    n = 1  # Top-1 match
    correct = 0

    for i in range(0, VPTrainResults.phish_test_idx.shape[0]):
        distances_to_train = evaluate.pairwise_distance[i, :]
        idx, values = evaluate.find_min_distances(np.ravel(distances_to_train), n)
        names_min_distance, only_names, min_distances = evaluate.find_names_min_distances(idx, values)
        found, found_idx = targetHelper.check_if_target_in_top(phish_test_file_names[i], only_names)
        print(names_min_distance)

        if found == 1:
            correct += 1
    logger.info("Correct match percentage: " + str(correct / VPTrainResults.phish_test_idx.shape[0]))

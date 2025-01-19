import logging
import os
from dataclasses import dataclass
import numpy as np
from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


# TODO: rename as it is not clear what it does (contains embeddings and labels only)
@dataclass
class DataSet:
    phish_test_idx: np.ndarray
    phish_train_idx: np.ndarray

    X_legit_train: np.ndarray
    y_legit_train: np.ndarray
    X_phish: np.ndarray
    y_phish: np.ndarray

    def __post_init__(self):
        self.X_phish_test = self.X_phish[self.phish_test_idx, :]
        self.y_phish_test = self.y_phish[self.phish_test_idx, :]

        self.X_phish_train = self.X_phish[self.phish_train_idx, :]
        self.y_phish_train = self.y_phish[self.phish_train_idx, :]


def get_label_from_name(name):
    first_half = name.split('_', 1)[0]
    number = int(first_half.replace('T', ''))
    return number


class Evaluate:
    # Find same-category website (matching is correct if it was matched to the same category (e.g. microsoft and outlook ))

    def __init__(self, dataset, legit_file_names, phish_train_file_names):
        self.X_phish_train = dataset.X_phish_train
        self.X_phish_test = dataset.X_phish_test
        self.phish_train_idx = dataset.phish_train_idx
        self.X_legit_train = dataset.X_legit_train

        self.legit_file_names = legit_file_names
        self.phish_train_file_names = phish_train_file_names

        self.pairwise_distance = self.compute_all_distances(self.X_phish_test)

    # L2 distance
    def compute_distance_pair(self, layer1, layer2):
        diff = layer1 - layer2
        l2_diff = np.sum(diff ** 2) / self.X_phish_train.shape[1]
        return l2_diff

    # Pairwise distance between query image and training
    def compute_all_distances(self, test_matrix):
        # TODO: refactor with trainer_phase2.py
        train_size = self.phish_train_idx.shape[0] + self.X_legit_train.shape[0]
        X_all_train = np.concatenate((self.X_phish_train, self.X_legit_train))
        pairwise_distance = np.zeros([test_matrix.shape[0], train_size])
        for i in range(0, test_matrix.shape[0]):
            pair1 = test_matrix[i, :]
            for j in range(0, train_size):
                pair2 = X_all_train[j, :]
                l2_diff = self.compute_distance_pair(pair1, pair2)
                pairwise_distance[i, j] = l2_diff
        return pairwise_distance

    # Find Smallest n distances
    @staticmethod
    def find_min_distances(distances, n):
        idx = distances.argsort()[:n]
        values = distances[idx]
        return idx, values

    # Find names of examples with min distance
    def find_names_min_distances(self, idx, values):
        names_min_distance = ''
        only_names = []
        distances = ''
        for i in range(0, idx.shape[0]):
            index_min_distance = idx[i]
            if index_min_distance < self.X_phish_train.shape[0]:
                names_min_distance = names_min_distance + 'Phish: ' + self.phish_train_file_names[
                    index_min_distance] + ','
                only_names.append(self.phish_train_file_names[index_min_distance])
            else:
                names_min_distance = names_min_distance + 'Legit: ' + self.legit_file_names[
                    index_min_distance - self.X_phish_train.shape[0]] + ','
                only_names.append(self.legit_file_names[index_min_distance - self.X_phish_train.shape[0]])
            distances = distances + str(values[i]) + ','
        names_min_distance = names_min_distance[:-1]
        distances = distances[:-1]
        return names_min_distance, only_names, distances


class TargetHelper:
    # parents_targets = ['microsoft','apple','google','alibaba']
    # sub_targets = [['ms_outlook','ms_office','ms_bing','ms_onedrive','ms_skype'],['itunes','icloud'],['google_drive'],['aliexpress']]

    parents_targets_idx = [90, 12, 65, 4]
    sub_targets = [[150, 152, 151, 149, 148], [153, 154], [147], [5]]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # this function maps sub targets if they were split
    def check_if_same_category(self, img_label1, img_label2):
        if_same = 0
        if img_label1 in self.parents_targets_idx:
            if img_label2 in self.sub_targets[self.parents_targets_idx.index(img_label1)]:
                if_same = 1
        elif img_label1 in self.sub_targets[0]:
            if img_label2 in self.sub_targets[0] or img_label2 == self.parents_targets_idx[0]:
                if_same = 1
        elif img_label1 in self.sub_targets[1]:
            if img_label2 in self.sub_targets[1] or img_label2 == self.parents_targets_idx[1]:
                if_same = 1
        elif img_label1 in self.sub_targets[2]:
            if img_label2 in self.sub_targets[2] or img_label2 == self.parents_targets_idx[2]:
                if_same = 1
        return if_same

    # Find if target is in the top closest n distances
    def check_if_target_in_top(self, test_file_name, only_names):
        found = 0
        idx = 0
        test_label = get_label_from_name(test_file_name)
        self.logger.info('***')
        self.logger.info('Test example: %s', test_file_name)
        for i in range(0, len(only_names)):
            label_distance = get_label_from_name(only_names[i])
            if label_distance == test_label or self.check_if_same_category(test_label, label_distance) == 1:
                found = 1
                idx = i + 1
                self.logger.info('found')
                break
        return found, idx

    # Get file names of each example
    @staticmethod
    def read_file_names(data_path, file_name):
        targets_file = open(data_path / file_name, "r")
        targets = targets_file.read()

        file_names_list = []
        targets_list = targets.splitlines()
        for i in range(0, len(targets_list)):
            target_path = data_path / targets_list[i]
            file_names = sorted(os.listdir(target_path))
            for j in range(0, len(file_names)):
                file_names_list.append(file_names[j])
        return file_names_list

    @staticmethod
    def get_label_from_name(name):
        first_half = name.split('_', 1)[0]
        number = int(first_half.replace('T', ''))
        return number


def get_phish_file_names(phish_file_names, phish_train_idx, phish_test_idx):
    phish_train_file_names = []
    for i in range(0, phish_train_idx.shape[0]):
        phish_train_file_names.append(phish_file_names[phish_train_idx[i]])

    phish_train_file_names = [phish_file_names[idx] for idx in phish_train_idx]

    phish_test_file_names = []
    for i in range(0, phish_test_idx.shape[0]):
        phish_test_file_names.append(phish_file_names[phish_test_idx[i]])

    return phish_train_file_names, phish_test_file_names


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger()
    logger.info("Evaluating VisualPhishNet")
    dataset_path = PROCESSED_DATA_DIR / 'smallerSampleDataset'
    output_dir_path = INTERIM_DATA_DIR / 'VisualPhish'

    # TODO: enable using wandb artifacts
    VPDatasSet = DataSet(
        X_legit_train=np.load(output_dir_path / 'whitelist_emb.npy'),
        y_legit_train=np.load(output_dir_path / 'whitelist_labels.npy'),
        X_phish=np.load(output_dir_path / 'phishing_emb.npy'),
        y_phish=np.load(output_dir_path / 'phishing_labels.npy'),
        phish_test_idx=np.load(output_dir_path / 'test_idx.npy'),
        phish_train_idx=np.load(output_dir_path / 'train_idx.npy'),
    )

    targetHelper = TargetHelper()

    legit_file_names_targets = targetHelper.read_file_names(dataset_path / 'trusted_list', 'targets.txt')
    phish_file_names_targets = targetHelper.read_file_names(dataset_path / 'phishing', 'targets.txt')
    phish_train_file_names, phish_test_file_names = get_phish_file_names(phish_file_names_targets,
                                                                         VPDatasSet.phish_train_idx,
                                                                         VPDatasSet.phish_test_idx)

    evaluate = Evaluate(VPDatasSet, legit_file_names_targets, phish_train_file_names)

    n = 1  # Top-1 match
    correct = 0

    for i in range(0, VPDatasSet.phish_test_idx.shape[0]):
        distances_to_train = evaluate.pairwise_distance[i, :]
        idx, values = evaluate.find_min_distances(np.ravel(distances_to_train), n)
        names_min_distance, only_names, min_distances = evaluate.find_names_min_distances(idx, values)
        found, found_idx = targetHelper.check_if_target_in_top(phish_test_file_names[i], only_names)
        print(names_min_distance)

        if found == 1:
            correct += 1
    logger.info("Correct match percentage: " + str(correct / VPDatasSet.phish_test_idx.shape[0]))

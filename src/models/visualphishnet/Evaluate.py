from __future__ import annotations

import numpy as np

import DataHelper as data


class Evaluate:
    # Find same-category website (matching is correct if it was matched to the same category (e.g. microsoft and outlook ))

    def __init__(self, train_results: data.TrainResults | None, legit_file_names, phish_train_file_names):
        self.X_phish_train = train_results.X_phish_train
        self.X_phish_test = train_results.X_phish_test
        self.phish_train_idx = train_results.phish_train_idx
        self.X_legit_train = train_results.X_legit_train

        self.legit_file_names = legit_file_names
        self.phish_train_file_names = phish_train_file_names

        if train_results is not None:
            self.pairwise_distance = self.compute_all_distances(self.X_phish_test)

    def compute_all_distances(self, test_matrix, train_legit=None, train_phish=None):
        """
        Compute pairwise distances between test matrix and training data.

        Args:
        test_matrix (numpy.ndarray): Matrix of test samples
        train_legit (numpy.ndarray, optional): Legitimate training samples
        train_phish (numpy.ndarray, optional): Phishing training samples

        Returns:
        numpy.ndarray: Pairwise distances between test matrix and training data
        """
        # If no explicit train sets are provided, use class attributes
        if train_legit is None:
            train_legit = self.X_legit_train
        if train_phish is None:
            train_phish = self.X_phish_train

        # Concatenate training samples
        train_size = train_legit.shape[0] + train_phish.shape[0]
        X_all_train = np.concatenate((train_legit, train_phish))

        # Initialize pairwise distance matrix
        pairwise_distance = np.zeros([test_matrix.shape[0], train_size])

        # Compute distances
        for i in range(test_matrix.shape[0]):
            pair1 = test_matrix[i, :]
            for j in range(train_size):
                pair2 = X_all_train[j, :]
                l2_diff = self.compute_distance_pair(pair1, pair2)
                pairwise_distance[i, j] = l2_diff

        return pairwise_distance

    # L2 distance
    def compute_distance_pair(self, layer1, layer2):
        diff = layer1 - layer2
        l2_diff = np.sum(diff**2) / self.X_phish_train.shape[1]
        return l2_diff

    # Find names of examples with min distance
    def find_names_min_distances(self, idx, values):
        names_min_distance = ""
        only_names = []
        distances = ""
        for i in range(0, idx.shape[0]):
            index_min_distance = idx[i]
            if index_min_distance < self.X_phish_train.shape[0]:
                filename = self.phish_train_file_names[index_min_distance]
                names_min_distance += f"Phish: {filename},"
                only_names.append(filename)
            else:
                adjusted_index = index_min_distance - self.X_phish_train.shape[0]
                filename = self.legit_file_names[adjusted_index]
                names_min_distance += f"Legit: {filename},"
                only_names.append(filename)

            distances = distances + str(values[i]) + ","

        names_min_distance.rstrip(",")
        distances.rstrip(",")

        return names_min_distance, only_names, distances

    # Find Smallest n distances
    @staticmethod
    def find_min_distances(distances, n):
        idx = distances.argsort()[:n]
        values = distances[idx]
        return idx, values

from __future__ import annotations

import numpy as np
from tqdm import tqdm

import DataHelper as data


class Evaluate:
    # Find same-category website (matching is correct if it was matched to the same category (e.g. microsoft and outlook ))

    def __init__(self, train_results: data.TrainResults, legit_file_names, phish_train_file_names):
        self.X_phish_train = train_results.X_phish_train
        self.X_phish_test = train_results.X_phish_test
        self.phish_train_idx = train_results.phish_train_idx
        self.X_legit_train = train_results.X_legit_train

        self.legit_file_names = legit_file_names
        self.phish_train_file_names = phish_train_file_names

        self.pairwise_distance = self._compute_all_distances(self.X_phish_test)

    @staticmethod
    def compute_all_distances(test_matrix, train_legit, train_phish):
        """
        Compute pairwise distances between test matrix and training data.

        Args:
        test_matrix (numpy.ndarray): Matrix of test samples
        train_legit (numpy.ndarray): Legitimate training samples
        train_phish (numpy.ndarray): Phishing training samples

        Returns:
        numpy.ndarray: Pairwise distances between test matrix and training data
        """
        X_all_train = np.concatenate((train_phish, train_legit))

        distances = (
            np.sum((test_matrix[:, np.newaxis, :] - X_all_train[np.newaxis, :, :]) ** 2, axis=2) / train_phish.shape[1]
        )

        return distances

    def _compute_all_distances(self, test_matrix):
        return self.compute_all_distances(test_matrix, self.X_legit_train, self.X_phish_train)

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

    @staticmethod
    def compute_all_distances_batched(test_matrix, targetlist_emb, batch_size=256):
        """
        Computes pairwise L2 distances between test_matrix and targetlist_emb in a batch-wise manner.
        """
        train_size = targetlist_emb.shape[0]
        test_size = test_matrix.shape[0]
        pairwise_distance = np.zeros((test_size, train_size))

        for batch_start in tqdm(range(0, test_size, batch_size), desc="Computing pairwise distances (batched)"):
            batch_end = min(batch_start + batch_size, test_size)
            batch = test_matrix[batch_start:batch_end]

            # Compute pairwise L2 distances efficiently
            diff = batch[:, np.newaxis, :] - targetlist_emb[np.newaxis, :, :]
            l2_diff = np.sum(diff**2, axis=2) / targetlist_emb.shape[1]

            pairwise_distance[batch_start:batch_end] = l2_diff

        return pairwise_distance

    # Find Smallest n distances
    @staticmethod
    def find_min_distances(distances, n):
        idx = distances.argsort()[:n]
        values = distances[idx]
        return idx, values

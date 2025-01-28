import numpy as np

import DataHelper as data

class Evaluate:
    # Find same-category website (matching is correct if it was matched to the same category (e.g. microsoft and outlook ))

    def __init__(self, trainResults: data.TrainResults, legit_file_names, phish_train_file_names):
        self.X_phish_train = trainResults.X_phish_train
        self.X_phish_test = trainResults.X_phish_test
        self.phish_train_idx = trainResults.phish_train_idx
        self.X_legit_train = trainResults.X_legit_train

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
                names_min_distance = names_min_distance + 'Phish: ' + str(self.phish_train_file_names[
                    index_min_distance]) + ','
                only_names.append(str(self.phish_train_file_names[index_min_distance].name))
            else:
                names_min_distance = names_min_distance + 'Legit: ' + str(self.legit_file_names[
                    index_min_distance - self.X_phish_train.shape[0]]) + ','
                only_names.append(str(self.legit_file_names[index_min_distance - self.X_phish_train.shape[0]].name))
            distances = distances + str(values[i]) + ','
        names_min_distance = names_min_distance[:-1]
        distances = distances[:-1]
        return names_min_distance, only_names, distances

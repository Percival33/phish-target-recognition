import numpy as np


class RandomSampling:
    def __init__(
        self,
        targetHelper,
        labels_start_end_train_phish,
        labels_start_end_test_phish,
        labels_start_end_train_legit,
    ):
        self.targetHelper = targetHelper
        self.labels_start_end_train_phish = labels_start_end_train_phish
        self.labels_start_end_test_phish = labels_start_end_test_phish
        self.labels_start_end_train_legit = labels_start_end_train_legit

    # sample triplets

    def get_batch(
        self,
        targetHelper,
        X_train_legit,
        y_train_legit,
        X_train_phish,
        labels_start_end_train_legit,
        batch_size,
        num_targets,
    ):
        # initialize 3 empty arrays for the input image batch
        h = X_train_legit.shape[1]
        w = X_train_legit.shape[2]
        triple = [np.zeros((batch_size, h, w, 3)) for i in range(3)]

        # TODO: simplify this by creating a dict with targets
        # https://github.com/lindsey98/PhishingBaseline/blob/main/VisualPhishnet/visualphish_model.py#L43
        for i in range(batch_size):
            img_idx_pair1 = self.pick_first_img_idx(labels_start_end_train_legit, num_targets)
            triple[0][i, :, :, :] = X_train_legit[img_idx_pair1, :]
            img_label = int(y_train_legit[img_idx_pair1])

            # get image for the second: positive
            triple[1][i, :, :, :] = self.pick_pos_img_idx(X_train_legit, X_train_phish, 0.15, img_label)

            # get image for the third: negative from legit
            # don't sample from the same cluster
            img_neg, label_neg = self.pick_neg_img(X_train_legit, img_label, num_targets)
            while targetHelper.check_if_same_category(img_label, label_neg) == 1:
                img_neg, label_neg = self.pick_neg_img(X_train_legit, img_label, num_targets)

            triple[2][i, :, :, :] = img_neg

        return triple

    @staticmethod
    # Sample anchor, positive and negative images
    def pick_first_img_idx(labels_start_end, num_targets):
        # TODO: same as in triplet_sampling.py
        random_target = -1
        while random_target == -1:
            random_target = np.random.randint(low=0, high=num_targets)
            if labels_start_end[random_target, 0] == -1:
                random_target = -1
        return random_target

    def pick_pos_img_idx(self, X_train_legit, X_train_phish, prob_phish, img_label):
        if np.random.uniform() > prob_phish:
            class_idx_start_end = self.labels_start_end_train_legit[img_label, :]
            same_idx = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
            img = X_train_legit[same_idx, :]
        else:
            if not self.labels_start_end_train_phish[img_label, 0] == -1:
                class_idx_start_end = self.labels_start_end_train_phish[img_label, :]
                same_idx = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
                img = X_train_phish[same_idx, :]
            else:
                class_idx_start_end = self.labels_start_end_train_legit[img_label, :]
                same_idx = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
                img = X_train_legit[same_idx, :]
        return img

    def pick_neg_img(self, X_train_legit, anchor_idx, num_targets):
        if anchor_idx == 0:
            targets = np.arange(1, num_targets)
        elif anchor_idx == num_targets - 1:
            targets = np.arange(0, num_targets - 1)
        else:
            targets = np.concatenate([np.arange(0, anchor_idx), np.arange(anchor_idx + 1, num_targets)])
        diff_target_idx = np.random.randint(low=0, high=num_targets - 1)
        diff_target = targets[diff_target_idx]

        class_idx_start_end = self.labels_start_end_train_legit[diff_target, :]
        idx_from_diff_target = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
        img = X_train_legit[idx_from_diff_target, :]

        return img, diff_target

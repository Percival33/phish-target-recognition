import numpy as np


# Don't sample negative image from the same category as the positive image (e.g. google and google drive)
# Create clusters of same-company websites (e.g. all microsoft websites)
class TargetHelper:
    # targets names of parent and sub websites
    target_lists = [['microsoft', 'ms_outlook', 'ms_office', 'ms_bing', 'ms_onedrive', 'ms_skype'],
                    ['apple', 'itunes', 'icloud'], ['google', 'google_drive'], ['alibaba', 'aliexpress']]

    def __init__(self, data_path):
        targets_file = open(data_path / 'targets.txt', "r")
        self.all_targets = targets_file.read().splitlines()
        self.parents_ids, self.sub_target_lists_idx = self._get_associated_targets_idx(self.target_lists,
                                                                                       self.all_targets)

    def check_if_same_category(self, img_label1, img_label2):
        if_same = 0
        if img_label1 in self.parents_ids:
            if img_label2 in self.sub_target_lists_idx[self.parents_ids.index(img_label1)]:
                if_same = 1
        elif img_label1 in self.sub_target_lists_idx[0]:
            if img_label2 in self.sub_target_lists_idx[0] or img_label2 == self.parents_ids[0]:
                if_same = 1
        elif img_label1 in self.sub_target_lists_idx[1]:
            if img_label2 in self.sub_target_lists_idx[1] or img_label2 == self.parents_ids[1]:
                if_same = 1
        elif img_label1 in self.sub_target_lists_idx[2]:
            if img_label2 in self.sub_target_lists_idx[2] or img_label2 == self.parents_ids[2]:
                if_same = 1
        return if_same

    def _get_associated_targets_idx(self, target_lists, all_targets):
        sub_target_lists_idx = []
        parents_ids = []
        for i in range(0, len(target_lists)):
            target_list = target_lists[i]
            parent_target = target_list[0]
            one_target_list = []
            parent_idx = self.get_idx_of_target(parent_target, all_targets)
            parents_ids.append(parent_idx)
            for child_target in target_list[1:]:
                child_idx = self.get_idx_of_target(child_target, all_targets)
                one_target_list.append(child_idx)
            sub_target_lists_idx.append(one_target_list)
        return parents_ids, sub_target_lists_idx

    @staticmethod
    def get_idx_of_target(target_name, all_targets):
        for i in range(0, len(all_targets)):
            if all_targets[i] == target_name:
                found_idx = i
                return found_idx


def pick_first_img_idx(labels_start_end, num_targets):
    random_target = -1
    while (random_target == -1):
        random_target = np.random.randint(low=0, high=num_targets)
        if labels_start_end[random_target, 0] == -1:
            random_target = -1
    return random_target


def pick_pos_img_idx(labels_start_end_train, X_train_new, img_label):
    class_idx_start_end = labels_start_end_train[img_label, :]
    same_idx = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
    img = X_train_new[same_idx, :]
    return img


def pick_neg_img(labels_start_end_train, X_train_new, anchor_idx, num_targets):
    if anchor_idx == 0:
        targets = np.arange(1, num_targets)
    elif anchor_idx == num_targets - 1:
        targets = np.arange(0, num_targets - 1)
    else:
        targets = np.concatenate([np.arange(0, anchor_idx), np.arange(anchor_idx + 1, num_targets)])
    diff_target_idx = np.random.randint(low=0, high=num_targets - 1)
    diff_target = targets[diff_target_idx]

    class_idx_start_end = labels_start_end_train[diff_target, :]
    idx_from_diff_target = np.random.randint(low=class_idx_start_end[0], high=class_idx_start_end[1] + 1)
    img = X_train_new[idx_from_diff_target, :]

    return img, diff_target


# Sample batch
def get_batch(targetHelper, X_train_legit, X_train_new, labels_start_end_train, batch_size, train_fixed_set,
              num_targets):
    # initialize 3 empty arrays for the input image batch
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple = [np.zeros((batch_size, h, w, 3)) for i in range(3)]

    for i in range(0, batch_size):
        img_idx_pair1 = pick_first_img_idx(labels_start_end_train, num_targets)
        triple[0][i, :, :, :] = train_fixed_set[img_idx_pair1, :]
        img_label = img_idx_pair1

        # get image for the second: positive
        triple[1][i, :, :, :] = pick_pos_img_idx(labels_start_end_train=labels_start_end_train, X_train_new=X_train_new,
                                                 img_label=img_label)

        # get image for the third: negative from legit
        img_neg, label_neg = pick_neg_img(labels_start_end_train=labels_start_end_train, X_train_new=X_train_new,
                                          anchor_idx=img_label, num_targets=num_targets)
        while targetHelper.check_if_same_category(img_label, label_neg) == 1:
            img_neg, label_neg = pick_neg_img(
                labels_start_end_train=labels_start_end_train,
                X_train_new=X_train_new,
                anchor_idx=img_label,
                num_targets=num_targets
            )

        triple[2][i, :, :, :] = img_neg

    return triple

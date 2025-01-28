# Don't sample negative image from the same category as the positive image (e.g. google and google drive)
# Create clusters of same-company websites (e.g. all microsoft websites)
import logging


class TargetHelper:
    # targets names of parent and sub websites
    target_lists = [['microsoft', 'ms_outlook', 'ms_office', 'ms_bing', 'ms_onedrive', 'ms_skype'],
                    ['apple', 'itunes', 'icloud'], ['google', 'google_drive'], ['alibaba', 'aliexpress']]
    parents_targets_idx = [90, 12, 65, 4]
    sub_targets = [[150, 152, 151, 149, 148], [153, 154], [147], [5]]

    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        self.read_targets(data_path)

    def read_targets(self, data_path):
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

    def check_if_target_in_top(self, test_file_name, only_names):
        found = 0
        idx = 0
        test_label = self.get_label_from_name(test_file_name)
        self.logger.info('***')
        self.logger.info('Test example: %s', test_file_name)
        for i in range(0, len(only_names)):
            label_distance = self.get_label_from_name(only_names[i])
            if label_distance == test_label or self.check_if_same_category(test_label, label_distance) == 1:
                found = 1
                idx = i + 1
                self.logger.info('found')
                break
        if found == 0:
            self.logger.info('not found')
        return found, idx

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

    # Get file names of each example
    @staticmethod
    def read_file_names(data_path, file_name):
        with open(data_path / file_name, "r") as targets_file:
            targets = targets_file.read()

        file_names_list = []
        targets_list = targets.splitlines()
        for i in range(0, len(targets_list)):
            target_path = data_path / targets_list[i]
            file_names = sorted(target_path.iterdir())
            for j in range(0, len(file_names)):
                file_names_list.append(file_names[j])
        return file_names_list

    @staticmethod
    def get_label_from_name(name):
        first_half = name.split('_', 1)[0]
        number = int(first_half.replace('T', ''))
        return number

# TODO: rename file to make it more descriptive
import logging
import logging.config

import os
from argparse import ArgumentParser
from dataclasses import dataclass

from skimage.transform import resize
from matplotlib.pyplot import imread
from sklearn.model_selection import train_test_split
import numpy as np

from tools.config import SRC_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR


def read_imgs_per_website(data_path, targets, imgs_num, reshape_size, start_target_count):
    all_imgs = np.zeros(shape=[imgs_num, 224, 224, 3])
    all_labels = np.zeros(shape=[imgs_num, 1])

    all_file_names = []
    targets_list = targets.splitlines()
    count = 0
    for i in range(0, len(targets_list)):
        target_path = data_path / targets_list[i]
        print(target_path)
        file_names = sorted(os.listdir(target_path))
        for j in range(0, len(file_names)):
            try:
                img = imread(target_path / file_names[j])
                img = img[:, :, 0:3]
                all_imgs[count, :, :, :] = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
                all_labels[count, :] = i + start_target_count
                all_file_names.append(file_names[j])
                count = count + 1
            except:
                # some images were saved with a wrong extensions
                try:
                    img = imread(target_path / file_names[j], format='jpeg')
                    img = img[:, :, 0:3]
                    all_imgs[count, :, :, :] = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
                    all_labels[count, :] = i + start_target_count
                    all_file_names.append(file_names[j])
                    count = count + 1
                except:
                    print('failed at:')
                    print('***')
                    print(file_names[j])
                    break
    return all_imgs, all_labels, all_file_names


def read_or_load_imgs(args):
    logging.config.fileConfig(SRC_DIR / 'logging.conf')
    logger = logging.getLogger(__name__)
    logger.info('Check for pre-saved data or load images')

    # Define paths for saved .npy files
    imgs_train_path = args.output_dir / 'all_imgs_train.npy'
    labels_train_path = args.output_dir / 'all_labels_train.npy'
    file_names_train_path = args.output_dir / 'all_file_names_train.npy'

    imgs_test_path = args.output_dir / 'all_imgs_test.npy'
    labels_test_path = args.output_dir / 'all_labels_test.npy'
    file_names_test_path = args.output_dir / 'all_file_names_test.npy'

    # Check if all .npy files exist
    if (imgs_train_path.exists() and labels_train_path.exists() and file_names_train_path.exists() and
            imgs_test_path.exists() and labels_test_path.exists() and file_names_test_path.exists()):
        logger.info('Loading pre-saved data')

        # Load pre-saved data
        all_imgs_train = np.load(imgs_train_path)
        all_labels_train = np.load(labels_train_path)
        all_file_names_train = np.load(file_names_train_path)

        all_imgs_test = np.load(imgs_test_path)
        all_labels_test = np.load(labels_test_path)
        all_file_names_test = np.load(file_names_test_path)

    else:
        logger.info('Processing and saving images')

        data_path_trusted = args.dataset_path / 'trusted_list'
        data_path_phish = args.dataset_path / 'phishing'

        # Read images legit (train)
        with open(data_path_trusted / 'targets.txt', 'r') as f:
            targets_trusted = f.read()
        all_imgs_train, all_labels_train, all_file_names_train = read_imgs_per_website(data_path_trusted,
                                                                                       targets_trusted,
                                                                                       args.legit_imgs_num,
                                                                                       args.reshape_size, 0)

        imgs_train_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(imgs_train_path, all_imgs_train)
        np.save(labels_train_path, all_labels_train)
        np.save(file_names_train_path, all_file_names_train)

        # Read images phishing (test)
        with open(data_path_phish / 'targets.txt', 'r') as f:
            targets_phishing = f.read()
        all_imgs_test, all_labels_test, all_file_names_test = read_imgs_per_website(data_path_phish,
                                                                                    targets_phishing,
                                                                                    args.phish_imgs_num,
                                                                                    args.reshape_size, 0)

        imgs_test_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(imgs_test_path, all_imgs_test)
        np.save(labels_test_path, all_labels_test)
        np.save(file_names_test_path, all_file_names_test)

    return all_imgs_train, all_labels_train, all_file_names_train, all_imgs_test, all_labels_test, all_file_names_test


def get_phish_file_names(phish_file_names, phish_train_idx, phish_test_idx):
    phish_train_file_names = []
    for i in range(0, phish_train_idx.shape[0]):
        phish_train_file_names.append(phish_file_names[phish_train_idx[i]])

    phish_train_file_names = [phish_file_names[idx] for idx in phish_train_idx]

    phish_test_file_names = []
    for i in range(0, phish_test_idx.shape[0]):
        phish_test_file_names.append(phish_file_names[phish_test_idx[i]])

    return phish_train_file_names, phish_test_file_names


def read_or_load_train_test_idx(output_dir, all_imgs_test, all_labels_test, phishing_test_size):
    idx_test, idx_train = None, None
    if (output_dir / 'test_idx.npy').exists() and (output_dir / 'train_idx.npy').exists():
        idx_train = np.load(output_dir / 'train_idx.npy')
        idx_test = np.load(output_dir / 'test_idx.npy')
    else:
        idx = np.arange(all_imgs_test.shape[0])
        _, _, _, _, idx_test, idx_train = train_test_split(all_imgs_test, all_labels_test, idx,
                                                           test_size=phishing_test_size)
        np.save(output_dir / 'test_idx', idx_test)
        np.save(output_dir / 'train_idx', idx_train)

    return idx_test, idx_train


# TODO: rename as it is not clear what it does (contains embeddings and labels only)
@dataclass
class TrainResults:
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


def save_embeddings(emb: TrainResults, output_dir, run=None):
    np.save(output_dir / 'whitelist_emb', emb.X_legit_train)
    np.save(output_dir / 'whitelist_labels', emb.y_legit_train)

    np.save(output_dir / 'phishing_emb', emb.X_phish)
    np.save(output_dir / 'phishing_labels', emb.y_phish)

    if run is not None:
        run.save(output_dir / 'whitelist_emb.npy')
        run.save(output_dir / 'whitelist_labels.npy')
        run.save(output_dir / 'phishing_emb.npy')
        run.save(output_dir / 'phishing_labels.npy')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset-path', type=str, default=RAW_DATA_DIR / 'VisualPhish')
    parser.add_argument('--output-dir', default=INTERIM_DATA_DIR / 'VisualPhish')

    data = read_or_load_imgs(parser.parse_args())

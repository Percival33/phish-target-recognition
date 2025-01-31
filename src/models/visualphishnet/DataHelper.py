# TODO: rename file to make it more descriptive
import logging
import logging.config
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tools.config import INTERIM_DATA_DIR, RAW_DATA_DIR, setup_logging
from tqdm import tqdm


def read_image(file_path, logger, format=None):
    """Helper function to read an image file with error handling."""
    try:
        img = imread(file_path) if format is None else imread(file_path, format=format)
        if len(img.shape) != 3 or img.shape[2] < 3:
            logger.warning(f"Skipping {file_path}: wrong number of channels")
            return None
        return img[:, :, :3]  # Take only first 3 channels if more exist
    except Exception as read_error:
        logger.warning(f"Image reading error for {file_path}: {str(read_error)}")
        return None


def read_imgs_per_website(data_path, targets, imgs_num, reshape_size, start_target_count):
    """Read and process images from multiple directories."""
    logger = logging.getLogger(__name__)
    logger.info("Starting image processing")

    all_imgs = np.zeros(shape=[imgs_num, reshape_size[0], reshape_size[1], 3])
    all_labels = np.zeros(shape=[imgs_num, 1])
    all_file_names = []

    targets_list = sorted(targets.strip().splitlines())
    count = 0

    with tqdm(targets_list, desc="Processing directories", position=0) as dir_pbar:
        for i, target_dir in enumerate(dir_pbar):
            target_path = data_path / target_dir

            files = sorted(target_path.iterdir())
            with tqdm(files, desc=f"Processing {target_path.name}", position=1, leave=False) as file_pbar:
                for file_path in file_pbar:
                    img = read_image(file_path, logger)

                    if img is None:
                        img = read_image(file_path, logger, format="jpeg")

                    if img is None:
                        logger.error(f"Failed to process {file_path}")
                        exit(1)

                    try:
                        all_imgs[count] = resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True)
                        all_labels[count] = i + start_target_count
                        all_file_names.append(file_path.name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {str(e)}")
                        logger.error("Full traceback:", exc_info=True)
                        exit(1)

    if count < imgs_num:
        logger.warning(f"Only found {count} images, expected {imgs_num}")
        return all_imgs[:count], all_labels[:count], all_file_names

    return all_imgs, all_labels, all_file_names


def get_data_paths(output_dir):
    """Return paths for all data files."""
    return {
        "train": {
            "imgs": output_dir / "all_imgs_train.npy",
            "labels": output_dir / "all_labels_train.npy",
            "file_names": output_dir / "all_file_names_train.npy",
        },
        "test": {
            "imgs": output_dir / "all_imgs_test.npy",
            "labels": output_dir / "all_labels_test.npy",
            "file_names": output_dir / "all_file_names_test.npy",
        },
    }


def all_files_exist(paths_dict):
    """Check if all files in the paths dictionary exist."""
    return all(path.exists() for subset in paths_dict.values() for path in subset.values())


def load_saved_data(paths_dict):
    """Load data from saved .npy files."""
    train_data = {
        "imgs": np.load(paths_dict["train"]["imgs"]),
        "labels": np.load(paths_dict["train"]["labels"]),
        "file_names": np.load(paths_dict["train"]["file_names"]),
    }

    test_data = {
        "imgs": np.load(paths_dict["test"]["imgs"]),
        "labels": np.load(paths_dict["test"]["labels"]),
        "file_names": np.load(paths_dict["test"]["file_names"]),
    }

    return train_data, test_data


def process_dataset(data_path, targets_file, num_imgs, reshape_size, start_label, output_paths):
    """Process a single dataset (train or test) and save results."""
    with open(data_path / targets_file, "r") as f:
        targets = f.read()

    imgs, labels, file_names = read_imgs_per_website(data_path, targets, num_imgs, reshape_size, start_label)

    output_paths["imgs"].parent.mkdir(parents=True, exist_ok=True)
    np.save(output_paths["imgs"], imgs)
    np.save(output_paths["labels"], labels)
    np.save(output_paths["file_names"], file_names)

    return {"imgs": imgs, "labels": labels, "file_names": file_names}


def read_or_load_imgs(args):
    """Load pre-saved data or process and save new data."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting data loading process")

    paths_dict = get_data_paths(args.dataset_path)

    if all_files_exist(paths_dict):
        logger.info("Loading pre-saved data")
        train_data, test_data = load_saved_data(paths_dict)
        logger.info(f"Loaded {len(train_data['imgs'])} training and {len(test_data['imgs'])} test samples")
    else:
        logger.info("Processing new data")

        train_data = process_dataset(
            data_path=args.dataset_path / "trusted_list",
            targets_file="targets.txt",
            num_imgs=args.legit_imgs_num,
            reshape_size=args.reshape_size,
            start_label=0,
            output_paths=paths_dict["train"],
        )
        logger.info(f"Processed {len(train_data['imgs'])} training samples")

        # Process phishing (test) dataset
        test_data = process_dataset(
            data_path=args.dataset_path / "phishing",
            targets_file="targets.txt",
            num_imgs=args.phish_imgs_num,
            reshape_size=args.reshape_size,
            start_label=0,
            output_paths=paths_dict["test"],
        )
        logger.info(f"Processed {len(test_data['imgs'])} test samples")

    return (
        train_data["imgs"],
        train_data["labels"],
        train_data["file_names"],
        test_data["imgs"],
        test_data["labels"],
        test_data["file_names"],
    )


def get_phish_file_names(phish_file_names, phish_train_idx, phish_test_idx):
    phish_train_file_names = [phish_file_names[idx] for idx in phish_train_idx]
    phish_test_file_names = [phish_file_names[idx] for idx in phish_test_idx]

    return phish_train_file_names, phish_test_file_names


def read_or_load_train_test_idx(dirname, all_imgs_test, all_labels_test, phishing_test_size):
    idx_test, idx_train = None, None
    if (dirname / "test_idx.npy").exists() and (dirname / "train_idx.npy").exists():
        idx_train = np.load(dirname / "train_idx.npy")
        idx_test = np.load(dirname / "test_idx.npy")
    else:
        idx = np.arange(all_imgs_test.shape[0])
        _, _, _, _, idx_test, idx_train = train_test_split(
            all_imgs_test, all_labels_test, idx, test_size=phishing_test_size
        )
        dirname.mkdir(parents=True, exist_ok=True)
        np.save(dirname / "test_idx", idx_test)
        np.save(dirname / "train_idx", idx_train)

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
    np.save(output_dir / "whitelist_emb", emb.X_legit_train)
    np.save(output_dir / "whitelist_labels", emb.y_legit_train)

    np.save(output_dir / "phishing_emb", emb.X_phish)
    np.save(output_dir / "phishing_labels", emb.y_phish)

    if run is not None:
        run.save(output_dir / "whitelist_emb.npy")
        run.save(output_dir / "whitelist_labels.npy")
        run.save(output_dir / "phishing_emb.npy")
        run.save(output_dir / "phishing_labels.npy")


def targets_start_end(num_target, labels):
    prev_target = labels[0]
    start_end_each_target = np.zeros((num_target, 2))
    start_end_each_target[0, 0] = labels[0]
    if not labels[0] == 0:
        start_end_each_target[0, 0] = -1
        start_end_each_target[0, 1] = -1
    # count_target = 0
    for i in range(1, labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[int(labels[i - 1]), 1] = int(i - 1)
            start_end_each_target[int(labels[i]), 0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]), 1] = int(labels.shape[0] - 1)

    for i in range(1, num_target):
        if start_end_each_target[i, 0] == 0:
            start_end_each_target[i, 0] = -1
            start_end_each_target[i, 1] = -1
    return start_end_each_target


# Store the start and end of each target in the training set (used later in triplet sampling)
def all_targets_start_end(num_target, labels, logger):
    prev_target = labels[0]
    start_end_each_target = np.zeros((num_target, 2))
    start_end_each_target[0, 0] = labels[0]
    if not labels[0] == 0:
        start_end_each_target[0, 0] = -1
        start_end_each_target[0, 1] = -1
    count_target = 0
    for i in range(1, labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[int(labels[i - 1]), 1] = int(i - 1)
            count_target = count_target + 1
            start_end_each_target[int(labels[i]), 0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]), 1] = int(labels.shape[0] - 1)

    for i in range(1, num_target):
        if start_end_each_target[i, 0] == 0:
            logger.warning(f"Target {str(i)} is not in the training set")
            start_end_each_target[i, 0] = -1
            start_end_each_target[i, 1] = -1
    return start_end_each_target


# Order random phishing arrays per website (from 0 to 155 target)
def order_random_array(orig_arr, y_orig_arr, targets):
    # TODO: remove duplicate with HardSubsetSampling
    sorted_arr = np.zeros(orig_arr.shape)
    y_sorted_arr = np.zeros(y_orig_arr.shape)
    count = 0
    for i in range(0, targets):
        for j in range(0, orig_arr.shape[0]):
            if y_orig_arr[j] == i:
                sorted_arr[count, :, :, :] = orig_arr[j, :, :, :]
                y_sorted_arr[count, :] = i
                count = count + 1
    return sorted_arr, y_sorted_arr


if __name__ == "__main__":
    setup_logging()
    parser = ArgumentParser()

    parser.add_argument("--dataset-path", type=str, default=RAW_DATA_DIR / "VisualPhish")
    parser.add_argument("--output-dir", default=INTERIM_DATA_DIR / "VisualPhish")
    parser.add_argument("--reshape-size", default=[224, 224, 3])
    parser.add_argument("--legit-imgs-num", default=9363)
    parser.add_argument("--phish-imgs-num", default=1195)
    args = parser.parse_args()
    data = read_or_load_imgs(parser.parse_args())

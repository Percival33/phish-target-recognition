from pathlib import Path
from torch.utils.data import Dataset
from perception import hashers
from collections import defaultdict
from src.config import config
import joblib

hasher = hashers.PHashF()
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR


class PhishIRISDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", preprocess: bool = False):
        """
        Initialize the dataset object.

        Args:
            data_dir (Path): Path to the root data directory.
            split (str): Dataset split, either 'train' or 'val'.
            preprocess (bool): Whether to preprocess and save the dataset.
        """
        self.data_dir = data_dir
        self.split = split.lower()  # Normalize split to lowercase
        self.preprocess = preprocess
        self.pickle_path = PROCESSED_DATA_DIR / f"phishIRIS_{self.split}.pkl"

        self.img_paths = []
        self.labels = []
        self.img_to_hash = {}
        self.hash_to_img = {}
        self.hash_to_company = defaultdict(str)
        self.img_per_company = defaultdict(list)

        if preprocess:
            self._process_and_save()
        else:
            self._load()

    def _read_images(self, folder):
        """Read images and metadata from the given directory."""
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(
                f"Folder {folder} does not exist or is not a directory"
            )

        for company_path in folder.iterdir():
            if not company_path.is_dir():
                continue
            company_name = company_path.name

            for img_path in company_path.iterdir():
                if not img_path.is_file():
                    continue

                img_path_str = str(img_path)
                self.img_paths.append(img_path_str)
                self.labels.append(company_name)

                hsh = hasher.compute(img_path_str)
                formatted_name = "-".join(img_path.name.split(" "))
                self.img_to_hash[formatted_name] = hasher.string_to_vector(hsh)
                self.hash_to_img[hsh] = formatted_name
                self.hash_to_company[hsh] = company_name
                self.img_per_company[company_name].append(hsh)

    def _process_and_save(self):
        """Process data and save to disk."""
        split_dir = (
            self.data_dir / "train" if self.split == "train" else self.data_dir / "val"
        )
        self._read_images(split_dir)
        joblib.dump(self, self.pickle_path)

    def _load(self):
        """Load preprocessed data from disk."""
        if self.pickle_path.exists():
            data = joblib.load(self.pickle_path)
            self.img_paths = data.img_paths
            self.labels = data.labels
            self.img_to_hash = data.img_to_hash
            self.hash_to_img = data.hash_to_img
            self.hash_to_company = data.hash_to_company
            self.img_per_company = data.img_per_company
        else:
            raise FileNotFoundError(
                f"Preprocessed file not found at {self.pickle_path}"
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.img_paths[idx], self.labels[idx]

from dataclasses import dataclass, field
from perception import hashers
from collections import defaultdict
from src.config import DATA_DIR

hasher = hashers.PHashF()
TRAIN_DIR = DATA_DIR / "raw" / "phishIRIS_DL_Dataset/train"


@dataclass
class PhishIRISData:
    img_to_hash: dict = field(default_factory=dict)
    hash_to_img: dict = field(default_factory=dict)
    img_per_company: defaultdict = field(default_factory=lambda: defaultdict(list))
    hash_to_company: defaultdict = field(default_factory=lambda: defaultdict(str))
    imgs: list = field(default_factory=list)
    _loaded: bool = field(default=False, init=False)  # Track if data has been loaded

    def load_data(self):
        """Load data from the dataset directory into the data structures."""
        if self._loaded:
            return  # Skip loading if already loaded

        for company_path in TRAIN_DIR.iterdir():
            if not company_path.is_dir() or company_path.name in {".DS_Store", "other"}:
                continue

            company_name = company_path.name
            for example_path in company_path.iterdir():
                if not example_path.is_file():
                    continue

                self.imgs.append(str(example_path))

                hsh = hasher.compute(str(example_path))
                formatted_name = "-".join(example_path.name.split(" "))
                self.img_to_hash[formatted_name] = hasher.string_to_vector(hsh)
                self.hash_to_img[hsh] = formatted_name
                self.hash_to_company[hsh] = company_name
                self.img_per_company[company_name].append(hsh)

        self._loaded = True

    def get_data(self):
        """Return loaded data, ensuring it is loaded first."""
        if not self._loaded:
            self.load_data()
        return self

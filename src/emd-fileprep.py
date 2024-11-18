import os
from os import listdir

from src.config import DATA_DIR

TRAIN_DIR = DATA_DIR / "phishIRIS_DL_Dataset" / "train"
OUTPUT_DIR = DATA_DIR / "EMD" / "test_sites"


def main():
    for company in listdir(TRAIN_DIR):
        if company in [".DS_Store", "other"]:
            continue

        imgs = listdir(str(TRAIN_DIR / company))

        for idx, img in enumerate(imgs):
            folder_name = f"{company}+{idx}"
            os.mkdir(str(OUTPUT_DIR / folder_name))
            print(folder_name)
            with open(str(TRAIN_DIR / company / img), "rb") as src_file:
                with open(str(OUTPUT_DIR / folder_name / "shot.png"), "wb") as dst_file:
                    dst_file.write(src_file.read())
            with open(str(OUTPUT_DIR / folder_name / "info.txt"), "w") as f:
                f.write(f"https://www.{company}.com")


main()

import pandas as pd
from pathlib import Path
import shutil
from tools.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from tqdm import tqdm
import argparse

def copy_and_rename_images(src_folder, dest_folder, is_phish):
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    data = []

    images = sorted(src_folder.glob("*"))
    num_images = len(images)
    name_length = len(str(num_images - 1))

    for i, image_path in tqdm(enumerate(images), total=num_images):
        if not image_path.is_dir() or image_path.name == ".DS_Store":
            continue

        new_name = f"{str(i).zfill(name_length)}.png"
        new_path = dest_folder / new_name
        shutil.copy(image_path / "shot.png", new_path)
        data.append(
            {
                # old path should start from this path: data/raw/phishpedia
                "new_name": new_name,
                # 'path': str(image_path),
                "path": str(image_path.relative_to(src_folder.parent)),
                "target": image_path.name.split("+")[0].lower(),
                "isPhish": is_phish,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(
        dest_folder.parent / ("phish.csv" if is_phish else "benign.csv"), index=False
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename images from source to destination folder")
    parser.add_argument(
        "--src_folder",
        type=str,
        default=str(RAW_DATA_DIR / "phishpedia" / "phish_sample_30k"),
        help="Source folder containing images"
    )
    parser.add_argument(
        "--dest_folder", 
        type=str,
        default=str(INTERIM_DATA_DIR / "phishpedia" / "phish_sample_30k"),
        help="Destination folder for renamed images"
    )
    parser.add_argument(
        "--is_phish",
        type=bool,
        default=True,
        help="Whether the images are phishing samples"
    )

    args = parser.parse_args()
    copy_and_rename_images(args.src_folder, args.dest_folder, args.is_phish)

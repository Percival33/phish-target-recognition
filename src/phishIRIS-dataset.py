import argparse
import pathlib
import shutil
from src.config import config


def get_url(target: str):
    return config.TARGET_URL.get(target, f"https://www.{target}.com")


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
args = parser.parse_args()

dataset_dir = args.input
output_dir = args.output

# if dataset_dir does not exist throw error
if not pathlib.Path(dataset_dir).exists():
    raise FileNotFoundError(f"{dataset_dir} does not exist")

# if output directory does not exist create it
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

for company in pathlib.Path(dataset_dir).iterdir():
    if not company.is_dir():
        continue
    counter = 1
    for img in company.iterdir():
        # create info.txt and shot.png
        new_img_name = "shot.png"
        new_img_path = pathlib.Path(output_dir) / company.name / new_img_name

        while new_img_path.exists():
            counter += 1
            new_img_path = (
                pathlib.Path(output_dir) / f"{company.name}-{counter}" / new_img_name
            )
        new_img_path.parent.mkdir(parents=True, exist_ok=False)

        with open(new_img_path.parent / "info.txt", "w") as f:
            f.write(get_url(company.name))
        shutil.copy(img, new_img_path)
        print(f"Created {new_img_path}")

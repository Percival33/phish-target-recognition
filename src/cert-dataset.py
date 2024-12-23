import argparse
import pathlib
import shutil
import csv

parser = argparse.ArgumentParser()
parser.add_argument(
    "--index", required=True, type=str, help="Path to the dataset index file"
)
parser.add_argument(
    "--output", required=True, type=str, help="Path to the output directory"
)

args = parser.parse_args()

index_path = args.index
output_dir = args.output

# if output directory does not exist create it
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Open and read the CSV file
with open(index_path, mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        url = row["url"]
        img_path = pathlib.Path(index_path).parent / row["screenshot_object"]
        target = row["affected_entity"]
        folder_name = row["screenshot_hash"]

        new_sample_path = pathlib.Path(output_dir) / folder_name
        new_sample_path.mkdir(parents=True, exist_ok=False)

        with open(new_sample_path / "info.txt", "w") as f:
            f.write(url)

        if not img_path.exists():
            raise FileNotFoundError(f"{img_path} does not exist")

        shutil.copy(img_path, new_sample_path)

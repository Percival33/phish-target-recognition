# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "tqdm",
# ]
# ///

import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm


def organize_by_sample(csv_path, screenshots_path, output_path):
    """
    Organize screenshots into separate sample folders, where each sample folder contains:
    - info.txt with the full URL
    - shot.png containing the screenshot

    Samples are placed directly in the phishing/trusted_list directories, not grouped by target.

    Args:
        csv_path: Path to the CSV file containing data
        screenshots_path: Path to the folder containing screenshots
        output_path: Path where the organized folders will be created
    """
    df = pd.read_csv(csv_path)

    required_columns = [
        "url",
        "fqdn",
        "screenshot_object",
        "affected_entity",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    output_path = Path(output_path)
    phishing_dir = output_path / "phishing"
    trusted_dir = output_path / "trusted_list"
    phishing_dir.mkdir(exist_ok=True, parents=True)
    trusted_dir.mkdir(exist_ok=True, parents=True)

    phishing_targets = set()
    trusted_targets = set()

    print("Organizing samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        target = (
            row["affected_entity"].lower()
            if not pd.isna(row["affected_entity"])
            else "unknown"
        )

        is_phishing = int(row.get("is_phishing", 1))
        if is_phishing:
            phishing_targets.add(target)
            parent_dir = phishing_dir
        else:
            trusted_targets.add(target)
            parent_dir = trusted_dir

        sample_id = f"sample{idx + 1}"
        sample_dir = parent_dir / f"{target}+{sample_id}"
        sample_dir.mkdir(exist_ok=True)

        screenshot_path = row["screenshot_object"]
        if not Path(screenshot_path).is_absolute():
            screenshot_path = Path(screenshots_path) / screenshot_path

        if Path(screenshot_path).exists():
            destination = sample_dir / "shot.png"
            shutil.copy2(screenshot_path, destination)

            with open(sample_dir / "info.txt", "w") as f:
                f.write(f"{row['url']}")

            print(f"Created sample {sample_id} in {parent_dir}")
        else:
            print(f"Warning: Screenshot not found at {screenshot_path}")
            sample_dir.rmdir()

    print(
        f"Organized {len(df)} entries into {len(phishing_targets)} phishing targets and {len(trusted_targets)} trusted targets"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Organize screenshots into separate sample folders"
    )
    parser.add_argument("--csv", required=True, help="Path to the CSV file with data")
    parser.add_argument(
        "--screenshots", required=True, help="Path to the folder containing screenshots"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the organized folders will be created",
    )

    args = parser.parse_args()

    organize_by_sample(args.csv, args.screenshots, args.output)


if __name__ == "__main__":
    main()

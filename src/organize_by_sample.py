#!/usr/bin/env python3
import os
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
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = [
        "url",
        "fqdn",
        "screenshot_object",
        "screenshot_hash",
        "affected_entity",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Create output directories
    phishing_dir = os.path.join(output_path, "phishing")
    trusted_dir = os.path.join(output_path, "trusted_list")
    os.makedirs(phishing_dir, exist_ok=True)
    os.makedirs(trusted_dir, exist_ok=True)

    # Track targets
    phishing_targets = set()
    trusted_targets = set()

    # Process each row
    print("Organizing samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get target name (affected entity)
        target = (
            row["affected_entity"].lower()
            if not pd.isna(row["affected_entity"])
            else "unknown"
        )

        # Check if this is a phishing or benign sample
        is_phishing = True
        if "is_phishing" in df.columns:
            is_phishing = False if row["is_phishing"] == False else True

        # Track target
        if is_phishing:
            phishing_targets.add(target)
            parent_dir = phishing_dir
        else:
            trusted_targets.add(target)
            parent_dir = trusted_dir

        # Create sample directory (sample1, sample2, etc.) directly in phishing/trusted_list folder
        sample_id = f"sample{idx + 1}"
        sample_dir = os.path.join(parent_dir, f"{target}+{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        # Get screenshot path
        screenshot_path = row["screenshot_object"]
        if not os.path.isabs(screenshot_path):
            screenshot_path = os.path.join(screenshots_path, screenshot_path)

        # Copy screenshot to sample directory as shot.png
        if os.path.exists(screenshot_path):
            destination = os.path.join(sample_dir, "shot.png")
            shutil.copy2(screenshot_path, destination)

            # Create info.txt with the URL and target information
            with open(os.path.join(sample_dir, "info.txt"), "w") as f:
                f.write(f"{row['url']}\n")

            print(f"Created sample {sample_id} in {parent_dir}")
        else:
            print(f"Warning: Screenshot not found at {screenshot_path}")
            # Remove the sample directory if screenshot doesn't exist
            os.rmdir(sample_dir)

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

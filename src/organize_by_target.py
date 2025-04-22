import pandas as pd
from pathlib import Path
import os
import shutil
from tqdm import tqdm


def organize_screenshots_by_target(csv_path, screenshots_dir, output_dir):
    """
    Organize screenshots into trusted_list and phishing folders based on targets.

    Args:
        csv_path (str): Path to CSV file with columns url, fqdn, screenshot_object, screenshot_hash, affected_entity
        screenshots_dir (str): Directory containing screenshots
        output_dir (str): Directory where organized folders will be created
    """
    # Create output directories
    output_dir = Path(output_dir)
    phishing_dir = output_dir / "phishing"
    trusted_dir = output_dir / "trusted_list"

    phishing_dir.mkdir(parents=True, exist_ok=True)
    trusted_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Group by affected_entity and is_phishing (if exists, otherwise assume all are phishing)
    if "is_phishing" in df.columns:
        phishing_df = df[df["is_phishing"] == True]
        trusted_df = df[df["is_phishing"] == False]
    else:
        phishing_df = df
        trusted_df = pd.DataFrame()

    # Process phishing samples
    phishing_targets = set()
    targets_processed = {}

    # First collect all unique targets and sort them alphabetically
    all_phishing_targets = sorted(
        [
            row["affected_entity"].lower()
            if pd.notna(row["affected_entity"])
            else "unknown"
            for _, row in phishing_df.iterrows()
        ]
    )
    unique_phishing_targets = []
    for target in all_phishing_targets:
        if target not in unique_phishing_targets:
            unique_phishing_targets.append(target)

    # Create a mapping from target to target_id
    target_to_id = {target: idx for idx, target in enumerate(unique_phishing_targets)}

    print("Organizing phishing samples...")
    for idx, row in tqdm(phishing_df.iterrows(), total=len(phishing_df)):
        target = (
            row["affected_entity"].lower()
            if pd.notna(row["affected_entity"])
            else "unknown"
        )
        screenshot = row["screenshot_object"]

        # Skip if screenshot file doesn't exist
        screenshot_path = Path(screenshots_dir) / screenshot
        if not screenshot_path.exists():
            continue

        # Add to the set of processed targets
        phishing_targets.add(target)

        # Initialize target counter if not exists
        if target not in targets_processed:
            targets_processed[target] = 0
            # Create target directory
            (phishing_dir / target).mkdir(exist_ok=True)

        # Create filename with format T{target_id}_{counter}.png
        target_id = target_to_id[target]
        filename = f"T{target_id}_{targets_processed[target]}.png"

        # Copy file
        shutil.copy(screenshot_path, phishing_dir / target / filename)

        # Increment counter
        targets_processed[target] += 1

    # Create targets2.txt in phishing folder with sorted targets
    sorted_phishing_targets = sorted(list(phishing_targets))
    with open(phishing_dir / "targets2.txt", "w") as f:
        for target in sorted_phishing_targets:
            f.write(f"{target}\n")

    # Reset counters for trusted samples
    trusted_targets = set()
    targets_processed = {}

    # First collect all unique targets and sort them alphabetically
    all_trusted_targets = sorted(
        [
            row["affected_entity"].lower()
            if pd.notna(row["affected_entity"])
            else "unknown"
            for _, row in trusted_df.iterrows()
        ]
    )
    unique_trusted_targets = []
    for target in all_trusted_targets:
        if target not in unique_trusted_targets:
            unique_trusted_targets.append(target)

    # Create a mapping from target to target_id
    target_to_id = {target: idx for idx, target in enumerate(unique_trusted_targets)}

    print("Organizing trusted samples...")
    for idx, row in tqdm(trusted_df.iterrows(), total=len(trusted_df)):
        target = (
            row["affected_entity"].lower()
            if pd.notna(row["affected_entity"])
            else "unknown"
        )
        screenshot = row["screenshot_object"]

        # Skip if screenshot file doesn't exist
        screenshot_path = Path(screenshots_dir) / screenshot
        if not screenshot_path.exists():
            continue

        # Add to the set of processed targets
        trusted_targets.add(target)

        # Initialize target counter if not exists
        if target not in targets_processed:
            targets_processed[target] = 0
            # Create target directory
            (trusted_dir / target).mkdir(exist_ok=True)

        # Create filename with format T{target_id}_{counter}.png
        target_id = target_to_id[target]
        filename = f"T{target_id}_{targets_processed[target]}.png"

        # Copy file
        shutil.copy(screenshot_path, trusted_dir / target / filename)

        # Increment counter
        targets_processed[target] += 1

    # Create targets.txt in trusted_list folder with sorted targets
    sorted_trusted_targets = sorted(list(trusted_targets))
    with open(trusted_dir / "targets.txt", "w") as f:
        for target in sorted_trusted_targets:
            f.write(f"{target}\n")

    print(f"Files organized into {output_dir}")
    print(f"Created targets2.txt with {len(sorted_phishing_targets)} phishing targets")
    print(f"Created targets.txt with {len(sorted_trusted_targets)} trusted targets")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize screenshots by target into phishing and trusted folders"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--screenshots", required=True, help="Directory containing screenshots"
    )
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    organize_screenshots_by_target(args.csv, args.screenshots, args.output)

#!/usr/bin/env python3
"""
Simple dataset creation script for phishing target recognition.
Creates symlinked datasets from phishpedia data in VisualPhish format.

VisualPhish format structure:
output_directory/
├── trusted_list/
│   ├── target_name_1/
│   │   ├── T0_0.png, T0_1.png, etc.
│   ├── target_name_2/
│   │   ├── T1_0.png, T1_1.png, etc.
│   └── targets.txt
├── phishing/
│   ├── target_name_1/
│   │   ├── T0_0.png, T0_1.png, etc.
│   ├── target_name_2/
│   │   ├── T1_0.png, T1_1.png, etc.
│   └── targets.txt

Usage:
    uv run create_dataset.py \
        --benign-dir /path/to/benign/folders \
        --phishing-dir /path/to/phishing/folders \
        --output-dir /path/to/output \
        --format visualphish
"""

import json
import os
import argparse
import pickle
import sys
import re
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent / "src" / "models" / "phishpedia" / "scripts")
)
from organize import get_special_domain_mapping


def load_domain_mapping():
    """Load target name mappings from pickle file or fallback to hardcoded mapping."""
    try:
        domain_map_path = (
            Path(__file__).parent.parent
            / "src"
            / "models"
            / "phishpedia"
            / "models"
            / "domain_map.pkl"
        )
        with open(domain_map_path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, PermissionError):
        print("Warning: Could not load domain_map.pkl, using fallback mapping")
        return get_special_domain_mapping()


def load_target_mappings():
    """Load target name mappings from target_mappings.json and expand them."""
    try:
        with open(Path(__file__).parent.parent / "target_mappings.json", "r") as f:
            original_mappings = json.load(f)

        # Expand mappings so each item in values also becomes a key
        # For key: [item1, item2] -> create:
        # key: [key, item1, item2]
        # item1: [key, item1, item2]
        # item2: [key, item1, item2]
        expanded_mappings = {}

        for key, values in original_mappings.items():
            all_items = [key] + values

            expanded_mappings[key] = all_items

            for value in values:
                expanded_mappings[value] = all_items

        return expanded_mappings
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load target_mappings.json: {e}")
        return {}


def sanitize_folder_name(name):
    """Sanitize target name for use as folder name."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_.")
    return sanitized


def save_visualphish_format(
    target_name, target_num, benign_images, phishing_images, output_dir
):
    """Save images in VisualPhish format using symlinks - trusted_list and phishing folders with target subfolders."""
    sanitized_name = sanitize_folder_name(target_name)

    trusted_list_dir = Path(output_dir) / "trusted_list"
    phishing_dir = Path(output_dir) / "phishing"

    benign_count = 0
    phishing_count = 0

    if benign_images:
        benign_target_folder = trusted_list_dir / sanitized_name
        benign_target_folder.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(benign_images):
            if Path(img_path).exists():
                link_name = f"T{target_num}_{i}.png"
                link_path = benign_target_folder / link_name

                if link_path.exists():
                    link_path.unlink()

                os.symlink(str(Path(img_path).absolute()), str(link_path))
                benign_count += 1

    if phishing_images:
        phishing_target_folder = phishing_dir / sanitized_name
        phishing_target_folder.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(phishing_images):
            if Path(img_path).exists():
                link_name = f"T{target_num}_{i}.png"
                link_path = phishing_target_folder / link_name

                if link_path.exists():
                    link_path.unlink()

                os.symlink(str(Path(img_path).absolute()), str(link_path))
                phishing_count += 1

    return benign_count, phishing_count


def find_benign_images(domains, benign_dir):
    """Find folders matching domains and return shot.png paths."""
    image_paths = []
    benign_path = Path(benign_dir)

    if not benign_path.exists():
        print(f"Warning: Benign directory {benign_dir} does not exist")
        return image_paths

    for domain in domains:
        domain = domain.strip()
        if not domain:
            continue

        domain_folder = benign_path / domain
        if domain_folder.exists() and domain_folder.is_dir():
            shot_png = domain_folder / "shot.png"
            if shot_png.exists():
                image_paths.append(str(shot_png))
            else:
                print(f"Warning: No shot.png found in {domain_folder}")
        else:
            print(f"Warning: Domain folder {domain_folder} not found")

    return image_paths


def find_phishing_images(target_name, phishing_dir, domain_mapping, target_mappings):
    """Map target name to company name and find matching folders."""
    image_paths = []
    phishing_path = Path(phishing_dir)

    if not phishing_path.exists():
        print(f"Warning: Phishing directory {phishing_dir} does not exist")
        return image_paths

    search_names = {
        target_name,
        target_name.replace("_", " "),
        target_name.replace(" ", "_"),
        target_name.lower(),
        target_name.lower().replace("_", " "),
        target_name.lower().replace(" ", "_"),
    }

    if target_name in target_mappings:
        search_names.update(target_mappings[target_name])

    for search_name in search_names:
        for folder in phishing_path.iterdir():
            if folder.is_dir() and "+" in folder.name:
                # Split folder name on '+' and get the first part (target name)
                folder_target = folder.name.split("+")[0]

                if folder_target.lower() == search_name.lower():
                    for img_file in folder.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in [
                            ".png",
                            ".jpg",
                            ".jpeg",
                        ]:
                            image_paths.append(str(img_file))

                    if not any(
                        f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
                        for f in folder.iterdir()
                    ):
                        print(f"Warning: No image files found in {folder}")

        if image_paths:
            break

    if not image_paths:
        print(f"Warning: No phishing folders found for target '{target_name}'")
        raise ValueError(f"No phishing folders found for target '{target_name}'")

    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset from phishpedia data using symlinks"
    )
    parser.add_argument(
        "--benign-dir", required=True, help="Directory containing benign domain folders"
    )
    parser.add_argument(
        "--phishing-dir",
        required=True,
        help="Directory containing phishing target folders",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for dataset"
    )
    parser.add_argument(
        "--format", default="visualphish", choices=["visualphish"], help="Output format"
    )
    parser.add_argument(
        "--json-file",
        default="mappings/pp-benign-trusted-logos-targets.json",
        help="JSON file with target definitions",
    )

    args = parser.parse_args()

    try:
        json_path = Path(args.json_file)
        if not json_path.is_absolute():
            json_path = Path(__file__).parent.parent / json_path

        with open(json_path) as f:
            all_targets = json.load(f)
        targets = [t for t in all_targets if t.get("phish")]
        print(f"Loaded {len(targets)} targets with phish=true from {args.json_file}")
    except FileNotFoundError:
        print(f"Error: JSON file {args.json_file} not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.json_file}")
        return 1

    domain_mapping = load_domain_mapping()
    print(f"Loaded domain mapping with {len(domain_mapping)} entries")

    target_mappings = load_target_mappings()
    print(f"Loaded target mappings with {len(target_mappings)} entries")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    total_benign = 0
    total_phishing = 0
    benign_targets = []
    phishing_targets = []

    for i, target in enumerate(targets):
        target_name = target["target"]
        print(f"\nProcessing target {i}: {target_name}")

        domains = target["domains"].split("\n") if target.get("domains") else []
        benign_images = find_benign_images(domains, args.benign_dir)

        phishing_images = find_phishing_images(
            target_name, args.phishing_dir, domain_mapping, target_mappings
        )

        if benign_images or phishing_images:
            benign_count, phishing_count = save_visualphish_format(
                target_name, i, benign_images, phishing_images, args.output_dir
            )
            total_benign += benign_count
            total_phishing += phishing_count

            if benign_count > 0:
                benign_targets.append(target_name)
            if phishing_count > 0:
                phishing_targets.append(target_name)

            print(f"  → {benign_count} benign, {phishing_count} phishing images")
        else:
            print(f"  → No images found for {target_name}")

    if benign_targets:
        trusted_targets_file = Path(args.output_dir) / "trusted_list" / "targets.txt"
        trusted_targets_file.parent.mkdir(parents=True, exist_ok=True)
        with open(trusted_targets_file, "w") as f:
            for target_name in benign_targets:
                f.write(f"{target_name}\n")
        print(f"\nCreated trusted_list/targets.txt with {len(benign_targets)} targets")

    if phishing_targets:
        phishing_targets_file = Path(args.output_dir) / "phishing" / "targets.txt"
        phishing_targets_file.parent.mkdir(parents=True, exist_ok=True)
        with open(phishing_targets_file, "w") as f:
            for target_name in phishing_targets:
                f.write(f"{target_name}\n")
        print(f"Created phishing/targets.txt with {len(phishing_targets)} targets")

    print("\n=== Summary ===")
    print(f"Processed {len(targets)} targets")
    print(f"Created {total_benign} benign symlinks")
    print(f"Created {total_phishing} phishing symlinks")
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

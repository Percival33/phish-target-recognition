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
    """Load target name mappings from target_mappings.json."""
    try:
        with open(Path(__file__).parent.parent / "target_mappings.json", "r") as f:
            mappings = json.load(f)

        # Create a flat mapping from all possible names to the key
        target_mappings = {}
        for key, names in mappings.items():
            # Add the key itself
            target_mappings[key.lower().replace(" ", "_")] = key
            # Add all alternative names
            for name in names:
                target_mappings[name.lower().replace(" ", "_")] = key

        return target_mappings
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load target_mappings.json: {e}")
        return {}


def sanitize_folder_name(name):
    """Sanitize target name for use as folder name."""
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip("_.")
    return sanitized


def save_visualphish_format(
    target_name, target_num, benign_images, phishing_images, output_dir
):
    """Save images in VisualPhish format using symlinks - trusted_list and phishing folders with target subfolders."""
    sanitized_name = sanitize_folder_name(target_name)

    # Create trusted_list and phishing top-level folders
    trusted_list_dir = Path(output_dir) / "trusted_list"
    phishing_dir = Path(output_dir) / "phishing"

    benign_count = 0
    phishing_count = 0

    # Save benign images in trusted_list/target_folder
    if benign_images:
        benign_target_folder = trusted_list_dir / sanitized_name
        benign_target_folder.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(benign_images):
            if Path(img_path).exists():
                link_name = f"T{target_num}_{i}.png"
                link_path = benign_target_folder / link_name

                # Remove existing symlink if it exists
                if link_path.exists():
                    link_path.unlink()

                os.symlink(str(Path(img_path).absolute()), str(link_path))
                benign_count += 1

    # Save phishing images in phishing/target_folder
    if phishing_images:
        phishing_target_folder = phishing_dir / sanitized_name
        phishing_target_folder.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(phishing_images):
            if Path(img_path).exists():
                link_name = f"T{target_num}_{i}.png"
                link_path = phishing_target_folder / link_name

                # Remove existing symlink if it exists
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

    # Try to map target name to known company names
    target_lower = target_name.lower().replace(" ", "_")

    # Create reverse mapping from company names to target keys
    reverse_mapping = {}
    for key, domain in domain_mapping.items():
        reverse_mapping[key] = key

    # Try different variations of the target name
    search_names = [
        target_name,
        target_mappings.get(target_lower, target_name),
        target_name.replace("_", " "),
        target_name.replace(" ", "_"),
    ]

    for search_name in search_names:
        # Find folders that start with the search name
        for folder in phishing_path.iterdir():
            if folder.is_dir() and folder.name.startswith(search_name + "+"):
                shot_png = folder / "shot.png"
                if shot_png.exists():
                    image_paths.append(str(shot_png))
                else:
                    print(f"Warning: No shot.png found in {folder}")

        if image_paths:  # Found some matches, stop searching
            break

    if not image_paths:
        print(f"Warning: No phishing folders found for target '{target_name}'")

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

    # Load JSON file and filter phish=true entries
    try:
        # If path is relative, make it relative to project root
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

    # Load domain mapping
    domain_mapping = load_domain_mapping()
    print(f"Loaded domain mapping with {len(domain_mapping)} entries")

    # Load target mappings
    target_mappings = load_target_mappings()
    print(f"Loaded target mappings with {len(target_mappings)} entries")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Process each target
    total_benign = 0
    total_phishing = 0
    benign_targets = []
    phishing_targets = []

    for i, target in enumerate(targets):
        target_name = target["target"]
        print(f"\nProcessing target {i}: {target_name}")

        # Find benign images by domain
        domains = target["domains"].split("\n") if target.get("domains") else []
        benign_images = find_benign_images(domains, args.benign_dir)

        # Find phishing images by target name
        phishing_images = find_phishing_images(
            target_name, args.phishing_dir, domain_mapping, target_mappings
        )

        # Save images in VisualPhish format
        if benign_images or phishing_images:
            benign_count, phishing_count = save_visualphish_format(
                target_name, i, benign_images, phishing_images, args.output_dir
            )
            total_benign += benign_count
            total_phishing += phishing_count

            # Track targets that have images in each category
            if benign_count > 0:
                benign_targets.append(target_name)
            if phishing_count > 0:
                phishing_targets.append(target_name)

            print(f"  → {benign_count} benign, {phishing_count} phishing images")
        else:
            print(f"  → No images found for {target_name}")

    # Create separate targets.txt files
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

#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
import sys
from urllib.parse import urlparse
from typing import List, Tuple, Optional, Dict, Any

def calculate_hash(file_path: Path) -> str:
    # TODO: Implement perceptual hash calculation for image file
    return None

def extract_fqdn(url: str) -> Optional[str]:
    """Extracts the FQDN from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return None

def find_image_files(image_folder: Path) -> List[Path]:
    """Finds and sorts image files in a folder."""
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    images = []
    for p in image_folder.iterdir():
        if p.is_file() and p.suffix.lower() in image_extensions:
            images.append(p)

    def sort_key(path: Path) -> int:
        # Extracts numbers from filename for robust sorting
        # e.g. "001.png" -> 1, "img_002.jpg" -> 2
        name_stem = path.stem
        numbers = re.findall(r'\d+', name_stem)
        if numbers:
            return int(numbers[-1]) # Use the last number found
        return -1 # Should not happen if filenames are numbered

    images.sort(key=sort_key)
    return images

def prepare_csv_data(
    image_folder_path: Path,
    labels_file_path: Path,
    output_csv_path: Path,
    default_is_phishing: bool,
) -> None:
    """
    Reads image folder and labels file (containing target names),
    then generates a CSV compatible with organize_by_sample.py.
    Screenshot paths are stored as relative filenames.
    URL and FQDN fields will be empty.
    """
    if not image_folder_path.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_folder_path}")
    if not labels_file_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_file_path}")

    with open(labels_file_path, "r") as f:
        # Each line in labels_file is now an affected_entity name
        target_names = [line.strip() for line in f if line.strip()]

    image_files = find_image_files(image_folder_path)

    if not image_files:
        print(f"Warning: No image files found in {image_folder_path}")
        if not target_names:
            print("Warning: No target names found in labels file either. Output CSV will be empty.")
        
    if len(target_names) != len(image_files):
        print(
            f"Error: Mismatch between number of target names ({len(target_names)}) and "
            f"image files ({len(image_files)}). Cannot proceed with mismatched data."
        )
        sys.exit(1)

    min_count = min(len(target_names), len(image_files))
    
    csv_data: List[Dict[str, Any]] = []

    for i in range(min_count):
        target_name = target_names[i]
        image_file = image_files[i]

        url = None  # URL is no longer sourced from labels.txt
        fqdn = None # FQDN is derived from URL, so also None
        
        screenshot_object = image_file.name # just the filename

        screenshot_hash = calculate_hash(image_file)
        affected_entity = target_name
        
        csv_data.append(
            {
                "url": url if url is not None else "",
                "fqdn": fqdn if fqdn is not None else "",
                "screenshot_object": screenshot_object,
                "screenshot_hash": screenshot_hash if screenshot_hash is not None else "",
                "affected_entity": affected_entity if affected_entity is not None else "",
                "is_phishing": default_is_phishing,
            }
        )
        
    if not csv_data and (target_names or image_files):
        print("Warning: No matching pairs of target names and images to process. Output CSV will only contain headers.")

    fieldnames = [
        "url",
        "fqdn",
        "screenshot_object",
        "screenshot_hash",
        "affected_entity",
        "is_phishing",
    ]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if csv_data:
            writer.writerows(csv_data)

    print(f"Successfully generated CSV: {output_csv_path} with {len(csv_data)} entries.")
    if len(csv_data) < len(target_names) or len(csv_data) < len(image_files):
        print(f"Note: {len(target_names) - len(csv_data)} target names and/or {len(image_files) - len(csv_data)} images were not processed due to mismatches.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a CSV file from an image folder and a labels file (target names) "
                    "for use with organize_by_sample.py."
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        required=True,
        help="Path to the folder containing image files (e.g., 00001.png, 00002.jpg).",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        required=True,
        help="Path to the text file where each line is a target name corresponding to an image.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to save the generated CSV file.",
    )
    parser.add_argument(
        "--is-phishing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flag to mark samples as phishing (default) or not (--no-is-phishing for legitimate).",
    )

    args = parser.parse_args()

    try:
        prepare_csv_data(
            args.image_folder,
            args.labels_file,
            args.output_csv,
            args.is_phishing,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 
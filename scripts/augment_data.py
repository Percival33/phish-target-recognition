#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
#     "pillow>=9.0.0",
# ]
# ///
"""
Data Augmentation Script for Phishing Target Recognition Dataset

This script augments datasets to meet minimum sample thresholds per target per class
using 4 core image augmentation techniques: blur, brightness, shift, and noise.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import random
from pathlib import Path
import argparse
from datetime import datetime

torch.manual_seed(42)
random.seed(42)

TO_TENSOR = transforms.ToTensor()
TO_PIL = transforms.ToPILImage()


def validate_inputs(args):
    """Essential validations only"""
    input_path = Path(args.input_folder)
    output_path = Path(args.output)

    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_path}")

    if args.threshold <= 0:
        raise ValueError(f"Threshold must be > 0, got: {args.threshold}")

    output_path.mkdir(parents=True, exist_ok=True)
    test_file = output_path / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        raise ValueError(f"Cannot write to output directory: {e}")


def count_samples(folder_path, target_name):
    """Count image files in target folder"""
    target_dir = folder_path / target_name
    if not target_dir.exists():
        return 0
    return len(
        [
            f
            for f in target_dir.glob("*")
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )


def get_targets_needing_augmentation(
    input_folder, threshold, phish_folder, benign_folder
):
    """Return dict of targets that need augmentation"""
    targets_to_augment = {}

    phish_dir = input_folder / phish_folder
    benign_dir = input_folder / benign_folder

    phish_targets = (
        {d.name for d in phish_dir.iterdir() if d.is_dir()}
        if phish_dir.exists()
        else set()
    )
    benign_targets = (
        {d.name for d in benign_dir.iterdir() if d.is_dir()}
        if benign_dir.exists()
        else set()
    )
    all_targets = phish_targets | benign_targets

    for target in all_targets:
        phish_count = count_samples(phish_dir, target) if target in phish_targets else 0
        benign_count = (
            count_samples(benign_dir, target) if target in benign_targets else 0
        )

        needs_phish = 0  # augumenting only benign
        needs_benign = max(0, threshold - benign_count)

        if needs_phish > 0 or needs_benign > 0:
            targets_to_augment[target] = {
                "phish_needed": needs_phish,
                "benign_needed": needs_benign,
                "phish_current": phish_count,
                "benign_current": benign_count,
            }

    return targets_to_augment


def apply_augmentation(pil_image, aug_type):
    """Apply single augmentation, return (augmented_pil, params_string)"""
    tensor_img = TO_TENSOR(pil_image)

    if aug_type == "blur":
        kernel_size = random.choice([3, 5, 7, 9, 11])
        augmented = transforms.GaussianBlur(kernel_size)(tensor_img)
        params = f"blur kernel={kernel_size}"

    elif aug_type == "brightness":
        factor = random.uniform(0.5, 1.8)
        augmented = transforms.ColorJitter(brightness=factor)(tensor_img)
        params = f"brightness factor={factor:.2f}"

    elif aug_type == "shift":
        translate = random.uniform(0.05, 0.15)
        augmented = transforms.RandomAffine(
            degrees=0, translate=(translate, translate)
        )(tensor_img)
        params = f"shift translate={translate:.2f}"

    elif aug_type == "noise":
        std = random.uniform(0.01, 0.05)
        noise = torch.randn_like(tensor_img) * std
        augmented = torch.clamp(tensor_img + noise, 0, 1)
        params = f"noise std={std:.3f}"

    return TO_PIL(augmented), params


def augment_target_class(
    source_dir, dest_dir, needed_count, log_file, target_name, class_name
):
    """Augment one target/class combination"""
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_files = [
        f for f in source_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    if not original_files:
        print(f"Warning: No images found in {source_dir}")
        return 0

    for orig_file in original_files:
        symlink_path = dest_dir / orig_file.name
        if not symlink_path.exists():
            try:
                symlink_path.symlink_to(orig_file.resolve())
            except Exception as e:
                print(f"Warning: Could not create symlink for {orig_file}: {e}")

    aug_counter = 1
    techniques = ["blur", "brightness", "shift", "noise"]
    augmented_count = 0

    for _ in range(needed_count):
        orig_file = random.choice(original_files)
        aug_type = random.choice(techniques)

        try:
            pil_image = Image.open(orig_file)
            # Convert RGBA to RGB if needed
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")
            augmented_pil, params = apply_augmentation(pil_image, aug_type)

            aug_filename = f"{orig_file.stem}_aug{aug_counter}{orig_file.suffix}"
            aug_path = dest_dir / aug_filename
            augmented_pil.save(aug_path)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | {target_name} | {class_name} | {orig_file.name} | {aug_filename} | {params}\n"

            with open(log_file, "a") as f:
                f.write(log_entry)

            aug_counter += 1
            augmented_count += 1

        except Exception as e:
            print(f"Error augmenting {orig_file}: {e}")
            continue

    return augmented_count


def print_summary(targets_to_augment, total_augmented):
    """Print simple completion summary"""
    print("\n=== Augmentation Complete ===")
    print(f"Targets processed: {len(targets_to_augment)}")
    print(f"Total augmented images created: {total_augmented}")
    print("Techniques used: blur, brightness, shift, noise")


def main():
    parser = argparse.ArgumentParser(
        description="Augment phishing target recognition dataset"
    )
    parser.add_argument("input_folder", help="Path to input folder containing dataset")
    parser.add_argument(
        "--label-strategy",
        default="directory",
        help="Label strategy (currently only directory supported)",
    )
    parser.add_argument(
        "--output", required=True, help="Output folder path for augmented dataset"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Minimum number of samples required per target per class",
    )
    parser.add_argument(
        "--phish-folder",
        default="phishing",
        help="Name of phishing subfolder (default: phishing)",
    )
    parser.add_argument(
        "--benign-folder",
        default="trusted_list",
        help="Name of benign subfolder (default: trusted_list)",
    )

    args = parser.parse_args()

    try:
        validate_inputs(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output)

    targets_to_augment = get_targets_needing_augmentation(
        input_folder, args.threshold, args.phish_folder, args.benign_folder
    )

    if not targets_to_augment:
        print(
            "No targets need augmentation. All targets meet the threshold requirements."
        )
        return 0

    print(f"Found {len(targets_to_augment)} targets that need augmentation:")
    for target, needs in targets_to_augment.items():
        print(
            f"  {target}: phish={needs['phish_current']}→{args.threshold} (+{needs['phish_needed']}), "
            f"benign={needs['benign_current']}→{args.threshold} (+{needs['benign_needed']})"
        )

    log_file = output_folder / "augmentation_log.txt"

    with open(log_file, "w") as f:
        f.write(
            "Timestamp | Target | Class | Original File | Augmented File | Parameters\n"
        )
        f.write("-" * 80 + "\n")

    total_augmented = 0

    for target_name, needs in targets_to_augment.items():
        print(f"\nProcessing target: {target_name}")

        if needs["phish_needed"] > 0:
            print(f"  Augmenting phishing class: {needs['phish_needed']} images needed")
            source_dir = input_folder / args.phish_folder / target_name
            dest_dir = output_folder / args.phish_folder / target_name

            augmented = augment_target_class(
                source_dir,
                dest_dir,
                needs["phish_needed"],
                log_file,
                target_name,
                "phishing",
            )
            total_augmented += augmented
            print(f"    Created {augmented} augmented images")

        if needs["benign_needed"] > 0:
            print(f"  Augmenting benign class: {needs['benign_needed']} images needed")
            source_dir = input_folder / args.benign_folder / target_name
            dest_dir = output_folder / args.benign_folder / target_name

            augmented = augment_target_class(
                source_dir,
                dest_dir,
                needs["benign_needed"],
                log_file,
                target_name,
                "benign",
            )
            total_augmented += augmented
            print(f"    Created {augmented} augmented images")

    print_summary(targets_to_augment, total_augmented)
    print(f"Log file created: {log_file}")

    return 0


if __name__ == "__main__":
    exit(main())

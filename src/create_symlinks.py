#!/usr/bin/env python3
"""
Script to create symlinks to all files in target folders.

Usage: python create_symlinks.py /path/to/source /path/to/symlink_dir [options]
"""

import os
import sys
import argparse


def create_symlinks_flat(source_dir, symlink_dir, prefix_folder=False, dry_run=False):
    """Create symlinks in a flat directory structure."""
    if not dry_run:
        os.makedirs(symlink_dir, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    symlink_count = 0

    # Get all target folders
    target_folders = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    for folder_name in sorted(target_folders):
        folder_path = os.path.join(source_dir, folder_name)

        # Get all image files in the folder
        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_exts
        ]

        for image_file in sorted(image_files):
            source_file = os.path.join(folder_path, image_file)

            # Determine symlink name
            if prefix_folder:
                symlink_name = f"{folder_name}_{image_file}"
            else:
                symlink_name = image_file

            symlink_path = os.path.join(symlink_dir, symlink_name)

            # Handle naming conflicts
            counter = 1
            original_symlink_name = symlink_name
            while os.path.exists(symlink_path) and not dry_run:
                name, ext = os.path.splitext(original_symlink_name)
                symlink_name = f"{name}_{counter}{ext}"
                symlink_path = os.path.join(symlink_dir, symlink_name)
                counter += 1

            if dry_run:
                print(
                    f"[DRY RUN] Would create symlink: {symlink_path} -> {source_file}"
                )
            else:
                try:
                    os.symlink(os.path.abspath(source_file), symlink_path)
                    print(f"Created symlink: {symlink_name} -> {source_file}")
                    symlink_count += 1
                except OSError as e:
                    print(f"Error creating symlink {symlink_path}: {e}")

    if not dry_run:
        print(f"\nCreated {symlink_count} symlinks in {symlink_dir}")
    else:
        print(f"\n[DRY RUN] Would create symlinks in {symlink_dir}")


def create_symlinks_structured(source_dir, symlink_dir, dry_run=False):
    """Create symlinks maintaining the original folder structure."""
    image_exts = {".jpg", ".jpeg", ".png"}
    symlink_count = 0

    # Get all target folders
    target_folders = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    for folder_name in sorted(target_folders):
        source_folder = os.path.join(source_dir, folder_name)
        symlink_folder = os.path.join(symlink_dir, folder_name)

        if not dry_run:
            os.makedirs(symlink_folder, exist_ok=True)

        # Get all image files in the folder
        image_files = [
            f
            for f in os.listdir(source_folder)
            if os.path.splitext(f)[1].lower() in image_exts
        ]

        for image_file in sorted(image_files):
            source_file = os.path.join(source_folder, image_file)
            symlink_path = os.path.join(symlink_folder, image_file)

            if dry_run:
                print(
                    f"[DRY RUN] Would create symlink: {symlink_path} -> {source_file}"
                )
            else:
                try:
                    if os.path.exists(symlink_path):
                        print(f"Skipping existing symlink: {symlink_path}")
                        continue

                    os.symlink(os.path.abspath(source_file), symlink_path)
                    print(
                        f"Created symlink: {folder_name}/{image_file} -> {source_file}"
                    )
                    symlink_count += 1
                except OSError as e:
                    print(f"Error creating symlink {symlink_path}: {e}")

    if not dry_run:
        print(f"\nCreated {symlink_count} symlinks in {symlink_dir}")
    else:
        print(f"\n[DRY RUN] Would create symlinks in {symlink_dir}")


def create_symlinks_numbered(source_dir, symlink_dir, dry_run=False):
    """Create symlinks with sequential numbering across all files."""
    if not dry_run:
        os.makedirs(symlink_dir, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    symlink_count = 0
    file_counter = 0

    # Get all target folders
    target_folders = [
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    ]

    for folder_name in sorted(target_folders):
        folder_path = os.path.join(source_dir, folder_name)

        # Get all image files in the folder
        image_files = [
            f
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_exts
        ]

        for image_file in sorted(image_files):
            source_file = os.path.join(folder_path, image_file)
            ext = os.path.splitext(image_file)[1]
            symlink_name = f"{file_counter:06d}{ext}"
            symlink_path = os.path.join(symlink_dir, symlink_name)

            if dry_run:
                print(
                    f"[DRY RUN] Would create symlink: {symlink_name} -> {source_file} (from {folder_name})"
                )
            else:
                try:
                    os.symlink(os.path.abspath(source_file), symlink_path)
                    print(
                        f"Created symlink: {symlink_name} -> {source_file} (from {folder_name})"
                    )
                    symlink_count += 1
                except OSError as e:
                    print(f"Error creating symlink {symlink_path}: {e}")

            file_counter += 1

    if not dry_run:
        print(f"\nCreated {symlink_count} symlinks in {symlink_dir}")
    else:
        print(f"\n[DRY RUN] Would create {file_counter} symlinks in {symlink_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create symlinks to all files in target folders."
    )
    parser.add_argument("source_dir", help="Source directory containing target folders")
    parser.add_argument("symlink_dir", help="Directory where symlinks will be created")
    parser.add_argument(
        "--mode",
        choices=["flat", "structured", "numbered"],
        default="flat",
        help="Symlink creation mode: flat (all in one dir), structured (maintain folders), numbered (sequential naming)",
    )
    parser.add_argument(
        "--prefix-folder",
        action="store_true",
        help="Prefix symlink names with folder name (only for flat mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating symlinks",
    )

    args = parser.parse_args()

    # Validate source directory
    if not os.path.exists(args.source_dir) or not os.path.isdir(args.source_dir):
        print(
            f"Error: Source directory '{args.source_dir}' does not exist or is not a directory."
        )
        sys.exit(1)

    # Create symlinks based on mode
    if args.mode == "flat":
        create_symlinks_flat(
            args.source_dir, args.symlink_dir, args.prefix_folder, args.dry_run
        )
    elif args.mode == "structured":
        create_symlinks_structured(args.source_dir, args.symlink_dir, args.dry_run)
    elif args.mode == "numbered":
        create_symlinks_numbered(args.source_dir, args.symlink_dir, args.dry_run)


if __name__ == "__main__":
    main()

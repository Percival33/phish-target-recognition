#!/usr/bin/env python3
"""
Fast script to find folders that don't contain either info.txt or shot.png
"""

import os
import sys
import argparse
import concurrent.futures


def check_folder(folder_path):
    """Check if a folder has both required files."""
    info_exists = os.path.isfile(os.path.join(folder_path, "info.txt"))
    shot_exists = os.path.isfile(os.path.join(folder_path, "shot.png"))
    # shot_exists = os.path.isfile(os.path.join(folder_path, "shot1.png"))
    # if shot_exists:
    #     return folder_path
    # return None
    # If either file is missing, return the folder path
    if not (info_exists and shot_exists):
        return folder_path
    return None


def find_missing_files(base_dir, max_workers=None, quiet=False):
    """
    Find all folders that don't contain both info.txt and shot.png.
    Uses a work queue and thread pool for maximum performance.
    """
    # Use all available cores unless specified
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        if not quiet:
            print(f"Error: '{base_dir}' is not a valid directory", file=sys.stderr)
        return []

    # Get all immediate subdirectories first
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Results list
    missing_files_folders = []

    # Process directories in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all directory checking tasks
        future_to_folder = {
            executor.submit(check_folder, folder): folder for folder in subdirs
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_folder):
            result = future.result()
            if result:
                missing_files_folders.append(result)

    # Sort results for consistent output
    missing_files_folders.sort()
    return missing_files_folders


def main():
    # Use argparse for better command-line argument handling
    parser = argparse.ArgumentParser(
        description="Find folders missing info.txt or shot.png files."
    )
    parser.add_argument("folder_path", help="Path to the base directory to scan")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all output except results"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: number of CPU cores)",
    )

    args = parser.parse_args()

    if not args.quiet:
        print(f"Scanning subfolders in '{args.folder_path}' for missing files...")

    # Find folders with missing files
    results = find_missing_files(
        args.folder_path, max_workers=args.max_workers, quiet=args.quiet
    )

    # Print results
    if results:
        if not args.quiet:
            print("\nFolders missing info.txt or shot.png:")
        for folder in results:
            print(folder)
        if not args.quiet:
            print(f"\nTotal: {len(results)} folders with missing files")
    elif not args.quiet:
        print("All folders contain both info.txt and shot.png files.")

    return 0


if __name__ == "__main__":
    import time

    start_time = time.time()
    exit_code = main()
    elapsed = time.time() - start_time

    # Only print execution time if not in quiet mode
    if len(sys.argv) > 1 and not ("-q" in sys.argv or "--quiet" in sys.argv):
        print(f"\nExecution time: {elapsed:.2f} seconds")

    sys.exit(exit_code)

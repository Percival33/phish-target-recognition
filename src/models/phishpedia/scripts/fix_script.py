#!/usr/bin/env python3
"""
Script to fix domain folders with missing files and handle multiple PNG cases.

Usage: python3 fix_domain_folders.py /path/to/main/folder list_of_error_folders.txt
"""

import os
import sys
import shutil
import glob


def create_info_txt(folder_path):
    """Create info.txt file with URL deduced from the folder name."""
    folder_name = os.path.basename(folder_path).split("--")[0]

    # Extract domain name
    domain = folder_name

    # Create URL (add https:// prefix)
    url = f"https://{domain}"

    # Create info.txt with the URL
    info_path = os.path.join(folder_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(url)

    print(f"Created info.txt with URL '{url}' in {folder_path}")
    return info_path


def handle_multiple_png(folder_path):
    """Handle folders with multiple PNG files."""
    # Get all PNG files
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    # If there's more than one PNG file
    if len(png_files) > 1:
        # Find shot1.png if it exists
        shot1_path = os.path.join(folder_path, "shot1.png")

        if os.path.exists(shot1_path):
            # Create new subfolder with --1 suffix
            folder_name = os.path.basename(folder_path)
            parent_dir = os.path.dirname(folder_path)
            new_folder_name = f"{folder_name}--1"
            new_folder_path = os.path.join(parent_dir, new_folder_name)

            # Create new folder if it doesn't exist
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Created new folder: {new_folder_path}")

            # Create info.txt in the new folder
            new_info_path = os.path.join(new_folder_path, "info.txt")
            url = f"https://{folder_name}"
            with open(new_info_path, "w") as f:
                f.write(url)

            # Move shot1.png to the new folder, renaming it to shot.png
            new_shot_path = os.path.join(new_folder_path, "shot.png")
            shutil.copy2(shot1_path, new_shot_path)
            print(f"Moved {shot1_path} to {new_shot_path}")

            # Remove other PNG files or process them further as needed
            return True
        else:
            print(f"Multiple PNG files found in {folder_path}, but no shot1.png")
            # Handle other PNG files - keep one and rename it to shot.png
            if len(png_files) > 0:
                # Keep the first PNG file and rename it to shot.png
                first_png = png_files[0]
                shot_png_path = os.path.join(folder_path, "shot.png")

                # If the file is not already named shot.png, rename it
                if os.path.basename(first_png) != "shot.png":
                    shutil.copy2(first_png, shot_png_path)
                    print(f"Renamed {first_png} to {shot_png_path}")

                # Remove or move other PNG files to separate folders as needed
                for png_file in png_files[1:]:
                    if os.path.basename(png_file) != "shot.png":
                        # Create additional folders for extra PNG files if needed
                        # or simply delete them if they're duplicates
                        # Uncomment the next line to delete extra PNGs
                        # os.remove(png_file)
                        # print(f"Removed extra PNG file: {png_file}")

                        # Alternative: Create new folders for each extra PNG
                        folder_name = os.path.basename(folder_path)
                        parent_dir = os.path.dirname(folder_path)
                        png_index = png_files.index(png_file) + 1
                        new_folder_name = f"{folder_name}--{png_index}"
                        new_folder_path = os.path.join(parent_dir, new_folder_name)

                        if not os.path.exists(new_folder_path):
                            os.makedirs(new_folder_path)
                            print(f"Created new folder: {new_folder_path}")

                        # Create info.txt in the new folder
                        new_info_path = os.path.join(new_folder_path, "info.txt")
                        url = f"https://{folder_name}"
                        with open(new_info_path, "w") as f:
                            f.write(url)

                        # Move the PNG to the new folder, renaming it to shot.png
                        new_shot_path = os.path.join(new_folder_path, "shot.png")
                        shutil.copy2(png_file, new_shot_path)
                        print(f"Moved {png_file} to {new_shot_path}")

    return False


def ensure_shot_png(folder_path):
    """Ensure there's exactly one PNG file named 'shot.png' in the folder."""
    # Get all PNG files
    png_files = (
        glob.glob(os.path.join(folder_path, "*.png"))
        + glob.glob(os.path.join(folder_path, "*.PNG"))
        + glob.glob(os.path.join(folder_path, "*.jpg"))
    )

    # If there are no PNG files, nothing to do
    if not png_files:
        print(f"No PNG files found in {folder_path}")
        return False

    # If there's exactly one PNG file but it's not named shot.png, rename it
    if len(png_files) == 1 and os.path.basename(png_files[0]) != "shot.png":
        shot_png_path = os.path.join(folder_path, "shot.png")
        shutil.copy2(png_files[0], shot_png_path)
        os.remove(png_files[0])
        print(f"Renamed {png_files[0]} to {shot_png_path}")
        return True

    return False


def process_folder(folder_path):
    """Process a single folder to fix missing files and handle multiple PNGs."""
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return False

    # Check if info.txt exists, create if missing
    info_path = os.path.join(folder_path, "info.txt")
    if not os.path.exists(info_path):
        create_info_txt(folder_path)

    # Handle multiple PNG files
    handle_multiple_png(folder_path)

    # Ensure single PNG is named shot.png
    ensure_shot_png(folder_path)

    return True


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <main_folder_path> <error_folders_list>")
        return 1

    main_folder = sys.argv[1]
    error_list_file = sys.argv[2]

    # Check if main folder exists
    if not os.path.isdir(main_folder):
        print(f"Error: Main folder '{main_folder}' does not exist")
        return 1

    # Check if error list file exists
    if not os.path.isfile(error_list_file):
        print(f"Error: Error list file '{error_list_file}' does not exist")
        return 1

    # Read error folders list
    with open(error_list_file, "r") as f:
        error_folders = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(error_folders)} folders with errors...")

    # Process each folder
    for folder in error_folders:
        # Handle both absolute paths and relative paths
        if os.path.isabs(folder):
            folder_path = folder
        else:
            folder_path = os.path.join(main_folder, folder)

        process_folder(folder_path)

    print("Processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
This script prepares sample dataset from VisualPhish/browsers/X dataset
"""
import os
import shutil
import csv
from src.config import PROJ_ROOT

def move_images(input_dir, output_dir):
    """
    Restructure the folder by moving PNG images to new subfolders.
    """
    os.makedirs(output_dir, exist_ok=True)

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        
        if os.path.isdir(folder_path):  # Check if it's a directory
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".PNG"):  # Check if it's a PNG file
                    # Extract the number from the file name
                    file_number = os.path.splitext(file_name)[0]
                    
                    # Create the new folder name
                    new_folder_name = f"{folder_name}-{file_number}"
                    new_folder_path = os.path.join(output_dir, new_folder_name)
                    
                    # Create the new folder
                    os.makedirs(new_folder_path, exist_ok=True)
                    
                    # Define the new file path
                    new_file_path = os.path.join(new_folder_path, "shot.png")
                    
                    # Copy the file to the new location
                    shutil.copy(os.path.join(folder_path, file_name), new_file_path)

    print("Folder restructuring complete!")

def move_url(input_folder, urls_file):
    """
    Match URLs from urls.txt to folders and create info.txt in each folder.
    """
    # Define a mapping of folder prefixes to URL keywords
    prefix_to_keyword = {
        "boa": "bankofamerica",
        "absa": "absa",
        "alibaba": "alibaba",
        "amazon": "amazon",
        "apple": "apple",
        "chase": "chase",
        "dropbox": "dropbox",
        "ebay": "ebay",
        "facebook": "facebook",
        "google": "google",
        "linkedin": "linkedin",
        "microsoft": "microsoft",
        "paypal": "paypal",
        "yahoo": "yahoo",
    }

    # Read URLs from urls.txt
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]  # Remove empty lines and whitespace

    # Create a mapping of folder prefixes to their respective folders
    folder_mapping = {}
    for folder_name in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, folder_name)):
            prefix = folder_name.rsplit('-', 1)[0]  # Extract the prefix (e.g., "boa" from "boa-1")
            if prefix not in folder_mapping:
                folder_mapping[prefix] = []
            folder_mapping[prefix].append(folder_name)

    # Sort folders for each prefix to ensure "-1", "-2", etc., are in order
    for key in folder_mapping:
        folder_mapping[key].sort()

    # Match URLs to folders and write info.txt
    url_assigned = set()  # Track used URLs
    for prefix, folders in folder_mapping.items():
        keyword = prefix_to_keyword.get(prefix, prefix)  # Get the corresponding keyword for the prefix
        for folder in folders:
            # Find the first unused URL that matches the keyword
            matched_url = None
            for url in urls:
                if url not in url_assigned and keyword in url:
                    matched_url = url
                    url_assigned.add(url)
                    break

            # If a URL is matched, write it to info.txt; otherwise, write "No URL available"
            folder_path = os.path.join(input_folder, folder)
            info_file_path = os.path.join(folder_path, "info.txt")
            with open(info_file_path, "w") as info_file:
                info_file.write(matched_url if matched_url else "No URL available")

    print("info.txt files created successfully!")

def create_csv(root_folder, csv_output_file):
    """
    Create a CSV file with columns: target, path, URL.
    """
    # Initialize the CSV data
    csv_data = [["target", "path", "URL"]]

    # Traverse the folder structure
    for folder_name in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Path to info.txt
            info_file_path = os.path.join(folder_path, "info.txt")
            shot_file_path = os.path.join(folder_path, "shot.png")

            # Read the URL from info.txt if it exists
            if os.path.exists(info_file_path):
                with open(info_file_path, "r") as info_file:
                    url = info_file.read().strip()  # Read and strip any whitespace
            else:
                url = "No URL available"

            # Prepare the target and path
            target = folder_name.rsplit('-', 1)[0]
            path = shot_file_path if os.path.exists(shot_file_path) else "No shot.png"

            # Append the data to the CSV list
            csv_data.append([target, path, url])

    # Write the CSV data to a file
    with open(csv_output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)

    print(f"CSV file created successfully: {csv_output_file}")

# Main execution
if __name__ == "__main__":
    print(os.getcwd())
    dataset_path = PROJ_ROOT / 'data' / 'raw' / 'VisualPhish'
    input_dir = dataset_path / 'browsers' 'chrome'
    output_dir = PROJ_ROOT / 'data' / 'processed' / 'miniDataset'
    urls_file = dataset_path / 'browsers' / 'urls.txt'
    csv_output_file = f"{output_dir}/output.csv"

    move_images(input_dir, output_dir)
    move_url(output_dir, urls_file)
    create_csv(output_dir, csv_output_file)

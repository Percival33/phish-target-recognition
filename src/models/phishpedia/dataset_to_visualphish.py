from pathlib import Path
import pandas as pd
import sys
import argparse


def get_unique_folder_names(directory, folder_type):
    unique_folders = set()
    for dir_path in Path(directory).iterdir():
        if dir_path.is_dir():
            folder_name = dir_path.name.split("+")[0].lower()
            unique_folders.add((folder_name, folder_type))
    return unique_folders


def main(dir1, dir2):
    unique_folders1 = get_unique_folder_names(dir1, 0)
    unique_folders2 = get_unique_folder_names(dir2, 1)

    union_folders = unique_folders1.union(unique_folders2)
    data = {
        "Folder Name": [folder[0] for folder in union_folders],
        "isPhish": [folder[1] for folder in union_folders],
        "Count in Dir1": [
            folder[0] in {f[0] for f in unique_folders1} for folder in union_folders
        ],
        "Count in Dir2": [
            folder[0] in {f[0] for f in unique_folders2} for folder in union_folders
        ],
    }

    df = pd.DataFrame(data)
    df = df.sort_values(by="Folder Name")
    filtered_df = df[(df["Count in Dir1"]) & (df["Count in Dir2"])]
    print(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare folders between benign and phishing directories"
    )
    parser.add_argument("benign_dir", help="Directory containing benign samples")
    parser.add_argument("phish_dir", help="Directory containing phishing samples")
    args = parser.parse_args()

    dir1 = args.benign_dir
    dir2 = args.phish_dir
    main(dir1, dir2)

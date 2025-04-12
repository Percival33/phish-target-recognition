from pathlib import Path
import pandas as pd


def get_unique_folder_names(directory, folder_type):
    unique_folders = set()
    for dir_path in Path(directory).iterdir():
        if dir_path.is_dir():
            folder_name = dir_path.name.split('+')[0].lower()
            unique_folders.add((folder_name, folder_type))
    return unique_folders


def main(dir1, dir2):
    unique_folders1 = get_unique_folder_names(dir1, 0)
    unique_folders2 = get_unique_folder_names(dir2, 1)

    union_folders = unique_folders1.union(unique_folders2)
    data = {
        'Folder Name':  [folder[0] for folder in union_folders],
        'isPhish': [folder[1] for folder in union_folders],
        'Count in Dir1': [folder[0] in {f[0] for f in unique_folders1} for folder in union_folders],
        'Count in Dir2': [folder[0] in {f[0] for f in unique_folders2} for folder in union_folders]
    }

    df = pd.DataFrame(data)
    df = df.sort_values(by='Folder Name')
    filtered_df = df[(df['Count in Dir1']) & (df['Count in Dir2'])]
    print(df)
    return df

if __name__ == "__main__":
    dir1 = '/Users/mjarczewski/Repositories/inz/data/raw/phishpedia/benign_sample_30k'
    dir2 = '/Users/mjarczewski/Repositories/inz/data/raw/phishpedia/phish_sample_30k'
    main(dir1, dir2)

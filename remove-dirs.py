#!/usr/bin/env python3

import shutil
from pathlib import Path

def delete_dirs(input_file="pp_unique_targets.txt", base_path="/home/phish-target-recognition/src/models/phishpedia/models/expand_targetlist"):
    base = Path(base_path)
    
    if not base.exists():
        print(f"Base path '{base_path}' not found!")
        return
    
    with open(input_file, 'r') as f:
        for line in f:
            dir_name = line.strip()
            if not dir_name:
                continue
                
            target = base / dir_name
            
            if target.exists() and target.is_dir():
                print(f"Deleting: {target}")
                shutil.rmtree(target)
            else:
                print(f"Not found: {target}")

if __name__ == "__main__":
    delete_dirs()
import os
import sys
import shutil
import re
import pandas as pd

def process_target(target_dir, target_name, inputDir, outputDir, csv_records):
    """Process a single target directory"""
    input_target = os.path.join(inputDir, target_dir)
    output_target = os.path.join(outputDir, target_name)
    
    if not os.path.exists(input_target) or not os.path.isdir(input_target):
        print(f"Skipping {target_name}: Input target folder '{input_target}' does not exist or is not a directory.")
        return
    
    os.makedirs(output_target, exist_ok=True)

    image_exts = {'.jpg', '.jpeg', '.png'}
    def is_image(fname):
        return os.path.isfile(os.path.join(input_target, fname)) and os.path.splitext(fname)[1].lower() in image_exts

    input_imgs = [f for f in os.listdir(input_target) if is_image(f)]
    if not input_imgs:
        print(f"No images found in {input_target}")
        return
        
    T_re = re.compile(r'^T(\d+)_\d+\.(?:jpg|jpeg|png)$', re.IGNORECASE)
    T_numbers = [int(m.group(1)) for fname in os.listdir(output_target) if (m := T_re.match(fname))]
    Tnum = max(T_numbers) if T_numbers else 0

    N_re = re.compile(rf'^T{Tnum}_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
    N_used = [int(m.group(1)) for fname in os.listdir(output_target) if (m := N_re.match(fname))]
    N_start = max(N_used)+1 if N_used else 0

    for idx, imgname in enumerate(input_imgs):
        ext = os.path.splitext(imgname)[1]
        destfname = f"T{Tnum}_{N_start+idx}{ext.lower()}"
        destpath = os.path.join(output_target, destfname)
        if os.path.exists(destpath):
            raise FileExistsError(f"Destination file already exists and cannot be overwritten: {destpath}")
        shutil.copy2(os.path.join(input_target, imgname), destpath)
        
        # Add record to CSV data
        csv_records.append({
            'target': target_name,
            'input_name': imgname,
            'output_name': destfname
        })

def main():
    if len(sys.argv) != 3:
        raise ValueError("Script requires exactly two arguments: inputDir and outputDir")
    inputDir, outputDir = sys.argv[1], sys.argv[2]
    
    if not os.path.exists(inputDir) or not os.path.isdir(inputDir):
        raise FileNotFoundError(f"Input directory '{inputDir}' does not exist or is not a directory.")
    
    # Setup CSV file
    csv_file = os.path.join(outputDir, "file_operations.csv")
    csv_records = []
    
    # Get all target directories in inputDir
    target_dirs = [d for d in os.listdir(inputDir) 
                   if os.path.isdir(os.path.join(inputDir, d))]
    
    if not target_dirs:
        print(f"No target directories found in {inputDir}")
        return
    
    # Process each target
    for target_dir in target_dirs[1:]:
        target_name = target_dir.split('.')[0]
        print(f"Processing target: {target_name} from directory: {target_dir}")
        process_target(target_dir, target_name, inputDir, outputDir, csv_records)
    
    # Save to CSV
    if csv_records:
        df = pd.DataFrame(csv_records)
        if os.path.exists(csv_file):
            # Append to existing CSV
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            # Create new CSV with headers
            df.to_csv(csv_file, index=False)
        print(f"Recorded {len(csv_records)} file operations in {csv_file}")
    else:
        print("No files were processed.")

if __name__ == "__main__":
    main()
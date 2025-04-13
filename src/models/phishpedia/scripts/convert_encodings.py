#!/usr/bin/env python3
"""
Fast and efficient text file encoding converter with progress visualization.
Usage: python3 convert_encodings.py /path/to/parent/folder
"""

import os
import sys
import time
import shutil
import chardet
import concurrent.futures
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # For progress bars

# Configuration
MAX_WORKERS = os.cpu_count() or 8  # Use all available CPU cores
CHUNK_SIZE = 1024 * 512  # 512KB chunks for chardet for better speed/accuracy balance


def setup_logging(base_dir):
    """Set up logging directories and files."""
    log_dir = Path(os.getcwd()) / "conversion_logs"
    log_dir.mkdir(exist_ok=True)

    logs = {
        "main": log_dir / "main_conversion.log",
        "failed": log_dir / "failed_conversions.log",
        "processed": log_dir / "processed_files.log",
    }

    # Initialize log files
    with open(logs["main"], "w") as f:
        f.write(
            f"=== Encoding Conversion Log {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        f.write(f"Processing folder: {base_dir}\n")

    with open(logs["failed"], "w") as f:
        f.write("=== Failed Conversions ===\n")

    with open(logs["processed"], "w") as f:
        f.write("=== Processed Files ===\n")

    return logs


def log_message(message, log_file):
    """Append a timestamped message to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def detect_encoding(file_path):
    """Detect file encoding using chardet with optimized reading strategy."""
    with open(file_path, "rb") as f:
        # Read just the beginning of the file for faster detection
        raw_data = f.read(CHUNK_SIZE)
        if len(raw_data) < CHUNK_SIZE:
            # Small file, no need for further reading
            result = chardet.detect(raw_data)
        else:
            # For larger files, check a chunk from the middle too
            f.seek(-min(CHUNK_SIZE, os.path.getsize(file_path) // 2), 1)
            middle_chunk = f.read(CHUNK_SIZE)
            # Combine results for better accuracy
            result1 = chardet.detect(raw_data)
            result2 = chardet.detect(middle_chunk)
            # Use the result with higher confidence
            result = (
                result1 if result1["confidence"] > result2["confidence"] else result2
            )

    encoding = result["encoding"].lower() if result["encoding"] else "unknown"
    return encoding


def convert_to_utf8(file_info):
    """Convert a file to UTF-8 encoding."""
    file_path, encoding = file_info

    # Skip if already UTF-8 or ASCII
    if encoding in ["utf-8", "ascii"]:
        return {"path": file_path, "status": "skipped", "reason": "already UTF-8"}

    # Skip if encoding is unknown
    if encoding == "unknown":
        return {"path": file_path, "status": "failed", "reason": "unknown encoding"}

    backup_path = f"{file_path}.backup"
    temp_path = f"{file_path}.temp"

    try:
        # Create backup
        shutil.copy2(file_path, backup_path)

        # Convert encoding
        with open(file_path, "rb") as source:
            content = source.read()

        try:
            # Try to decode with the detected encoding
            decoded_content = content.decode(encoding, errors="replace")

            # Write with UTF-8 encoding
            with open(temp_path, "w", encoding="utf-8") as target:
                target.write(decoded_content)

            # Replace original file with the converted one
            shutil.move(temp_path, file_path)

            # Remove backup if successful
            os.remove(backup_path)
            return {"path": file_path, "status": "success"}

        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            # Restore from backup
            if os.path.exists(backup_path):
                shutil.move(backup_path, file_path)

            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return {"path": file_path, "status": "failed", "reason": str(e)}

    except Exception as e:
        # Handle any other exceptions
        if os.path.exists(backup_path):
            shutil.move(backup_path, file_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {"path": file_path, "status": "failed", "reason": str(e)}


def find_info_files(base_dir):
    """Find all info.txt files in the directory."""
    return list(Path(base_dir).glob("**/info.txt"))


def main():
    # Check arguments
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder_path>")
        print(f"Example: {sys.argv[0]} /path/to/parent/folder")
        return 1

    folder_path = os.path.abspath(sys.argv[1])
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist")
        return 1

    # Setup logging
    logs = setup_logging(folder_path)

    # Find all info.txt files
    print("Scanning for info.txt files...")
    info_files = find_info_files(folder_path)
    total_files = len(info_files)
    print(f"Found {total_files} files to process")

    if total_files == 0:
        print("No files to process. Exiting.")
        return 0

    # Detect encoding for all files
    print("Detecting file encodings...")
    file_encodings = []
    with tqdm(total=total_files, desc="Detecting encodings") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for file_path in info_files:

                def detect_and_update(file_path):
                    encoding = detect_encoding(file_path)
                    pbar.update(1)
                    return (file_path, encoding)

                future = executor.submit(detect_and_update, file_path)
                file_encodings.append(future)

            # Get results
            file_encodings = [
                future.result()
                for future in concurrent.futures.as_completed(file_encodings)
            ]

    # Convert files to UTF-8
    print("Converting files to UTF-8...")
    results = {"success": 0, "skipped": 0, "failed": 0}
    failed_files = []

    with tqdm(total=len(file_encodings), desc="Converting files") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(convert_to_utf8, file_info): file_info
                for file_info in file_encodings
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                status = result["status"]

                # Log the result
                if status == "success":
                    results["success"] += 1
                    with open(logs["processed"], "a") as f:
                        f.write(f"{result['path']}\n")
                elif status == "skipped":
                    results["skipped"] += 1
                    with open(logs["processed"], "a") as f:
                        f.write(f"{result['path']} (skipped: {result['reason']})\n")
                else:  # failed
                    results["failed"] += 1
                    failed_files.append(result)
                    with open(logs["failed"], "a") as f:
                        f.write(
                            f"{result['path']} - {result.get('reason', 'unknown error')}\n"
                        )

                pbar.update(1)

    # Generate summary
    summary = f"""
=== Conversion Summary ===
Total files processed: {total_files}
Successfully converted: {results["success"]}
Skipped (already UTF-8): {results["skipped"]}
Failed conversions: {results["failed"]}
=========================
    """

    log_message(summary, logs["main"])
    print(summary)

    # Report failures
    if results["failed"] > 0:
        print(
            f"WARNING: {results['failed']} conversions failed. Check {logs['failed']} for details."
        )
        return 1
    else:
        print("All conversions completed successfully!")
        return 0


if __name__ == "__main__":
    # Time the execution
    start_time = time.time()
    exit_code = main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    sys.exit(exit_code)

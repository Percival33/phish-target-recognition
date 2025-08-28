#!/usr/bin/env python3
"""
Simple threshold optimization for baseline phishing detection.
Grid search over thresholds to find optimal F1 and MCC values.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging

# Add parent directories to path for tools import
sys.path.append(str(Path(__file__).parent.parent.parent))
from tools.metrics import calculate_metrics
from tools.config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_validation_data(val_csv_path: str) -> pd.DataFrame:
    """Load validation data from CSV file."""
    val_df = pd.read_csv(val_csv_path)
    logger.info(f"Loaded validation data: {len(val_df)} samples")
    logger.info(f"Columns: {val_df.columns.tolist()}")
    logger.info(f"Class distribution: {val_df['true_class'].value_counts().to_dict()}")
    return val_df

def create_labels_file_for_directory(val_df: pd.DataFrame, data_dir: str, output_labels_path: str):
    """Create a labels.txt file for a specific directory based on validation data."""
    # Create a mapping from new_path to true_target
    path_to_target = {}
    for _, row in val_df.iterrows():
        if row['new_path'].startswith(data_dir):
            # Extract the filename from the relative path
            filename = Path(row['new_path']).name
            path_to_target[filename] = row['true_target']
    
    # Get the actual image paths from the directory to match get_image_paths() order
    from common import get_image_paths
    images_dir = Path(f"/home/phish-target-recognition/data_splits/visualphish/phishpedia/data/val/{data_dir}")
    image_paths = get_image_paths(images_dir)
    
    # Create labels in the same order as get_image_paths()
    labels = []
    missing_count = 0
    for img_path in image_paths:
        filename = img_path.name
        if filename in path_to_target:
            labels.append(path_to_target[filename])
        else:
            # Fallback to directory-based naming if not found in CSV
            labels.append(img_path.parent.name.split("+")[0])
            missing_count += 1
    
    # Create labels.txt file
    with open(output_labels_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    logger.info(f"Created labels file for {data_dir}: {len(labels)} images ({missing_count} fallback labels)")
    return len(labels)

def run_query_with_threshold(threshold: float, val_csv_path: str) -> bool:
    """Run query.py with specific threshold on validation data."""
    val_data_base = "/home/phish-target-recognition/data_splits/visualphish/phishpedia/data/val"
    phish_data_path = f"{val_data_base}/phishing"
    benign_data_path = f"{val_data_base}/trusted_list"
    index_path = "/home/phish-target-recognition/logs/vp/vp-for-baseline/index.faiss"
    
    phish_output = f"phish_results_threshold_{threshold}.csv"
    benign_output = f"benign_results_threshold_{threshold}.csv"
    combined_output = f"results_threshold_{threshold}.csv"
    
    # Load validation data
    val_df = load_validation_data(val_csv_path)
    
    # Create temporary labels files
    phish_labels_path = f"phish_labels_threshold_{threshold}.txt"
    benign_labels_path = f"benign_labels_threshold_{threshold}.txt"
    
    # Create labels files for each directory
    phish_count = create_labels_file_for_directory(val_df, "phishing", phish_labels_path)
    benign_count = create_labels_file_for_directory(val_df, "trusted_list", benign_labels_path)
    
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    
    try:
        print(f"Processing phishing samples...")
        # Run query on phishing samples with proper labels
        phish_cmd = [
            "uv", "run", "query.py",
            "--images", phish_data_path,
            "--index", index_path,
            "--output", phish_output,
            "--threshold", str(threshold),
            "--labels", phish_labels_path,
            "--is-phish",
            "--overwrite"
        ]
        
        result = subprocess.run(phish_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.error(f"Phishing query failed for threshold {threshold}: {result.stderr}")
            return False
        
        print(f"  🔍 Processing benign samples...")
        # Run query on trusted_list samples with proper labels
        benign_cmd = [
            "uv", "run", "query.py",
            "--images", benign_data_path,
            "--index", index_path,
            "--output", benign_output,
            "--threshold", str(threshold),
            "--labels", benign_labels_path,
            "--overwrite"
        ]
        
        result = subprocess.run(benign_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.error(f"Benign query failed for threshold {threshold}: {result.stderr}")
            return False
        
        print(f"Combining results...")
        # Combine results
        phish_df = pd.read_csv(phish_output)
        benign_df = pd.read_csv(benign_output)
        combined_df = pd.concat([phish_df, benign_df], ignore_index=True)
        combined_df.to_csv(combined_output, index=False)
        
        # Clean up temporary files
        Path(phish_output).unlink(missing_ok=True)
        Path(benign_output).unlink(missing_ok=True)
        Path(phish_labels_path).unlink(missing_ok=True)
        Path(benign_labels_path).unlink(missing_ok=True)
        
        logger.info(f"Successfully ran query for threshold {threshold} ({len(phish_df)} phish + {len(benign_df)} benign)")
        return True
        
    except Exception as e:
        logger.error(f"Error running query for threshold {threshold}: {e}")
        # Clean up temporary files on error
        Path(phish_labels_path).unlink(missing_ok=True)
        Path(benign_labels_path).unlink(missing_ok=True)
        return False

def calculate_metrics_from_results(results_file: str) -> tuple:
    """Calculate F1 and MCC from query results, treating benign as 'benign' target."""
    df = pd.read_csv(results_file)
    
    cls_true = df['true_class'].values
    cls_pred = df['baseline_class'].values
    targets_true = df['true_target'].values.copy()
    targets_pred = df['baseline_target'].values.copy()
    
    # Treat benign samples as 'benign' target
    targets_true = np.where(cls_true == 0, 'benign', targets_true)
    targets_pred = np.where(cls_pred == 0, 'benign', targets_pred)
    
    class_metrics, target_metrics = calculate_metrics(cls_true, cls_pred, targets_true, targets_pred)
    
    logger.info(f"Class F1: {class_metrics['f1_weighted']:.4f}, Class MCC: {class_metrics['mcc']:.4f}")
    logger.info(f"Target MCC: {target_metrics['target_mcc']:.4f}")
    
    return (class_metrics['f1_weighted'], class_metrics['mcc'], target_metrics['target_mcc'])

def main():
    """Main threshold optimization function."""
    thresholds = [30, 50, 70, 90, 110, 130, 150, 170, 200]
    results = []
    best_f1 = 0
    best_mcc = -1
    best_target_mcc = -1
    best_threshold_f1 = None
    best_threshold_mcc = None
    best_threshold_target_mcc = None
    
    # Path to validation CSV
    val_csv_path = "/home/phish-target-recognition/data_splits/visualphish/phishpedia/val.csv"
    
    print(f"Starting threshold optimization with {len(thresholds)} thresholds: {thresholds}")
    logger.info(f"Starting threshold optimization with thresholds: {thresholds}")
    logger.info(f"Using validation CSV: {val_csv_path}")
    
    # Change to baseline directory
    baseline_dir = Path(__file__).parent
    os.chdir(baseline_dir)
    logger.info(f"Working directory: {baseline_dir}")
    print(f"Working directory: {baseline_dir}")
    
    for i, threshold in enumerate(thresholds, 1):
        print(f"\n[{i}/{len(thresholds)}] > Testing threshold: {threshold}")
        logger.info(f"Testing threshold: {threshold}")
        
        if run_query_with_threshold(threshold, val_csv_path):
            output_file = f"results_threshold_{threshold}.csv"
            f1, mcc, target_mcc = calculate_metrics_from_results(output_file)
            
            results.append({
                'threshold': threshold,
                'f1_weighted': f1,
                'mcc': mcc,
                'target_mcc': target_mcc
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = threshold
                
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold_mcc = threshold
                
            if target_mcc > best_target_mcc:
                best_target_mcc = target_mcc
                best_threshold_target_mcc = threshold
            
            print(f"Results: F1={f1:.4f}, MCC={mcc:.4f}, Target MCC={target_mcc:.4f}")
            logger.info(f"Threshold {threshold}: F1={f1:.4f}, MCC={mcc:.4f}, Target MCC={target_mcc:.4f}")
            
            # Clean up temporary file
            Path(output_file).unlink(missing_ok=True)
        else:
            print(f"Failed to process threshold {threshold}")
            logger.error(f"Skipping threshold {threshold} due to query failure")
    
    # Save results
    if results:
        # Save results summary
        results_df = pd.DataFrame(results)
        results_df.to_csv("results_summary.csv", index=False)
        logger.info("Saved results_summary.csv")
        
        # Save best thresholds
        Path("best_threshold_f1.txt").write_text(str(best_threshold_f1))
        Path("best_threshold_mcc.txt").write_text(str(best_threshold_mcc))
        Path("best_threshold_target_mcc.txt").write_text(str(best_threshold_target_mcc))
        
        logger.info(f"Best F1 threshold: {best_threshold_f1}")
        logger.info(f"Best MCC threshold: {best_threshold_mcc}")
        logger.info(f"Best Target MCC threshold: {best_threshold_target_mcc}")
        
        print(f"\nOPTIMIZATION COMPLETE!")
        print(f"Best F1: {best_f1:.4f} at threshold {best_threshold_f1}")
        print(f"Best MCC: {best_mcc:.4f} at threshold {best_threshold_mcc}")
        print(f"Best Target MCC: {best_target_mcc:.4f} at threshold {best_threshold_target_mcc}")
        print(f"Results saved to results_summary.csv")
    else:
        logger.error("No successful results obtained")
        sys.exit(1)

if __name__ == "__main__":
    main()

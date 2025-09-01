#!/usr/bin/env python3
"""
Script to parse threshold results CSV and find best target_f1_macro scores.
"""

import pandas as pd
import ast
import sys
from typing import List, Tuple


def parse_metrics_string(metrics_str: str) -> dict:
    """Parse metrics string representation to dictionary."""
    try:
        # Handle numpy float64 types in the string
        metrics_str = metrics_str.replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(metrics_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing metrics string: {metrics_str}")
        print(f"Error: {e}")
        return {}


def find_best_thresholds(csv_file: str, top_n: int = 3) -> List[Tuple[float, float]]:
    """
    Find the best N thresholds based on target_f1_macro score.
    
    Args:
        csv_file: Path to CSV file
        top_n: Number of top results to return
        
    Returns:
        List of tuples (threshold, target_f1_macro_score)
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Parse target_metrics and extract target_f1_macro
    results = []
    
    for idx, row in df.iterrows():
        threshold = row['threshold']
        target_metrics_str = row['target_metrics']
        
        # Parse the target_metrics string
        target_metrics = parse_metrics_string(target_metrics_str)
        
        if 'target_f1_macro' in target_metrics:
            target_f1_macro = target_metrics['target_f1_macro']
            results.append((threshold, target_f1_macro))
        else:
            print(f"Warning: target_f1_macro not found in row {idx}")
    
    # Sort by target_f1_macro in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N results
    return results[:top_n]


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python parse_threshold_results.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        best_thresholds = find_best_thresholds(csv_file, top_n=3)
        
        print("\nTop 3 thresholds with best target_f1_macro scores:")
        print("-" * 50)
        
        for i, (threshold, score) in enumerate(best_thresholds, 1):
            print(f"{i}. Threshold: {threshold:8.4f} | target_f1_macro: {score:.6f}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

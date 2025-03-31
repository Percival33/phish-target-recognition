import csv
import os
import argparse
import pandas as pd
from dataclasses import dataclass


@dataclass
class ImageSimilarityRecord:
    """Data class to hold image similarity information"""
    filename: str
    is_benign: bool  # True if class is [0.], False if class is [1.]
    distance: float
    closest_image: str


def parse_to_dataframe(file_path):
    """
    Parse the custom CSV file and return a pandas DataFrame with columns:
    file, vp_class, vp_distance, vp_target, true_class

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pandas.DataFrame: DataFrame with the parsed data
    """
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            # Handle empty rows
            if not row:
                continue

            # Extract the values from the row
            filename = row[0]
            vp_class_raw = row[1]  # This will be [0.] or [1.]

            # Convert class to integer: 0 for benign, 1 for malicious
            vp_class = 0 if vp_class_raw == "[0.]" else 1

            # Convert distance to float
            vp_distance = float(row[2])

            # The closest image might contain spaces, so join remaining elements
            vp_target = " ".join(row[3:]) if len(row) > 3 else ""
            vp_target = vp_target.split("/")[0]

            # Extract true class from filename
            # Assuming filename starts with either "benign/" or something else
            true_class = 0 if filename.startswith("benign/") else 1

            data.append({
                "file": filename,
                "vp_class": vp_class,
                "vp_distance": vp_distance,
                "vp_target": vp_target,
                "true_class": true_class,
                "true_target": filename.split("/")[0],
            })

    return pd.DataFrame(data)


def analyze_records(records):
    """
    Perform basic analysis on the parsed records

    Args:
        records (list): List of ImageSimilarityRecord objects

    Returns:
        dict: Analysis results
    """
    total = len(records)
    benign_count = sum(1 for r in records if r.is_benign)
    malicious_count = total - benign_count

    # Calculate average distances
    if benign_count > 0:
        avg_benign_distance = sum(r.distance for r in records if r.is_benign) / benign_count
    else:
        avg_benign_distance = 0

    if malicious_count > 0:
        avg_malicious_distance = sum(r.distance for r in records if not r.is_benign) / malicious_count
    else:
        avg_malicious_distance = 0

    return {
        "total_records": total,
        "benign_count": benign_count,
        "malicious_count": malicious_count,
        "avg_benign_distance": avg_benign_distance,
        "avg_malicious_distance": avg_malicious_distance
    }


def analyze_dataframe(df):
    """
    Perform analysis on the pandas DataFrame

    Args:
        df (pandas.DataFrame): DataFrame with the parsed data

    Returns:
        dict: Analysis results
    """
    total = len(df)
    benign_pred_count = sum(df['vp_class'] == 0)
    malicious_pred_count = sum(df['vp_class'] == 1)

    benign_true_count = sum(df['true_class'] == 0)
    malicious_true_count = sum(df['true_class'] == 1)

    # Calculate accuracy
    accuracy = sum(df['vp_class'] == df['true_class']) / total

    # Calculate average distances
    avg_benign_distance = df[df['vp_class'] == 0]['vp_distance'].mean()
    avg_malicious_distance = df[df['vp_class'] == 1]['vp_distance'].mean()

    return {
        "total_records": total,
        "benign_predicted": benign_pred_count,
        "malicious_predicted": malicious_pred_count,
        "benign_actual": benign_true_count,
        "malicious_actual": malicious_true_count,
        "accuracy": accuracy,
        "avg_benign_distance": avg_benign_distance,
        "avg_malicious_distance": avg_malicious_distance
    }


def main():
    parser = argparse.ArgumentParser(description="Parse and analyze image similarity CSV data")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("--output", help="Output file for processed data (optional)")
    parser.add_argument("--pandas-output", help="Output file for pandas DataFrame (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist")
        return

    # Parse with both methods
    df = parse_to_dataframe(args.csv_file)

    # Display sample DataFrame rows
    print("\nSample DataFrame rows:")
    print(df.head(3))

    df_analysis = analyze_dataframe(df)
    print("\nAnalysis from DataFrame:")
    print(f"Total records: {df_analysis['total_records']}")
    print(
        f"Predicted benign: {df_analysis['benign_predicted']} ({df_analysis['benign_predicted'] / df_analysis['total_records'] * 100:.2f}%)")
    print(
        f"Predicted malicious: {df_analysis['malicious_predicted']} ({df_analysis['malicious_predicted'] / df_analysis['total_records'] * 100:.2f}%)")
    print(
        f"Actual benign: {df_analysis['benign_actual']} ({df_analysis['benign_actual'] / df_analysis['total_records'] * 100:.2f}%)")
    print(
        f"Actual malicious: {df_analysis['malicious_actual']} ({df_analysis['malicious_actual'] / df_analysis['total_records'] * 100:.2f}%)")
    print(f"Accuracy: {df_analysis['accuracy'] * 100:.2f}%")
    print(f"Average distance for benign predictions: {df_analysis['avg_benign_distance']:.4f}")
    print(f"Average distance for malicious predictions: {df_analysis['avg_malicious_distance']:.4f}")

    if args.pandas_output:
        df.to_csv(args.pandas_output, index=False)
        print(f"\nPandas DataFrame saved to {args.pandas_output}")


if __name__ == "__main__":
    main()

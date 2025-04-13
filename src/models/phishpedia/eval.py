import pandas as pd
import argparse
import os
import io


def parse_phishpedia_csv(file_path, is_phish=False):
    """
    Parse Phishpedia CSV file into a pandas DataFrame with specific columns.
    Handles Unicode encoding properly.

    Args:
        file_path (str): Path to the CSV file
        is_phish (bool): Whether to set true_class to 1 (True) or 0 (False)

    Returns:
        pandas.DataFrame: DataFrame with the parsed data
    """
    # Set true_class based on is_phish argument
    true_class_value = 1 if is_phish else 0

    # Try different encoding approaches
    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            # Read the CSV file
            df_raw = pd.read_csv(
                file_path,
                sep="\t",
                names=[
                    "folder",
                    "url",
                    "phish_category",
                    "pred_target",
                    "matched_domain",
                    "siamese_conf",
                    "logo_recog_time",
                    "logo_match_time",
                ],
                encoding=encoding,
                # on_bad_lines="skip",  # Skip problematic lines
            )

            # Handle any Unicode conversion for string columns
            string_columns = ["folder", "url", "pred_target", "matched_domain"]
            for col in string_columns:
                if col in df_raw.columns:
                    # Convert to unicode strings properly
                    df_raw[col] = df_raw[col].apply(
                        lambda x: x.decode(encoding).encode("utf-8")
                        if isinstance(x, bytes)
                        else (x if isinstance(x, str) else str(x))
                    )

            # Create a new DataFrame with the required columns
            df_processed = pd.DataFrame(
                {
                    "file": df_raw["folder"],
                    "pp_class": df_raw["phish_category"],
                    "pp_target": df_raw["pred_target"],
                    "url": df_raw["url"],
                    "true_class": true_class_value,  # Set based on is_phish argument
                    "true_target": df_raw["folder"].str.split("+").str[0]
                    if is_phish
                    else "benign",
                    "pp_conf": df_raw["siamese_conf"],
                }
            )

            print(f"Successfully parsed with {encoding} encoding")
            print(
                f"Data classified as {'phishing' if is_phish else 'non-phishing'} (true_class={true_class_value})"
            )
            return df_processed

        except Exception as e:
            print(f"Failed with {encoding} encoding: {str(e)}")

    # Last resort: use open() with encoding handling and then pass to pandas
    try:
        print("Trying with explicit file opening and error handling...")
        with io.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Create a StringIO object to be read by pandas
        from io import StringIO

        df_raw = pd.read_csv(
            StringIO(content),
            sep="\t",
            names=[
                "folder",
                "url",
                "phish_category",
                "pred_target",
                "matched_domain",
                "siamese_conf",
                "logo_recog_time",
                "logo_match_time",
            ],
            on_bad_lines="skip",
        )

        # Create a new DataFrame with the required columns
        df_processed = pd.DataFrame(
            {
                "file": df_raw["folder"],
                "pp_class": df_raw["phish_category"],
                "pp_target": df_raw["pred_target"],
                "url": df_raw["url"],
                "true_class": true_class_value,  # Set based on is_phish argument
                "true_target": df_raw["folder"].split("+")[0] if is_phish else "benign",
                "pp_conf": df_raw["siamese_conf"],
            }
        )

        print("Successfully parsed with explicit encoding handling")
        print(
            f"Data classified as {'phishing' if is_phish else 'non-phishing'} (true_class={true_class_value})"
        )
        return df_processed

    except Exception as e:
        raise Exception(f"All encoding methods failed. Last error: {str(e)}")


def analyze_dataframe(df):
    """
    Perform basic analysis on the processed DataFrame

    Args:
        df (pandas.DataFrame): DataFrame with the parsed data

    Returns:
        dict: Analysis results
    """
    total = len(df)

    # Count phishing vs non-phishing classifications
    phishing_count = sum(df["pp_class"] == 1)
    non_phishing_count = total - phishing_count

    # Count true class distribution
    true_phish_count = sum(df["true_class"] == 1)
    true_non_phish_count = sum(df["true_class"] == 0)

    # Calculate average confidence
    avg_confidence = df["pp_conf"].mean()
    avg_phishing_confidence = (
        df[df["pp_class"] == 1]["pp_conf"].mean() if phishing_count > 0 else 0
    )
    avg_non_phishing_confidence = (
        df[df["pp_class"] == 0]["pp_conf"].mean() if non_phishing_count > 0 else 0
    )

    # Get unique target brands
    unique_targets = df["pp_target"].nunique()
    top_targets = df["pp_target"].value_counts().head(5)

    return {
        "total_records": total,
        "phishing_count": phishing_count,
        "non_phishing_count": non_phishing_count,
        "true_phish_count": true_phish_count,
        "true_non_phish_count": true_non_phish_count,
        "avg_confidence": avg_confidence,
        "avg_phishing_confidence": avg_phishing_confidence,
        "avg_non_phishing_confidence": avg_non_phishing_confidence,
        "unique_targets": unique_targets,
        "top_targets": top_targets,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse Phishpedia CSV to pandas DataFrame"
    )
    parser.add_argument("csv_file", help="Path to the Phishpedia CSV file")
    parser.add_argument(
        "--output", help="Output file for processed DataFrame (optional)"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Perform analysis on the data"
    )
    parser.add_argument(
        "--encoding",
        help="Force a specific encoding (optional, will auto-detect if not specified)",
        default=None,
    )
    parser.add_argument(
        "--is-phish",
        action="store_true",
        help="Set true_class to 1 (phishing), default is 0 (non-phishing)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist")
        return

    try:
        # Parse the CSV file with is_phish argument
        df = parse_phishpedia_csv(args.csv_file, args.is_phish)
        print(f"Successfully parsed {len(df)} records")

        # Display sample rows
        print("\nSample rows:")
        print(df.head(5))

        # Perform analysis if requested
        if args.analyze:
            analysis = analyze_dataframe(df)
            print("\nAnalysis:")
            print(f"Total records: {analysis['total_records']}")
            print(
                f"Phishing sites (predicted): {analysis['phishing_count']} ({analysis['phishing_count'] / analysis['total_records'] * 100:.2f}%)"
            )
            print(
                f"Non-phishing sites (predicted): {analysis['non_phishing_count']} ({analysis['non_phishing_count'] / analysis['total_records'] * 100:.2f}%)"
            )
            print(
                f"Phishing sites (true class): {analysis['true_phish_count']} ({analysis['true_phish_count'] / analysis['total_records'] * 100:.2f}%)"
            )
            print(
                f"Non-phishing sites (true class): {analysis['true_non_phish_count']} ({analysis['true_non_phish_count'] / analysis['total_records'] * 100:.2f}%)"
            )
            print(f"Average confidence: {analysis['avg_confidence']:.4f}")
            print(
                f"Average phishing confidence: {analysis['avg_phishing_confidence']:.4f}"
            )
            print(
                f"Average non-phishing confidence: {analysis['avg_non_phishing_confidence']:.4f}"
            )
            print(f"Unique target brands: {analysis['unique_targets']}")
            print("\nTop 5 targeted brands:")
            for brand, count in analysis["top_targets"].items():
                print(f"  {brand}: {count}")

        # Save DataFrame if output file is specified
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nProcessed DataFrame saved to {args.output}")

    except Exception as e:
        print(f"Error processing CSV file: {e}")


if __name__ == "__main__":
    main()

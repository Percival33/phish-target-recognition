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
    df_raw = None

    def create_processed_df(raw_df):
        return pd.DataFrame(
            {
                "file": raw_df["folder"],
                "pp_class": raw_df["phish_category"],
                "pp_target": raw_df["pred_target"],
                "url": raw_df["url"],
                "true_class": true_class_value,  # Set based on is_phish argument
                "true_target": raw_df["folder"].str.split("+").str[0]
                if is_phish
                else "benign",
                "pp_conf": raw_df["siamese_conf"],
            }
        )

    column_names = [
        "folder",
        "url",
        "phish_category",
        "pred_target",
        "matched_domain",
        "siamese_conf",
        "logo_recog_time",
        "logo_match_time",
    ]

    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            df_raw = pd.read_csv(
                file_path,
                sep="\t",
                names=column_names,
                encoding=encoding,
                on_bad_lines="skip",
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
            df_processed = create_processed_df(df_raw)

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
            names=column_names,
            on_bad_lines="skip",
        )

        df_processed = create_processed_df(df_raw)

        print("Successfully parsed with explicit encoding handling")
        print(
            f"Data classified as {'phishing' if is_phish else 'non-phishing'} (true_class={true_class_value})"
        )
        return df_processed

    except Exception as e:
        raise Exception(f"All encoding methods failed. Last error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Phishpedia CSV to pandas DataFrame"
    )
    parser.add_argument("csv_file", help="Path to the Phishpedia CSV file")
    parser.add_argument(
        "--output", help="Output file for processed DataFrame (optional)"
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
        exit(1)

    try:
        # Parse the CSV file with is_phish argument
        df = parse_phishpedia_csv(args.csv_file, args.is_phish)
        print(f"Successfully parsed {len(df)} records")

        # Display sample rows
        print("\nSample rows:")
        print(df.head(5))

        # Save DataFrame if output file is specified
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nProcessed DataFrame saved to {args.output}")

    except Exception as e:
        print(f"Error processing CSV file: {e}")

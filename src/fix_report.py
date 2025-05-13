import argparse
import re

import pandas as pd


def fix_report(input_file: str, output_file: str = None):
    """
    Fix a CSV report by deduplicating columns that have the same base name but with .1 suffix.
    For each pair of such columns, values are merged by taking the non-null value.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file. If None, will modify the input file in place.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Get all column names
    columns = df.columns.tolist()

    # Find duplicate columns (those ending with .1)
    duplicate_cols = [col for col in columns if col.endswith(".1")]

    # For each duplicate column, merge it with its original counterpart
    for dup_col in duplicate_cols:
        # Get original column name by removing .1
        orig_col = re.sub(r"\.1$", "", dup_col)

        if orig_col in columns:
            # Merge the columns by taking non-null values from either column
            df[orig_col] = df[orig_col].combine_first(df[dup_col])
            # Drop the duplicate column
            df = df.drop(columns=[dup_col])

    # Save the result
    output_path = output_file if output_file else input_file
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fix CSV report by deduplicating columns"
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "--output-file",
        "-o",
        help="Path to output CSV file. If not specified, will modify input file in place",
    )

    args = parser.parse_args()
    fix_report(args.input_file, args.output_file)


if __name__ == "__main__":
    main()

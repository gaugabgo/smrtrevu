import pandas as pd
from pathlib import Path

def merge_csv_files(csv_file_paths, merged_csv_path):
    """
    Merge multiple CSV files with only id, doi, title, and abstract columns.

    Parameters:
    - csv_file_paths: list of str, paths to CSV files to merge
    - merged_csv_path: str, output path for the merged CSV

    Returns:
    - None (writes merged CSV to merged_csv_path)
    """
    if not csv_file_paths:
        raise ValueError("No CSV file paths provided.")

    # Define the columns we want to keep
    columns_to_keep = ['id', 'doi', 'title', 'language', 'abstract']

    print("\n🔍 Merging CSV files with selected columns: {', '.join(columns_to_keep)}...")

    # Read all CSV files and select only the specified columns
    dataframes = []
    for path in csv_file_paths:
        df = pd.read_csv(path, header=0, usecols=columns_to_keep)
        dataframes.append(df)

    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save merged CSV
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"✅ Merged CSV saved to: {merged_csv_path}")
    print(f"📋 Total rows: {len(merged_df)}")
    print(f"📋 Columns included: {', '.join(columns_to_keep)}")

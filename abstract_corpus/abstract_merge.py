import pandas as pd

def merge_csv_files(csv_file_paths, merged_csv_path):
    """
    Merge multiple CSV files ensuring headers match the first CSV's headers.

    Parameters:
    - csv_file_paths: list of str, paths to CSV files to merge
    - merged_csv_path: str, output path for the merged CSV
    - drop_duplicates: bool, whether to drop duplicate rows in the merged DataFrame

    Returns:
    - None (writes merged CSV to merged_csv_path)
    """
    if not csv_file_paths:
        raise ValueError("No CSV file paths provided.")
    
    # Read the first CSV and use its header as the standard
    merged_df = pd.read_csv(csv_file_paths[0], header=0)
    
    # Read remaining CSVs, force their columns to match the first
    for path in csv_file_paths[1:]:
        df = pd.read_csv(path, header=0)
        df.columns = merged_df.columns  # force columns to match first CSV
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # Save merged CSV
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"✅ Merged CSV saved to: {merged_csv_path}")

import pandas as pd
from pathlib import Path

def analyze_column_differences(csv_file_paths):
    """
    Analyze column differences across multiple CSV files.

    Parameters:
    - csv_file_paths: list of str, paths to CSV files to analyze

    Returns:
    - None (prints analysis to console)
    """
    if not csv_file_paths:
        raise ValueError("No CSV file paths provided.")

    print("\n" + "="*80)
    print("COLUMN SCHEMA ANALYSIS")
    print("="*80)

    # Read headers from all files
    file_columns = {}
    for path in csv_file_paths:
        df = pd.read_csv(path, nrows=0)  # Read only headers
        file_columns[Path(path).name] = set(df.columns)

    # Get reference columns from first file
    reference_file = Path(csv_file_paths[0]).name
    reference_cols = file_columns[reference_file]

    print(f"\n📊 Total files to analyze: {len(csv_file_paths)}")
    print(f"📋 Reference file: {reference_file}")
    print(f"📋 Reference column count: {len(reference_cols)}\n")

    # Check each file against reference
    all_columns = set()
    files_with_differences = []

    for i, (filename, cols) in enumerate(file_columns.items()):
        all_columns.update(cols)

        missing_cols = reference_cols - cols
        extra_cols = cols - reference_cols

        if missing_cols or extra_cols:
            files_with_differences.append(filename)
            print(f"\n{'─'*80}")
            print(f"File {i+1}/{len(csv_file_paths)}: {filename}")
            print(f"{'─'*80}")
            print(f"Total columns: {len(cols)}")

            if missing_cols:
                print(f"\n❌ Missing columns ({len(missing_cols)}):")
                for col in sorted(missing_cols)[:10]:  # Show first 10
                    print(f"   - {col}")
                if len(missing_cols) > 10:
                    print(f"   ... and {len(missing_cols) - 10} more")

            if extra_cols:
                print(f"\n➕ Extra columns ({len(extra_cols)}):")
                for col in sorted(extra_cols)[:10]:  # Show first 10
                    print(f"   + {col}")
                if len(extra_cols) > 10:
                    print(f"   ... and {len(extra_cols) - 10} more")
        else:
            print(f"✅ File {i+1}/{len(csv_file_paths)}: {filename} - Schema matches reference")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files with matching schema: {len(csv_file_paths) - len(files_with_differences)}")
    print(f"Files with schema differences: {len(files_with_differences)}")
    print(f"Total unique columns across all files: {len(all_columns)}")
    print(f"Columns in reference file: {len(reference_cols)}")
    print(f"Columns only in other files: {len(all_columns - reference_cols)}")
    print("="*80 + "\n")

    if files_with_differences:
        print("⚠️  WARNING: Schema mismatches detected. Merge will fail unless schemas are compatible.")
        print("Consider using only files with matching schemas or update merge logic to handle differences.\n")
    else:
        print("✅ All files have matching schemas. Safe to merge.\n")

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

    # Analyze column differences before merging
    print("\n🔍 Analyzing column schemas before merge...")
    analyze_column_differences(csv_file_paths)

    # Read the first CSV and use its header as the standard
    merged_df = pd.read_csv(csv_file_paths[0], header=0)

    # Read remaining CSVs, force their columns to match the first
    for path in csv_file_paths[1:]:
        df = pd.read_csv(path, header=0)
        df = df[merged_df.columns]  # force column order, not just names
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Save merged CSV
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"✅ Merged CSV saved to: {merged_csv_path}")

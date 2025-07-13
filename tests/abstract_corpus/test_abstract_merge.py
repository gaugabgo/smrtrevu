import pandas as pd
import pytest
from pathlib import Path

from revu.abstract_corpus.abstract_merge import merge_csv_files

@pytest.fixture
def sample_csv_files(tmp_path: Path):
    """
    Create three temporary CSV files with compatible data but different column orders.
    Returns:
        - List of input file paths
        - Output file path
    """
    # Create input DataFrames
    df1 = pd.DataFrame({'title': ['Doc1', 'Doc2'], 'year': [2020, 2021]})
    df2 = pd.DataFrame({'year': [2022, 2023], 'title': ['Doc3', 'Doc4']})
    df3 = pd.DataFrame({'title': ['Doc5'], 'year': [2024]})

    # Write to CSV files
    files = []
    for i, df in enumerate([df1, df2, df3], start=1):
        file = tmp_path / f"file{i}.csv"
        df.to_csv(file, index=False)
        files.append(file)

    # Output path for merged CSV
    merged_file = tmp_path / "merged.csv"
    return files, merged_file

def test_merge_csv_files(sample_csv_files):
    """
    Test that merge_csv_files correctly merges CSVs:
    - Ensures consistent column order (based on first CSV)
    - Combines all rows
    - Writes a valid output file
    """
    input_files, merged_path = sample_csv_files

    # Run merge function
    merge_csv_files([str(f) for f in input_files], str(merged_path))

    # Assert output file was created
    assert merged_path.exists()

    # Manually define expected DataFrame
    expected_df = pd.DataFrame({
        'title': ['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5'],
        'year': [2020, 2021, 2022, 2023, 2024]
    })

    # Load actual result and compare
    result_df = pd.read_csv(merged_path)
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_merge_csv_no_input():
    """
    Test that merge_csv_files raises a ValueError when no input files are provided.
    """
    with pytest.raises(ValueError, match="No CSV file paths provided."):
        merge_csv_files([], "dummy_output.csv")

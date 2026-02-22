import click
from revu.abstract_corpus.abstract_metadata_linker import extract_and_deduplicate_metadata


@click.command()
@click.option('--original_csv', required=True, multiple=True, help='Path(s) to original CSV file(s) with metadata (can specify multiple times)')
@click.option('--metadata_csv', required=True, help='Path to output merged metadata CSV file')
@click.option('--metadata_deduplicated_csv', required=True, help='Path to output deduplicated metadata CSV file')
@click.option('--metadata_columns', required=True, help='Comma-separated list of metadata columns to extract (should include id,doi,title)')
@click.option('--topic_model_csv', default=None, help='Path to topic model CSV for ID validation (optional)')
@click.option('--log_file', default=None, help='Path to log file (optional)')
def extract_metadata(original_csv, metadata_csv, metadata_deduplicated_csv, metadata_columns, topic_model_csv, log_file):
    """
    Extract and deduplicate metadata from original CSV files.

    This script:
    1. Extracts specified metadata columns from original CSV files
    2. Saves merged metadata to CSV
    3. Applies deduplication logic prioritizing records with full abstract information
    4. Saves deduplicated metadata to separate CSV
    5. Validates ID matching with topic model data (if provided)

    The metadata files are kept separate from topic model data and can be linked
    during visualization or analysis using the ID column.

    Example usage:

    \b
    # Extract metadata columns and deduplicate
    python -m revu.scripts.cli_link_metadata \\
        --original_csv data/abstract_output/parsed_output_BE.csv \\
        --original_csv data/abstract_output/parsed_output_BE_nbib.csv \\
        --metadata_csv data/metadata_merged.csv \\
        --metadata_deduplicated_csv data/metadata_deduplicated.csv \\
        --metadata_columns "id,doi,title,au,dp,jt"

    \b
    # With topic model validation and logging
    python -m revu.scripts.cli_link_metadata \\
        --original_csv data/abstract_output/parsed_output_BE.csv \\
        --metadata_csv data/metadata_merged.csv \\
        --metadata_deduplicated_csv data/metadata_deduplicated.csv \\
        --metadata_columns "id,doi,title,au,dp,jt" \\
        --topic_model_csv data/topic_model_output.csv \\
        --log_file data/metadata_extraction_log.txt
    """

    # Parse metadata columns
    metadata_columns_list = [col.strip() for col in metadata_columns.split(',')]

    # Convert original_csv tuple to list
    original_csv_files = list(original_csv)

    # Call the extraction function
    extract_and_deduplicate_metadata(
        original_csv_files=original_csv_files,
        metadata_csv=metadata_csv,
        metadata_deduplicated_csv=metadata_deduplicated_csv,
        metadata_columns=metadata_columns_list,
        topic_model_csv=topic_model_csv,
        log_file=log_file
    )


if __name__ == '__main__':
    extract_metadata()

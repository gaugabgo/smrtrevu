import click
from revu.abstract_corpus.abstract_metadata_linker import validate_and_link_metadata


@click.command()
@click.option('--metadata_deduplicated_csv', required=True, help='Path to deduplicated metadata CSV file (input)')
@click.option('--topic_model_csv', required=True, help='Path to topic model CSV file (input)')
@click.option('--validated_metadata_csv', required=True, help='Path to output CSV with validated (1:1 matched) records')
@click.option('--log_file', default=None, help='Path to log file (optional)')
def link_metadata(metadata_deduplicated_csv, topic_model_csv, validated_metadata_csv, log_file):
    """
    Validate one-to-one ID matching between deduplicated metadata and topic model data,
    then save matched records to a validated metadata CSV.

    Records in the topic model that have no metadata match are reported but expected.
    Metadata records with no topic model match are excluded from the output.
    Linking is done on the 'id' column present in both files.

    Example usage:

    \b
    python -m revu.scripts.cli_link_metadata \
        --metadata_deduplicated_csv data/causal_metadata_deduplicated.csv \
        --topic_model_csv data/causal_modeled_25Mar2026.csv \
        --validated_metadata_csv data/causal_metadata_validated.csv \
        --log_file data/metadata_validation_log.txt
    """

    validate_and_link_metadata(
        metadata_deduplicated_csv=metadata_deduplicated_csv,
        topic_model_csv=topic_model_csv,
        validated_metadata_csv=validated_metadata_csv,
        log_file=log_file
    )


if __name__ == '__main__':
    link_metadata()

import click
import os
from abstract_corpus.abstract_parser import parse_nbib_folder, parse_ris_folder, preprocess_and_save
from abstract_corpus.abstract_merge import merge_csv_files
from abstract_corpus.abstract_deduplicate import deduplicate_csv

@click.group()
def cli():
    """Citation parsing toolkit for NBIB and RIS files."""
    pass

@cli.command()
@click.argument("filetype", type=click.Choice(["nbib", "ris"]))
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path())
@click.argument("output_csv", type=click.Path())
def parse(filetype, input_folder, output_csv):
    """
    Parse all citation files of FILETYPE in INPUT_FOLDER into OUTPUT_CSV.

    FILETYPE: 'nbib' or 'ris'

    INPUT_FOLDER: Folder containing citation files.

    OUTPUT_FOLDER: Destination folder for parsed data

    OUTPUT_CSV: CSV file to write parsed output.
    """
    if filetype == "nbib":
        parse_nbib_folder(input_folder, output_csv)
    elif filetype == "ris":
        parse_ris_folder(input_folder, output_csv)

@cli.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path())
def preprocess_ris(input_folder, output_folder):
    """
    Normalize date formats for ris files from EBSCO in INPUT_FOLDER and save results to OUTPUT_FOLDER.
    This is a first step prior to parsing

    INPUT_FOLDER: Folder containing raw .ris files.

    OUTPUT_FOLDER: Folder to save normalized .ris files.
    """
    preprocess_and_save(input_folder, output_folder)

@cli.command()
@click.argument("csv_file_paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.argument('--output', '-o', 'merged_csv_path', type=click.Path(), required=True, help='Output CSV file path')
def merge_abstracts(csv_file_paths, merged_csv_path):
    merge_csv_files(csv_file_paths, merged_csv_path)

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, file_okay=False))
@click.argument("output_file", type=click.Path())
@click.argument("log_file", type=click.Path())
def deduplicate(input_file, output_file, log_file):
    deduplicate_csv(input_file, output_file, log_file)

if __name__ == "__main__":
    cli()

# - Example run from command line - 
# python abstract_corpus/abstract_parser.py parse nbib data/abstract_import/nbib_raw data/abstract_output/nbib_parsed output.csv
# python abstract_corpus/abstract_parser.py parse ris data/abstract_import/ris_raw data/abstract_output/ris_parsed output.csv
# python abstract_corpus/abstract_parser.py data/abstract_import/raw_EBSCO_ris data/abstract_import/ris_raw
# python abstract_corpus/abstract_merge.py data/abstract_output/nbib_parsed data/abstract_output/ris_parsed -o data/abstract_output/merged
# python abstract_corpus/abstract_deduplicate.py data/abstract_output/merged data/abstract_output/deduplicated deduplicationlog.txt
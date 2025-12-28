import click
import os

from revu.abstract_corpus.abstract_parser import parse_nbib_folder, parse_ris_folder
from revu.abstract_corpus.ris_normalizer import preprocess_and_save
from revu.abstract_corpus.abstract_merge import merge_csv_files
from revu.abstract_corpus.abstract_deduplicate import deduplicate_csv

@click.group()
def cli():
    """Citation parsing toolkit for NBIB and RIS files."""
    pass

@cli.command()
@click.argument("filetype", type=click.Choice(["nbib", "ris"]))
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path())
@click.argument("output_csv", type=str)  # Just the filename
def parse(filetype, input_folder, output_folder, output_csv):
    """
    Parse all citation files of FILETYPE in INPUT_FOLDER into OUTPUT_CSV.

    FILETYPE: 'nbib' or 'ris'
    INPUT_FOLDER: Folder containing citation files.
    OUTPUT_FOLDER: Destination folder for parsed data
    OUTPUT_CSV: Name of CSV file to write parsed output (e.g., output.csv)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct full path
    full_output_path = os.path.join(output_folder, output_csv)

    if filetype == "nbib":
        parse_nbib_folder(input_folder, full_output_path)
    elif filetype == "ris":
        parse_ris_folder(input_folder, full_output_path)

    print(f"Parsing {input_folder} and saving to {full_output_path}")

@cli.command(name="preprocess")
@click.argument('input_folder', type=click.Path(exists=True, file_okay=False))
@click.argument('output_folder', type=click.Path(file_okay=False))
def preprocess_and_save_command(input_folder, output_folder):
    """
    Normalize date formats for RIS files from EBSCO in INPUT_FOLDER and save results to OUTPUT_FOLDER.

    INPUT_FOLDER: Folder containing raw .ris files.
    OUTPUT_FOLDER: Folder to save normalized .ris files.
    """
    preprocess_and_save(input_folder, output_folder)
    print(f"Normalizing {input_folder} and saving to {output_folder}")

@cli.command()
@click.option('--input', '-i', 'csv_file_paths', type=click.Path(exists=True), multiple=True, required=True, help='Input CSV file path(s)')
@click.option('--output', '-o', 'merged_csv_path', type=click.Path(), required=True, help='Output CSV file path')
def cli_merge_csv_files(csv_file_paths, merged_csv_path):
    if not merged_csv_path.endswith('.csv'):
        merged_csv_path += '.csv'
    
    merge_csv_files(csv_file_paths, merged_csv_path)
    print(f"Merging {csv_file_paths} and saving to {merged_csv_path}")

@cli.command()
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True), required=True, help='Input CSV file')
@click.option('--output', '-o', 'output_file', type=click.Path(), required=True, help='Output CSV file')
@click.option('--log', '-l', 'log_file', type=click.Path(), required=True, help='Log file path')
def cli_deduplicate_csv(input_file, output_file, log_file):
    if not output_file.endswith('.txt'):
        output_file += '.txt'
    deduplicate_csv(input_file, output_file, log_file)
    print(f"Deduplicating {input_file} and saving to {output_file}")
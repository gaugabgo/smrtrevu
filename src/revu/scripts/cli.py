import click

# Import commands from various modules
from revu.scripts.cli_parsers import parse, preprocess_and_save_command, cli_merge_csv_files, cli_deduplicate_csv
from revu.scripts.cli_modeling import run_model, visualize
from revu.scripts.cli_preprocessing import preprocess_texts
from revu.scripts.cli_oa import download, extract

@click.group()
def cli():
    """ Revu: A toolkit for citation parsing, topic modeling, and OA text extraction."""
    pass

# ----------------------------
# Parsing Commands
# ----------------------------
cli.add_command(parse)
cli.add_command(preprocess_and_save_command, name="preprocess")
cli.add_command(cli_merge_csv_files, name="merge")
cli.add_command(cli_deduplicate_csv, name="deduplicate")

# ----------------------------
# Preprocessing (text) Command
# ----------------------------
cli.add_command(preprocess_texts, name="preprocess-texts")

# ----------------------------
# Modeling Commands
# ----------------------------
@click.group()
def model():
    """Topic modeling with BERTopic"""
    pass

model.add_command(run_model, name="run")
model.add_command(visualize, name="visualize")

cli.add_command(model)

# ----------------------------
# OA Downloader Commands
# ----------------------------
@click.group()
def oa():
    """Open Access article management"""
    pass

oa.add_command(download)
oa.add_command(extract)

cli.add_command(oa)

"""
Example CL commands:
# Parse citations
revu parse nbib data/abstract_import/nbib_raw data/abstract_output/nbib_parsed nbibparsed.csv
revu parse ris data/abstract_import/ris_raw data/abstract_output/ris_parsed risparsed.csv

# Normalize RIS
revu preprocess data/abstract_import/raw_EBSCO_ris data/abstract_output/normalized_ris

# Merge parsed abstract data
revu merge 

# Deduplicate merged abstracts
revu deduplicate 

# Run topic modeling
revu model run --input_csv parsed.csv --output_csv topics.csv

# Visualize topics
revu model visualize --input_csv topics.csv

# Download OA PDFs
revu oa download -i dois.txt -o metadata.csv --email user@example.com

# Extract from OA PDFs
revu oa extract --pdf-dir oa_pdfs --output-dir oa_texts

"""
import click

# Import commands from various modules
from revu.scripts.cli_parsers import parse, preprocess_and_save_command, cli_merge_csv_files, cli_deduplicate_csv
from revu.scripts.cli_modeling import run_model
from revu.scripts.cli_preprocessing import preprocess_texts
from revu.scripts.cli_oa import metadata, download, extract

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
# Modeling and Visulization Commands
# ----------------------------
@click.group()
def model():
    """Topic modeling with BERTopic"""
    pass

model.add_command(run_model, name="run")

cli.add_command(model)

# ----------------------------
# OA Downloader Commands
# ----------------------------
@click.group()
def oa():
    """Open Access article management"""
    pass

oa.add_command(metadata, name="metadata")
oa.add_command(download, name="download")
oa.add_command(extract)

cli.add_command(oa)

"""
Example CL commands:
# Parse citations
revu parse nbib data/abstract_import/estimand_review/nbib data/estimand_review/abstract_output estimand_nbibparsed.csv
revu parse ris data/abstract_import/estimand_review/ris data/estimand_review/abstract_output estimand_risparsed.csv

# Normalize RIS
revu preprocess data/abstract_import/raw_EBSCO_ris data/abstract_output/normalized_ris

# Merge parsed abstract data
revu merge -i data/estimand_review/*.csv -o data/estimand_review/abstract_merged.csv

# Deduplicate merged abstracts
revu deduplicate -i data/estimand_review/abstract_merged.csv -o data/estimand_review/abstract_deduplicated.csv -l data/estimand_review

# Pre-process texts for modeling
revu preprocess-texts \
  --source-type abstract \
  --input-path data/estimand_review/estimand_urbanhealthpolicy_24Oct2025.csv \
  --filtering disable

# Run topic modeling + visualizations
revu model run \
  --input_csv data/estimand_review/processed_estimand_urbanhealthpolicy_24Oct2025.csv \
  --output_csv estimand_modeled_24Oct2025_scibert.csv \
  --output_dir data/estimand_review/visualizations \
  --model_name "allenai/scibert_scivocab_uncased" \
  --n_neighbors 5 \
  --min_cluster_size 5

# fetch metadata
revu oa metadata -i data/BE_DOIs.csv --cache-dir data/ft_import --email gaugabgo.dev@gmail.com

# Download OA PDFs
revu oa download --input data/ft_import/metadata.csv --cache_dir data/ft_output/oa_pdfs

# Extract from OA PDFs
revu oa extract --pdf-dir oa_pdfs --output-dir oa_texts

"""
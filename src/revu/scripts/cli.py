import click

# Import commands from various modules
from revu.scripts.cli_parsers import parse, preprocess_and_save_command, cli_merge_csv_files, cli_deduplicate_csv
from revu.scripts.cli_modeling import run_model
from revu.scripts.cli_topic_modeling import run_topic_modeling
from revu.scripts.cli_visualizations import create_visualizations
from revu.scripts.cli_embeddings import compute_embeddings
from revu.scripts.cli_reduce_embeddings import reduce_embeddings
from revu.scripts.cli_preprocessing import preprocess_texts
from revu.scripts.cli_oa import metadata, download, extract
from revu.scripts.cli_link_metadata import extract_metadata
from revu.scripts.cli_bibliographicanalysis import run_bibliometric_analysis, build_network
from revu.scripts.cli_visu_bibliographic import visualize_bibliometric

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
cli.add_command(extract_metadata, name="extract-metadata")

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
model.add_command(compute_embeddings, name="embed")
model.add_command(reduce_embeddings, name="reduce")
model.add_command(run_topic_modeling, name="fit")
model.add_command(create_visualizations, name="visualize")

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

# ----------------------------
# Bibliometric Analysis Commands
# ----------------------------
@click.group()
def biblio():
    """Bibliometric analysis with citation networks"""
    pass

biblio.add_command(run_bibliometric_analysis, name="analyze")
biblio.add_command(build_network, name="build-network")
biblio.add_command(visualize_bibliometric, name="visualize")

cli.add_command(biblio)

"""
Example CL commands:
# Parse citations
revu parse nbib data/abstract_import/estimand_review/nbib data/estimand_review/abstract_output estimand_nbibparsed.csv
revu parse ris data/abstract_import/estimand_review/ris data/estimand_review/abstract_output estimand_risparsed.csv

# Normalize RIS
revu preprocess data/abstract_import/raw_EBSCO_ris data/abstract_output/normalized_ris

# Merge parsed abstract data
revu merge -i 'data/estimand_review/Originals/*.csv' -o data/estimand_review/abstract_merged_3Jan2025.csv

# Deduplicate merged abstracts
revu deduplicate -i data/estimand_review/abstract_merged_3Jan2025.csv -o data/estimand_review/abstract_deduplicated_3Jan2025.csv -l data/estimand_review

# Extract and deduplicate metadata for linking with topic models
revu extract-metadata \
  --original_csv data/estimand_review/Originals/works-csv-5QZ56CVXSi6s4LecbzXh54.csv \
  --original_csv data/estimand_review/Originals/works-csv-7QXPEioaEJthoVLybDf3ro.csv \
  --original_csv data/estimand_review/Originals/works-csv-auBgT4UEkrvrsNj9J7p9fX.csv \
  --original_csv data/estimand_review/Originals/works-csv-AVHHeR6NfhHJA4JwCnPk8Y.csv \
  --original_csv data/estimand_review/Originals/works-csv-AXps4nTtx4PSb8tzUpUWEq.csv \
  --original_csv data/estimand_review/Originals/works-csv-BPivDNz5RjkhF74pUMWNrS.csv \
  --original_csv data/estimand_review/Originals/works-csv-cHihEHLEtfH723moCaUUBj.csv \
  --original_csv data/estimand_review/Originals/works-csv-CzXnsKZMUcnykZQqGanRUS.csv \
  --original_csv data/estimand_review/Originals/works-csv-PTBpo3B8eP7dky7HghQwYL.csv \
  --original_csv data/estimand_review/Originals/works-csv-PZcKXZsKmENorxMq5qFWdQ.csv \
  --original_csv data/estimand_review/Originals/works-csv-Wq6WYXiUcSqkjgmECi483h.csv \
  --metadata_csv data/estimand_review/metadata_merged.csv \
  --metadata_deduplicated_csv data/estimand_review/metadata_deduplicated.csv \
  --metadata_columns "id,doi,title,publication_year,language,type,cited_by_count,referenced_works_count,referenced_works,primary_topic.display_name,primary_topic.subfield.display_name,primary_topic.field.display_name,primary_topic.domain.display_name,topics.display_name,topics.subfield.display_name,topics.field.display_name,topics.domain.display_name,authorships.raw_author_name,keywords.display_name" \
  --topic_model_csv data/estimand_review/causalinference_modeled_03Jan2026.csv \
  --log_file data/metadata_log.txt

# Pre-process texts for modeling
revu preprocess-texts \
  --source-type abstract \
  --input-path openalex_CAUSALurbanhealth_results_02032026.csv \
  --filtering disable \
  --checkpoint-dir data/estimand_review/urbanhealth_causal

# Step 1: Compute embeddings separately (saves to disk)
revu model embed \
  --input_csv processed_openalex_CAUSALurbanhealth_results_02032026.csv \
  --output_path data/estimand_review/estimand_CAUSAL_UH_embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32

# Step 2: Run topic modeling with pre-computed embeddings
revu model fit \
  --input_csv data/estimand_review/urbanhealth_causal/processed_openalex_CAUSALurbanhealth_results_02032026.csv \
  --output_csv Estimand_UH_CAUSAL_modeled_3Mar2026.csv \
  --output_dir data/estimand_review \
  --embeddings_path data/estimand_review/urbanhealth_causal/estimand_CAUSAL_UH_embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 50 \
  --n_components 10 \
  --min_cluster_size 100

# Step 3: Reduce embeddings to 2D for visualizations (avoids memory issues)
revu model reduce \
  --embeddings_path data/estimand_review/urbanhealth_causal/estimand_CAUSAL_UH_embeddings.npy \
  --output_path data/estimand_review/urbanhealth_causal/estimand_UH_CAUSAL_embedding_2d.npy \
  --n_neighbors 50 \
  --min_dist 0.1

# Step 4: Create visualizations from existing model with pre-computed 2D embeddings
revu model visualize \
  --input_csv data/CausalInf_Poster_2026/causalinference_modeled_03Jan2026.csv\
  --model_path data/CausalInf_Poster_2026/bertopic_model \
  --embeddings_2d_path data/CausalInf_Poster_2026/embeddings_2d.npy \
  --metadata_csv data/CausalInf_Poster_2026/metadata_merged.csv \
  --output_dir data/CausalInf_Poster_2026

# fetch metadata
revu oa metadata -i data/BE_DOIs.csv --cache-dir data/ft_import --email gaugabgo.dev@gmail.com

# Download OA PDFs
revu oa download --input data/ft_import/metadata.csv --cache_dir data/ft_output/oa_pdfs

# Extract from OA PDFs
revu oa extract --pdf-dir oa_pdfs --output-dir oa_texts

# Run bibliometric analysis
revu biblio analyze \
  --topic_csv data/estimand_review/causalinference_modeled_03Jan2026.csv \
  --metadata_csv data/estimand_review/metadata_deduplicated.csv \
  --output_dir data/estimand_review/bibliometric_results \
  --window_size 10 \
  --topic_column topic \
  --id_column id

# Run specific bibliographic analysis types only
revu biblio analyze \
  --topic_csv data/topics.csv \
  --metadata_csv data/metadata.csv \
  --output_dir data/results \
  --analysis_type centrality

# Build citation network and save as GraphML
revu biblio build-network \
  --topic_csv data/topics.csv \
  --metadata_csv data/metadata.csv \
  --output_path data/citation_network.graphml

# Generate visualizations from bibliometric analysis results with topic names
revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --output_dir data/estimand_review/visualizations \
  --topic_info_file data/estimand_review/causalinference_modeled_03Jan2026_topic_info.csv \
  --top_n_prevalent 20 

# Generate visualizations for top 15 most prevalent topics only
revu biblio visualize \
  --data_dir data/results \
  --top_n_prevalent 15

# Generate homophily visualization for most homophilic topics
revu biblio visualize \
  --data_dir data/results \
  --viz_type homophily \
  --top_n_homophilic 10

# Generate trend visualization for growing/declining topics only
revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type trends \
  --top_n_growing 20

  revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type trends \
  --bottom_n_declining 10

  revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type centrality \
  --top_n_central 15

  revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type homophily \
  --top_n_homophilic 10

  revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type influence \
  --top_n_influential 10

  revu biblio visualize \
  --data_dir data/estimand_review/bibliometric_results \
  --viz_type influence \
  --top_n_influential 10 \
  --influence_metric h_index_proxy

  model evaulator (not part of main cli)
  python3 src/revu/model_evaluator.py \
  --input_csv data/estimand_review/urbanhealth_causal/processed_openalex_CAUSALurbanhealth_results_02032026.csv \
  --embeddings_path data/estimand_review/urbanhealth_causal/estimand_CAUSAL_UH_embeddings.npy \
  --model_name sentence-transformers/allenai-specter
  --output_dir data/estimand_review/urbanhealth_causal

"""
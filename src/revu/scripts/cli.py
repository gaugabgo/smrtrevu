import click

# Import commands from various modules
from revu.scripts.cli_parsers import parse, preprocess_and_save_command, cli_merge_csv_files, cli_deduplicate_csv
from revu.scripts.cli_modeling import run_model
from revu.scripts.cli_topic_modeling import run_topic_modeling
from revu.scripts.cli_visualizations import create_visualizations
from revu.scripts.cli_embeddings import compute_embeddings
from revu.scripts.cli_reduce_embeddings import reduce_embeddings
from revu.scripts.cli_preprocessing import preprocess_texts
from revu.scripts.cli_link_metadata import link_metadata
from revu.scripts.cli_bibliographicanalysis import run_bibliometric_analysis, build_network
from revu.scripts.cli_visu_bibliographic import visualize_bibliometric
from revu.scripts.cli_coauthorship import coauthorship_analysis
from revu.scripts.cli_visu_author_network import visualize_author_network
from revu.scripts.visu_author_network_dmp_cli import visualize_author_network_dmp
from revu.scripts.cli_visu_coauthorship_hub_spoke import visualize_coauth_topic, visualize_coauth_global
from revu.scripts.cli_agg_within_topic_community_nodes import agg_within_topic_community_nodes
from revu.scripts.cli_visu_dmp_coauth_overlay import visualize_dmp_coauth_overlay
from revu.scripts.cli_agg_global_community_topic_affinity import agg_global_community_topic_affinity
from revu.scripts.cli_visu_topic_community_network import visualize_topic_community_network
from revu.scripts.cli_openalex import fetch_openalex
from revu.scripts.cli_evaluate import evaluate_model

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
cli.add_command(link_metadata, name="extract-metadata")
cli.add_command(fetch_openalex, name="fetch-openalex")

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
model.add_command(evaluate_model, name="evaluate")

cli.add_command(model)

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
biblio.add_command(coauthorship_analysis, name="coauthorship-analysis")
biblio.add_command(visualize_author_network, name="visualize-author-network")
biblio.add_command(visualize_author_network_dmp, name="visualize-author-network-dmp")
biblio.add_command(visualize_coauth_topic, name="visualize-coauth-topic")
biblio.add_command(visualize_coauth_global, name="visualize-coauth-global")
biblio.add_command(agg_within_topic_community_nodes, name="agg-within-topic-community-nodes")
biblio.add_command(visualize_dmp_coauth_overlay, name="visualize-dmp-coauth-overlay")
biblio.add_command(agg_global_community_topic_affinity, name="agg-global-community-topic-affinity")
biblio.add_command(visualize_topic_community_network, name="visualize-topic-community-network")

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
  --input-path data/abstract_deduplicated_3Jan2025.csv \
  --filtering disable \
  --checkpoint-dir data/preprocess_checkpints

# Step 1: Compute embeddings separately (saves to disk)
revu model embed \
  --input_csv data/processed_abstract_deduplicated_25Mar2026.csv \
  --output_path data \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32

# Step 2: Run topic modeling with pre-computed embeddings
revu model fit \
  --input_csv data/processed_abstract_deduplicated_25Mar2026.csv \
  --output_csv data/causal_modeled_25Mar2026.csv \
  --output_dir data \
  --embeddings_path data/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --n_neighbors 15 \
  --n_components 5 \
  --min_cluster_size 100 \
  --cluster_selection_method eom \
  --outlier_strategy none

# Step 3: Reduce embeddings to 2D for visualizations (avoids memory issues)
revu model reduce \
  --embeddings_path data/embeddings.npy \
  --output_path data \
  --n_neighbors 15 \
  --min_dist 0.1

# Step 4: Create visualizations from existing model with pre-computed 2D embeddings
revu model visualize \
  --input_csv data/causal_modeled_25Mar2026.csv \
  --model_path data/bertopic_model \
  --embeddings_2d_path data/embeddings_2d.npy \
  --metadata_csv data/causal_metadata_validated.csv \
  --label_mode custom \
  --output_dir data

# Run bibliometric analysis
revu biblio analyze \
  --topic_csv data/causal_modeled_25Mar2026.csv \
  --metadata_csv data/causal_metadata_validated.csv \
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

# Generate all visualizations (outlier topics filtered automatically)
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --output_dir data/visualizations \
  --topic_info_file data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
  --top_n_prevalent 20

# Generate all visualizations including geographic map
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --output_dir data \
  --topic_info_file data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
  --metadata_csv data/causal_metadata_validated.csv

# PageRank network with paper satellite nodes
revu biblio visualize \
  --data_dir data/bibliometric_results/ \
  --topic_info_file data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
  --viz_type centrality \
  --top_n_central 15 \
  --works_file data/causal_modeled_25Mar2026.csv

# Topic Influence graphs
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --viz_type influence \
  --top_n_influential 10 \
  --normalize_by_works \
  --min_citations 20

# Trend visualizations
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --viz_type trends \
  --top_n_each 8 \
  --sort_spans_by span_length \
  --color_growing '#d0d1e6' \
  --color_declining '#67a9cf'

# Geographic map only (static PNG)
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --viz_type geo \
  --metadata_csv data/causal_metadata_validated.csv \
  --geo_cmap PuBuGn

# Geographic map (interactive HTML, custom colormap)
revu biblio visualize \
  --data_dir data/bibliometric_results \
  --viz_type geo \
  --metadata_csv data/causal_metadata_validated.csv \
  --geo_format interactive \
  --geo_cmap PuBuGn

  model evaulator (not part of main cli)
  python src/revu/model_evaluator.py \
  --input_csv data/processed_abstract_deduplicated_25Mar2026.csv \
  --embeddings_path data/embeddings.npy \
  --model_name "sentence-transformers/all-MiniLM-L6-v2"
  --output_dir data/

revu biblio coauthorship-analysis \
  --metadata_csv data/causal_metadata_validated.csv \
  --topic_csv data/causal_modeled_25Mar2026.csv \
  --output_dir data/coauthorship_results \
  --per_topic \
  --min_works_per_topic 15 \
  --analysis_type all \
  --affiliation_mode both  

  revu biblio visualize-coauth-topic \
  --coauthorship-dir data/coauthorship_results \
  --top-n-topics 10 \
  --top-k-per-topic 5 \
  --topic-info-file data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
  --output data/visualizations/coauth_hub_topic.png


revu biblio visualize-coauth-global \
  --coauthorship-dir data/coauthorship_results \
  --top-n 100 \
  --label-top-n 30 \
  --output data/visualizations/coauth_hub_global.png

revu biblio visualize-dmp-coauth-overlay \
      --modeled         data/causal_modeled_25Mar2026.csv \
      --embeddings      data/embeddings_2d.npy \
      --community-nodes data/within_topic_community_nodes.csv \
      --topic-info      data/causal_modeled_1Apr2026_topic_info_customlabels.csv \
      --metadata        data/causal_metadata_validated.csv \
      --output          data/visualizations/topic_dmp_coauth_overlay.html \
      --top-k-communities 5

"""
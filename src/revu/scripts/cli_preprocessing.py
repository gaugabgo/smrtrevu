import click
import pandas as pd
from pathlib import Path
from revu.pipeline.preprocessing_dispatcher import preprocess_abstract_df, preprocess_fulltext_txt

def run_preprocessing(data_source, source_type):
    if source_type == "abstract":
        return preprocess_abstract_df(data_source)
    elif source_type == "fulltext":
        return preprocess_fulltext_txt(data_source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

@click.command()
@click.option("--source-type", type=click.Choice(["abstract", "fulltext"]), required=True)
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--filtering", type=click.Choice(["enable", "disable"]), default="disable",
              help="Enable or disable filtering for fulltext preprocessing")
def preprocess_texts(source_type, input_path, filtering):
    input_path_resolved = Path(input_path).resolve()
    if source_type == "abstract":
        df = pd.read_csv(input_path_resolved)
        processed = run_preprocessing(df, source_type)
    elif source_type == "fulltext":
        # Pass filtering option as needed, modify preprocess_fulltext_txt to accept it
        processed = run_preprocessing(input_path_resolved, source_type)
    
    print("✅ Preprocessing completed.")


import click
from pathlib import Path
import pandas as pd

from revu.pipeline.preprocessing_dispatcher import preprocess_abstract_df, preprocess_fulltext

def run_preprocessing(data_source, source_type):
    if source_type == "abstract":
        return preprocess_abstract_df(data_source)
    elif source_type == "fulltext":
        return preprocess_fulltext(data_source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

@click.command()
@click.option("--source-type", type=click.Choice(["abstract", "fulltext"]), required=True)
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--filtering", type=click.Choice(["enable", "disable"]), default="disable",
              help="Enable or disable filtering for fulltext preprocessing")
def preprocess_texts(source_type, input_path, filtering):
    """Preprocess abstract or fulltext data"""
    input_path_resolved = Path(input_path).resolve()
    print(f"\n📥 Reading from: {input_path_resolved}\n")
    if source_type == "abstract":
        df = pd.read_csv(input_path_resolved)
        df.columns = df.columns.str.strip()
        print("✅ Columns found:")
        for col in df.columns:
            print(f"- {repr(col)}")
        processed = run_preprocessing(df, source_type)
    elif source_type == "fulltext":
        processed = run_preprocessing(input_path_resolved, source_type)
    
     # DEBUG: Show header and sample rows
    print("✅ CSV loaded. Columns:")
    print(df.columns.tolist())
    print("\nSample rows:")
    print(df.head())

    output_path = input_path_resolved.parent / f"processed_{input_path_resolved.name}"
    processed.to_csv(output_path, index=False)
    print(f"✅ Preprocessing completed. Output saved to {output_path}")

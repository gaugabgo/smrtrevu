import click
from pathlib import Path
import pandas as pd
from revu.oa_downloader.oa_metadata_fetcher import OaMetadataFetcher
from revu.oa_downloader.download_2 import OaDownloader  

@click.group()
def cli():
    pass

# Metadata Fetcher
@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to input file (CSV/TXT) with DOIs')
@click.option('--cache-dir', default='json', type=click.Path(), help='Directory to save metadata file')
@click.option('--email', required=True, help='Email address for Unpaywall API (required)')
def metadata(input, cache_dir, email):
    
    # read DOIS
    dois_df = pd.read_csv(input)
    dois = dois_df["DOI"]

    metadata_fetcher = OaMetadataFetcher(
        cache_dir=cache_dir,
        email=email
    )

    metadata_fetcher.fetch_all_metadata(dois)

    # output == cache_dir
    metadata_fetcher.build_metdata_table()


# Download
@cli.command()
@click.option("--input", required=True, help="CSV input file path")
@click.option("--cache_dir", required=True, help="Cache directory")
@click.option('--licenses', multiple=True, default=['cc-by', 'cc0', 'public-domain', 'cc-by-sa'], show_default=True,
              help='License(s) to accept (e.g. cc-by, cc0)')
@click.option('--download-method', default='curl', show_default=True, 
              type=click.Choice(['curl', 'selenium', 'requests']),
              help='Download method to use')
def download(input, cache_dir, licenses, download_method):
    """Download PDFs from Open Access sources."""
    
    downloader = OaDownloader(
        cache_dir=cache_dir,
        allowed_licenses=list(licenses),  
        download_method=download_method    
    )
    
    downloader.run(
        input=input   
    )


# Extract Text
@cli.command()
@click.option('--pdf-dir', required=True, type=click.Path(exists=True), help='Directory containing PDFs')
@click.option('--output-dir', required=True, type=click.Path(), help='Directory to save extracted text files')
@click.option('--method', default='default', show_default=True,
              type=click.Choice(['default', 'layout', 'dict', 'blocks'], case_sensitive=False),
              help='PDF extraction method to use')
def extract(pdf_dir, output_dir, method):
    """
    Parse text data from XML structured articles
    """
    click.echo(f"📑 Extracting text from PDFs in: {pdf_dir}")
    click.echo(f"📁 Saving extracted text to: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)



import click
from pathlib import Path
from oa_downloader.download import download_oa_articles  # Adjust if your function lives elsewhere
from oa_downloader.extract import extract_and_save_pdf_text  # Same here

@click.group()
def cli():
    """📚 OA Downloader CLI — Manage download and extraction of OA articles."""
    pass

# ---------------------
# Download Command
# ---------------------
@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Path to input file (CSV/TXT) with DOIs')
@click.option('--output', '-o', default='oa_articles.csv', type=click.Path(), help='Path to save output metadata CSV')
@click.option('--pdf-dir', default='oa_pdfs', type=click.Path(), help='Directory to save downloaded PDFs')
@click.option('--email', required=True, help='Email address for Unpaywall API (required)')
@click.option('--rate-limit', default=5.0, show_default=True, help='Delay between Unpaywall API requests (seconds)')
@click.option('--license', 'licenses', multiple=True, default=['cc-by', 'cc0'], show_default=True,
              help='License(s) to accept (e.g. cc-by, cc0)')
def download(input, output, pdf_dir, email, rate_limit, licenses):
    """
    🔽 Download Open Access articles using Unpaywall and save PDFs + metadata.
    """
    click.echo(f"📥 Downloading OA articles from: {input}")
    click.echo(f"📄 Saving metadata to: {output}")
    click.echo(f"📂 Saving PDFs to: {pdf_dir}")
    
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    download_oa_articles(
        input_file=input,
        output_file=output,
        pdf_dir=pdf_dir,
        email=email,
        rate_limit=rate_limit,
        allowed_licenses=list(licenses)
    )

# ---------------------
# Extract Command
# ---------------------
@cli.command()
@click.option('--pdf-dir', required=True, type=click.Path(exists=True), help='Directory containing PDFs')
@click.option('--output-dir', required=True, type=click.Path(), help='Directory to save extracted text files')
@click.option('--method', default='default', show_default=True,
              type=click.Choice(['default', 'layout', 'dict', 'blocks'], case_sensitive=False),
              help='PDF extraction method to use')
def extract(pdf_dir, output_dir, method):
    """
    📄 Extract text from downloaded PDFs using PyMuPDF or similar.
    """
    click.echo(f"📑 Extracting text from PDFs in: {pdf_dir}")
    click.echo(f"📁 Saving extracted text to: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    extract_and_save_pdf_text(
        pdf_directory=pdf_dir,
        output_directory=output_dir,
        extraction_method=method
    )

# ---------------------
# Entry Point
# ---------------------
if __name__ == '__main__':
    cli()

# - Example run in command line - 
# python scripts/run_oadownloader.py download \
#  --input data/abstract_import/dois.txt \
#  --output data/abstract_output/oa_metadata.csv \
#  --pdf-dir data/ft_import \
#  --email your@email.com
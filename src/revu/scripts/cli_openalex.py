import click

from revu.abstract_corpus.openalex_getter import fetch_all, save_csv


@click.command()
@click.option("--query", "-q", required=True,
              help="Full-text search query (OpenAlex filter syntax, Boolean operators supported).")
@click.option("--output", "-o", required=True,
              help="Output CSV file path.")
@click.option("--api-key", required=True, envvar="OPENALEX_API_KEY",
              help="OpenAlex API key. Can also be set via OPENALEX_API_KEY environment variable.")
@click.option("--types", default="article|dissertation|preprint|book-chapter|book-section",
              show_default=True,
              help="Publication types to include, pipe-separated.")
@click.option("--from-date", default="1900-01-01", show_default=True,
              help="Start date (YYYY-MM-DD).")
@click.option("--to-date", default="2099-12-31", show_default=True,
              help="End date (YYYY-MM-DD).")
def fetch_openalex(query, output, api_key, types, from_date, to_date):
    """Retrieve records from the OpenAlex API and save to a CSV file.

    Results include title, abstract, authors, affiliations, citation counts,
    and referenced works — fields used by downstream bibliometric commands.
    """
    date_filter = f"from_publication_date:{from_date},to_publication_date:{to_date}"
    click.echo(f"Querying OpenAlex: {query}")
    records = fetch_all(api_key=api_key, search_str=query, type_filter=types, date_filter=date_filter)
    save_csv(records, output)

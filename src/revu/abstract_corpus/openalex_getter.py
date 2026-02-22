import requests
import time
import csv
import json

API_KEY     = "MEyGPb6mG78k6SjsFqUZCN"
SEARCH_STR  = "https://api.openalex.org/works?page=1&filter=title_and_abstract.search:(%22causal+inference%22)+AND+(%22urban+health%22+OR+%22urban+plan*%22+OR+%22urban+environment%22+OR+%22mobility%22+OR+%22geospatial%22+OR+%22spatial%22+OR+%22transport%22+OR+%22environm*%22+OR+%22built+environ*%22),type:article|dissertation|preprint|book-chapter|report|book-section&sort=relevance_score:desc&per_page=10&include_xpac=true&mailto=ui@openalex.org"   
OUTPUT_FILE = "openalex_CAUSALurbanhealth_results.csv"

SEARCH_STR  = '("causal inference" OR "causal*") AND ("urban health" OR "urban plan*" OR "urban environment" OR "mobility" OR "geospatial" OR "spatial" OR "transport" OR "environm*" OR "built environ*")'

TYPE_FILTER = "article|dissertation|preprint|book-chapter|report|book-section"

SELECT_FIELDS = ",".join([
    "id",
    "doi",
    "title",
    "publication_year",
    "language",
    "type",
    "cited_by_count",
    "referenced_works_count",
    "referenced_works",
    "primary_topic",
    "topics",
    "authorships",
    "keywords",
    "abstract_inverted_index",
])
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_abstract(inv_index: dict) -> str:
    """Convert OpenAlex inverted abstract index back to plain text."""
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def flatten_record(r: dict) -> dict:
    """Extract and flatten the requested metadata fields."""
    # primary_topic
    pt = r.get("primary_topic") or {}
    pt_subfield = pt.get("subfield") or {}
    pt_domain   = pt.get("domain") or {}

    # topics (all assigned topics, semi-colon separated per level)
    topics = r.get("topics") or []
    topics_name     = "; ".join(t.get("display_name", "")                       for t in topics)
    topics_subfield = "; ".join((t.get("subfield") or {}).get("display_name", "") for t in topics)
    topics_field    = "; ".join((t.get("field")    or {}).get("display_name", "") for t in topics)
    topics_domain   = "; ".join((t.get("domain")   or {}).get("display_name", "") for t in topics)

    # authors (raw names)
    authors = "; ".join(
        a.get("raw_author_name", "")
        for a in r.get("authorships", [])
    )

    # keywords
    keywords = "; ".join(
        k.get("display_name", "") for k in r.get("keywords", [])
    )

    # referenced works (list of OpenAlex IDs → pipe-separated string)
    ref_works = "|".join(r.get("referenced_works", []))

    return {
        "id":                          r.get("id", ""),
        "doi":                         r.get("doi", ""),
        "title":                       r.get("title", ""),
        "publication_year":            r.get("publication_year", ""),
        "language":                    r.get("language", ""),
        "type":                        r.get("type", ""),
        "cited_by_count":              r.get("cited_by_count", ""),
        "referenced_works_count":      r.get("referenced_works_count", ""),
        "referenced_works":            ref_works,
        "primary_topic":               pt.get("display_name", ""),
        "primary_topic.subfield":      pt_subfield.get("display_name", ""),
        "primary_topic.domain":        pt_domain.get("display_name", ""),
        "topics":                      topics_name,
        "topics.subfield":             topics_subfield,
        "topics.field":                topics_field,
        "topics.domain":               topics_domain,
        "authors":                     authors,
        "keywords":                    keywords,
        "abstract":                    reconstruct_abstract(r.get("abstract_inverted_index")),
    }


def fetch_all(api_key: str) -> list[dict]:
    """Page through all results using cursor pagination."""
    params = {
        "filter":   f"title_and_abstract.search:{SEARCH_STR},type:{TYPE_FILTER}",
        "select":   SELECT_FIELDS,
        "sort":     "relevance_score:desc",
        "per_page": 200,
        "cursor":   "*",
        "include_xpac":  "true",
        "api_key":  api_key,
    }
    all_records = []
    page = 1

    while True:
        resp = requests.get("https://api.openalex.org/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        meta    = data.get("meta", {})
        results = data.get("results", [])

        if page == 1:
            print(f"Total results: {meta.get('count', '?')}")

        if not results:
            break

        all_records.extend(flatten_record(r) for r in results)
        print(f"  Page {page} – fetched {len(all_records)} records so far...")

        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break

        params["cursor"] = next_cursor
        page += 1
        time.sleep(0.1)   # stay well within rate limits

    return all_records


def save_csv(records: list[dict], path: str) -> None:
    if not records:
        print("No records to save.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records → {path}")


if __name__ == "__main__":
    records = fetch_all(API_KEY)
    save_csv(records, OUTPUT_FILE)
# Citation Pipeline

This document covers the full citation corpus management workflow. Records can be retrieved in two ways: directly from the OpenAlex API, or by parsing citation exports from PubMed and reference managers. Both paths converge at the merge and deduplication steps.

## Overview

```
Option A: revu fetch-openalex      Option B: .nbib / .ris files
               |                                   |
               |                              revu parse
               |                                   |
               └───────────────┬──────────────────┘
                               |
                         revu merge          (if multiple sources)
                               |
                         revu deduplicate
                               |
                           dedup.csv
```

---

## Option A: Retrieve records from OpenAlex

OpenAlex is a free, open bibliographic database. The `revu fetch-openalex` command queries the API directly using a keyword search and retrieves metadata including abstracts, author affiliations, citation counts, and referenced works — fields that are not always available in citation exports.

```bash
revu fetch-openalex \
  --query '("machine learning" OR "deep learning") AND "systematic review"' \
  --output data/openalex_results.csv \
  --api-key YOUR_API_KEY \
  --from-date 2010-01-01 \
  --to-date 2025-12-31
```

The API key can also be set via the `OPENALEX_API_KEY` environment variable:

```bash
export OPENALEX_API_KEY=your_key_here
revu fetch-openalex \
  --query '("machine learning" OR "deep learning") AND "systematic review"' \
  --output data/openalex_results.csv
```

The command paginates through all matching results and saves a CSV with columns: `id`, `doi`, `title`, `publication_year`, `language`, `type`, `cited_by_count`, `referenced_works_count`, `referenced_works`, `primary_topic`, `primary_topic.subfield`, `primary_topic.domain`, `topics`, `authors`, `keywords`, `abstract`.

The `referenced_works` column (pipe-separated OpenAlex IDs) is used by `revu biblio build-network` for citation network construction.

**When to use OpenAlex retrieval:** When you need richer metadata than is available in citation exports (author affiliations, citation counts, reference lists), or when you want to avoid the manual export step. OpenAlex results can also be merged with citation exports to combine multiple sources.

---

## Option B: Parse citation exports

### Supported input formats

| Format | Source | Notes |
|---|---|---|
| `.nbib` | PubMed | Direct export from PubMed search results |
| `.ris` | Zotero, Endnote, EBSCO, Web of Science | EBSCO exports require normalization (see below) |

### Step 1: Parse citation files

Place all citation files of the same format in one directory, then run:

```bash
# PubMed NBIB
revu parse nbib data/raw/nbib/ data/parsed/ pubmed.csv

# RIS (Zotero, Endnote, etc.)
revu parse ris data/raw/ris/ data/parsed/ ris_citations.csv
```

Each file in the input folder is parsed and all records are combined into the output CSV.

**Output columns:** `id`, `doi`, `au` (authors), `ti` (title), `dp` (date), `jt` (journal), `ab` (abstract).

### EBSCO exports

RIS files exported from EBSCO use a non-standard date format that must be normalized before parsing:

```bash
revu preprocess data/raw/ebsco_ris/ data/raw/ebsco_ris_normalized/
revu parse ris data/raw/ebsco_ris_normalized/ data/parsed/ ebsco.csv
```

---

## Merge from multiple sources

If you have exports from multiple databases (or a mix of OpenAlex and citation exports), merge them before deduplication:

```bash
revu merge -i "data/parsed/*.csv" -o data/merged.csv
```

The `-i` option accepts a glob pattern. Alternatively, specify files individually:

```bash
revu merge -i data/openalex_results.csv -i data/parsed/pubmed.csv -o data/merged.csv
```

## Deduplicate

```bash
revu deduplicate \
  -i data/merged.csv \
  -o data/dedup.csv \
  -l data/dedup.log
```

The deduplication logic:

1. Groups records by DOI (case-insensitive). Among duplicates, retains the record with the most complete metadata.
2. For remaining duplicates without DOIs, groups by normalized title.
3. Drops non-English records.
4. Among records with the same DOI or title, prefers records that have an abstract.

The log file records every removed record and the reason for removal.

## Output

The deduplicated CSV is the starting point for all downstream analysis:

- Feed into `revu preprocess-texts` for topic modeling
- Feed into `revu biblio analyze` for bibliometric analysis

## Tips

**Working with very large exports:** If a single database export produces multiple files (e.g. PubMed limits bulk exports to 10,000 records), parse each file separately and merge them all in one step.

**Checking deduplication results:** Compare row counts at each step. A significant reduction at the merge-to-deduplicate step indicates substantial overlap between databases, which is expected for well-covered fields.

**Record IDs:** The `id` column assigned at parse or fetch time is used to link records across downstream files (topic assignments, metadata, visualizations). Do not modify it after this step.

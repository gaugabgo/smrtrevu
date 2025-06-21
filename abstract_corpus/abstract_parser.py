import os
import csv
import glob
import re
import rispy
import nbib
from typing import List


def parse_nbib_folder(input_folder: str, output_csv_path: str) -> None:
    """Parse all .nbib files in a folder into a single CSV."""
    fieldnames = ["DOI", "AU", "TI", "DP", "JT", "AB"]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in glob.glob(os.path.join(input_folder, "*.nbib")):
            filename = os.path.basename(filepath)
            print(f"📄 Parsing file: {filename}")

            try:
                with open(filepath, "r", encoding="utf-8") as nbib_file:
                    file_content = nbib_file.read()

                references = nbib.read(file_content)
                print(f"🔍 Found {len(references)} entries")

                for ref in references:
                    title = ref.get("title", "No Title")
                    authors = ref.get("authors", [])
                    author_names = []

                    for author in authors:
                        first = author.get("first_name", "")
                        last = author.get("last_name", "")
                        full_name = f"{first} {last}".strip()
                        author_names.append(full_name if full_name else "Unknown Author")

                    authors_str = "; ".join(author_names) if author_names else "No Authors"

                    date = ref.get("publication_date", "")
                    year = re.search(r"\b(19|20)\d{2}\b", date)
                    year_only = year.group(0) if year else "No Year Found"

                    writer.writerow({
                        "AU": authors_str,
                        "TI": title,
                        "DP": year_only,
                        "JT": ref.get("journal", "No Journal"),
                        "AB": ref.get("abstract", "No Abstract"),
                        "DOI": ref.get("doi", "No DOI")
                    })

                print(f"✅ Finished parsing: {filename}\n")

            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}\n")

    print(f"📁 All .nbib files parsed. CSV saved to: {output_csv_path}")


def parse_ris_folder(input_folder: str, output_csv_path: str) -> None:
    """Parse all .ris files in a folder into a single CSV."""
    fieldnames = ["DOI", "AU", "TI", "PY", "T2", "AB"]

    def extract_year(raw: str) -> str:
        match = re.search(r"\d{4}", raw or "")
        return match.group(0) if match else "No Year"

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in glob.glob(os.path.join(input_folder, "*.ris")):
            filename = os.path.basename(filepath)
            print(f"📄 Parsing file: {filename}")

            try:
                with open(filepath, "r", encoding="utf-8") as ris_file:
                    entries = rispy.load(ris_file)

                print(f"🔍 Found {len(entries)} entries")

                for entry in entries:
                    authors = entry.get("authors", [])
                    authors_str = "; ".join(authors) if authors else "No Authors"

                    year = extract_year(
                        entry.get("year") or entry.get("y1") or entry.get("da") or entry.get("dp") or ""
                    )

                    writer.writerow({
                        "DOI": entry.get("doi", "No DOI"),
                        "AU": authors_str,
                        "TI": entry.get("title") or entry.get("primary_title") or entry.get("t1") or "No Title",
                        "PY": year,
                        "T2": entry.get("journal_name") or entry.get("jo") or entry.get("secondary_title") or "No Journal",
                        "AB": entry.get("abstract") or entry.get("n2") or "No Abstract"
                    })

                print(f"✅ Finished parsing: {filename}\n")

            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}\n")

    print(f"📁 All .ris files parsed. CSV saved to: {output_csv_path}")

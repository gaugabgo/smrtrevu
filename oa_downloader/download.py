# doi_fetcher.py
import os, time, requests, csv

def load_dois(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def fetch_metadata(doi, email):
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        response = requests.get(api_url, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Error fetching {doi}: {e}")
        return None

def download_article(doi, location, allowed_licenses, save_dir):
    license = location.get("license", "").lower()
    pdf_url = location.get("url_for_pdf")
    html_url = location.get("url")
    if license and license not in allowed_licenses:
        return False

    safe_doi = doi.replace("/", "_")
    filepath = os.path.join(save_dir, f"{safe_doi}.pdf")

    if pdf_url:
        try:
            r = requests.get(pdf_url, stream=True, timeout=15)
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                return True
        except Exception:
            pass

    if html_url:
        try:
            r = requests.get(html_url, timeout=15)
            if r.status_code == 200:
                html_path = os.path.join(save_dir, f"{safe_doi}.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                return True
        except Exception:
            pass
    return False

def download_oa_articles(input_file, output_file, pdf_dir, email, rate_limit, allowed_licenses):
    """
    Batch download OA articles using Unpaywall metadata.
    Saves metadata to a CSV and PDFs/HTMLs to `pdf_dir`.
    """
    # Read DOIs from file
    with open(input_file, 'r') as f:
        dois = [line.strip() for line in f if line.strip()]

    headers = {"User-Agent": f"mailto:{email}"}
    metadata_records = []

    for doi in dois:
        api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[WARN] DOI {doi} — Unpaywall returned {response.status_code}")
                continue

            data = response.json()
            best_oa_location = data.get("best_oa_location")
            if not best_oa_location:
                print(f"[INFO] DOI {doi} — No OA location found")
                continue

            success = download_article(
                doi=doi,
                location=best_oa_location,
                allowed_licenses=allowed_licenses,
                save_dir=pdf_dir
            )

            if success:
                print(f"[✓] Downloaded {doi}")
                metadata_records.append({
                    "doi": doi,
                    "title": data.get("title"),
                    "oa_url": best_oa_location.get("url_for_pdf") or best_oa_location.get("url"),
                    "license": best_oa_location.get("license", "").lower(),
                    "is_oa": True
                })
            else:
                print(f"[X] Failed to download {doi}")

            time.sleep(rate_limit)

        except Exception as e:
            print(f"[ERROR] DOI {doi} — {e}")

    if metadata_records:
        df = pd.DataFrame(metadata_records)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"[✓] Saved metadata to {output_file}")
    else:
        print("[!] No metadata saved — no successful downloads")
        
def fetch_all_dois(dois, email, allowed_licenses, save_dir, rate_limit=5.0):
    os.makedirs(save_dir, exist_ok=True)
    results = []
    failed = []

    for doi in dois:
        time.sleep(rate_limit)
        data = fetch_metadata(doi, email)
        if not data:
            failed.append(doi)
            continue

        is_oa = data.get("is_oa", False)
        location = data.get("best_oa_location") or next((loc for loc in data.get("oa_locations", []) if loc.get("url_for_pdf")), None)

        result = {
            "DOI": doi,
            "is_oa": is_oa,
            "license": location.get("license") if location else None,
            "pdf_url": location.get("url_for_pdf") if location else None,
            "html_url": location.get("url") if location else None
        }
        results.append(result)

        if is_oa and location:
            success = download_article(doi, location, allowed_licenses, save_dir)
            if not success:
                failed.append(doi)
        else:
            failed.append(doi)
    return results, failed

def save_results_csv(results, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

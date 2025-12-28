import requests
import typings
import json
import glob
import re
from pathlib import Path
import pandas as pd
import os

class OaMetadataFetcher:

    def __init__(self, cache_dir:str, email:str):
        self.email = email
        self.cache_dir = cache_dir

        self.cache_dir_path = Path(cache_dir)
        self.json_dir_path = self.cache_dir_path / "json"
        self.json_dir_path.mkdir(parents=True, exist_ok=True)
        
    def build_metdata_table(self):
        json_files_path = self.json_dir_path / '*.json'
        all_dataframes = []
        for file_path in glob.glob(str(json_files_path)):
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                all_dataframes.append(self._convert_metadata_csv(file_path, json_data))
            except Exception as e:
                raise Exception(f"Exception while processing doi. Filepath: {file_path}, Exception: {e}")


        combined_df = pd.DataFrame.from_records(all_dataframes)
        combined_df.to_csv(self.cache_dir_path / "metadata.csv")
        return combined_df


    def fetch_all_metadata(self, dois: list[str]) -> list[dict]:

        results = []

        for doi in dois:
            result, cache_miss = self.get_or_fetch_metadata(doi)
            print(f"Cache miss: {cache_miss}")
            results.append(result)
        
        return results
    
    def get_or_fetch_metadata(self, doi: str) -> dict:
        filename = self._create_filename_from_doi(doi)
        filepath = self.cache_dir_path / 'json' / filename

        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f), False
        else:
            result = self.fetch_one_metadata(doi)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            return result, True
        
    def fetch_one_metadata(self, doi: str) -> dict:
        api_url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}!"
        headers = {"User-Agent": f"mailto:{self.email}"}
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            status_code = response.status_code

            if status_code == 200:
                metadata = response.json()
                return {
                    'status_code': status_code,
                    'metadata': metadata
                }
            elif status_code == 429:
                raise Exception("Rate limit exceeded")
            return {
                'status_code': status_code,
                'metadata': None
            }
        except Exception as e:
            print(f"Error fetching {doi}: {e}")
            return None

    def _create_filename_from_doi(self, doi):
        clean_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
        safe_filename = re.sub(r'[^\w\-_.]', '_', clean_doi)
        safe_filename = re.sub(r'_+', '_', safe_filename)
        safe_filename = safe_filename.strip('_')
        return f"{safe_filename}.json"
    
    def _convert_metadata_csv(self, file_path, json_data):
        def safe_get(d, key, default=''):
            return d.get(key, default) if isinstance(d, dict) else default

        metadata = safe_get(json_data, 'metadata', {})

        top_level_columns = {
            'filename': Path(file_path).name,
            'status_code': safe_get(json_data, 'status_code', ''),
            'doi': safe_get(metadata, 'doi', ''),
            'doi_url': safe_get(metadata, 'doi_url', ''),
            'date': safe_get(metadata, 'date', ''),
            'title': safe_get(metadata, 'title', ''),
            'year': safe_get(metadata, 'year', ''),
            'journal_is_oa': safe_get(metadata, 'journal_is_oa', ''),
            'is_oa': safe_get(metadata, 'is_oa', ''),
        }

        oa_columns = {
            'oa_status': '',
            'oa_pdf_link': '',
            'oa_license': '',
            'oa_version': '',
        }

        if top_level_columns['is_oa']:
            best_oa = safe_get(metadata, 'best_oa_location', {})
            oa_columns.update({
                'oa_status': safe_get(metadata, 'oa_status', ''),
                'oa_pdf_link': safe_get(best_oa, 'url_for_pdf', ''),
                'oa_license': safe_get(best_oa, 'license', ''),
                'oa_version': safe_get(best_oa, 'version', ''),
            })

        return {**top_level_columns, **oa_columns}
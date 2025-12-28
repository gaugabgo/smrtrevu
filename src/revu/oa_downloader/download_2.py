from pathlib import Path
import pandas as pd
import time

from revu.oa_downloader.single_pdf_downloader import CurlDownloader, SeleniumDownloader, RequestsDownloader

"""
Download open access articles based on oa_pdf_link from Unpaywall API, from oa_metadata_getter.py
"""

class OaDownloader:
    def __init__(self, cache_dir: str, allowed_licenses, download_method="curl"):
        self.cache_dir = cache_dir
        self.cache_dir_path = Path(cache_dir)
        self.pdf_dir_path = self.cache_dir_path
        self.pdf_dir_path.mkdir(parents=True, exist_ok=True)
        self.df_metadata = None
        self.result = {}
        self.allowed_licenses = allowed_licenses  
        self.cached_filenames = {f.name for f in self.cache_dir_path.iterdir() if f.is_file()}
        if download_method == "curl":
            self.downloader = CurlDownloader()
        elif download_method == "selenium":
            self.downloader = SeleniumDownloader(self.pdf_dir_path)
        elif download_method == "requests":
            self.downloader = RequestsDownloader()
        else:
            raise ValueError(f"Unknown download method: {download_method}")

    def get_most_recent_file(self):
        try:
            files = list(self.cache_dir_path.iterdir())
            files = [f for f in files if f.is_file()]
            
            if not files:
                return None
            
            return max(files, key=lambda f: f.stat().st_mtime)
            
        except Exception as e:
            print(f"Error finding recent file: {e}")
            return None
    
    def load_metadata(self, input):
        self.df_metadata = pd.read_csv(input, na_values=[""])
    
    def run(self, input):
        self.load_metadata(input)
        results = self.download_all_oa_pdfs(self.allowed_licenses)
        return results
    
    def download_all_oa_pdfs(self, allowed_licenses):
        result_list = []

        for i, row in self.df_metadata.iterrows():
            self.result = {
                'doi': None,
                'is_oa': False,
                'oa_license_allowed': False,
                'link_for_pdf': False,
                'pdf_name': None,
                'pdf_exists': False,
                'pdf_downloaded': False,
                'status_code': None,
                'error': None,
            }

            print(f"\nProcessing row {i} - DOI: {row.get('doi', 'N/A')}")

            self.result['doi'] = row.get('doi')
            self._check_oa_status(i, row, allowed_licenses)
            result_list.append(self.result)

        df_pdf_results = pd.DataFrame(result_list)
        df_pdf_results.to_csv(self.cache_dir_path/"pdf_downloads.csv", index=False)
        return df_pdf_results
    
    def _check_oa_status(self, index, row, allowed_licenses):
        if row['is_oa'] == True:
            self.result['is_oa'] = f"Article is Open Access"
            print("Article is open access, passing to next function.")
            self._check_license(index, row, allowed_licenses)
        else:
            self.result['is_oa'] = f"Article is NOT Open Access"
            print("Row is FALSE for is_oa, skipping.")

    def _check_license(self, index, row, allowed_licenses):
        license_name = row['oa_license']

        if license_name not in allowed_licenses:
            self.result['oa_license_allowed'] = f"Article is not under an allowed license: {license_name}"
            print("Article not under allowed license")
        else:
            self.result['oa_license_allowed'] = f"Article is under an allowed license: {license_name}"
            print("Article under allowed license")
            self._check_link(index, row)

    def _check_link(self, index, row): 
        link = row['oa_pdf_link']  
        if pd.isna(link):
            self.result['link_for_pdf'] = f"Article does not have a pdf link"
            print("No link to retrieve pdf article")
        else:
            self.result['link_for_pdf'] = f"Article has a pdf link: {link}"
            print("PDF article has a link")
            self._name_and_get_pdf(index, row)

    def _name_and_get_pdf(self, index, row):
        filename = row['filename'][:-5] + ".pdf"
        
        self.get_or_fetch_pdf(index, row, filename)
    
    def get_or_fetch_pdf(self, index, row, filename):
        pdf_dir = Path(self.cache_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)  
        print(f"Checking if {filename} in cached_filenames")
        
        if filename in self.cached_filenames:
            print(self.cached_filenames)
            print(f"Exact match found? {filename in self.cached_filenames}")
            self.result['pdf_name'] = f"{filename}"
            self.result['pdf_exists'] = True
            self.result['pdf_downloaded'] = False  
            print(f"PDF already exists: {pdf_dir}")
            return filename, False
        else:
            pdf_link = row['oa_pdf_link']  
            self.result['pdf_name'] = f"{filename}"
            print(f"Downloading PDF from: {pdf_link}")
            
            download_success = self.download_oa_pdf(index, pdf_link, filename)
            if download_success:
                self.result['pdf_exists'] = True
                self.result['pdf_downloaded'] = True
                print(f"Successfully downloaded: {filename}")
                return filename, True
            else:
                self.result['pdf_exists'] = False
                self.result['pdf_downloaded'] = False
                print("Download failed; no file saved.")
                return None, False

    def download_oa_pdf(self, index, pdf_link: str, filename):
        output_path = self.cache_dir_path / filename
        
        if self.downloader.download(pdf_link, output_path):
            self.cached_filenames.add(filename)
            return True
        else:
            return False

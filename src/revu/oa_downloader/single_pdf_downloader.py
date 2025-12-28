import subprocess
from pathlib import Path
import time
import random
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import pandas as pd
import os
import logging
import time

class ResourceDownloader:
    def download(self, url: str, output_path: Path) -> bool:
        raise NotImplementedError

class CurlDownloader(ResourceDownloader):
    def download(self, url: str, output_path: Path) -> bool:
        try:
            time.sleep(random.uniform(2, 5))
            result = subprocess.run([
                'curl', '-L', '-o', str(output_path), url
            ], timeout=60, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                print(f"Successfully downloaded with curl: {output_path.name}")
                return True
            else:
                print(f"Curl download failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error during curl download: {e}")
            return False

class SeleniumDownloader(ResourceDownloader):
    def __init__(self, download_dir: Path):
        self.download_dir = download_dir
    
    def download(self, url: str, output_path: Path) -> bool:
        print(f"Starting selenium download: {url}")
        print(f"Target: {output_path}")
        
        options = Options()
        prefs = {
            "download.default_directory": str(self.download_dir),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,
            "profile.default_content_setting_values.plugins": 1,
            "profile.managed_default_content_settings.images": 2
        }
        options.add_experimental_option("prefs", prefs)
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins-discovery")
        
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(60)
            
            time.sleep(random.uniform(2, 5))
            
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                print(f"URL returned status code {response.status_code}")
                return False
            
            download_start_time = time.time()
            initial_files = set(f.name for f in self.download_dir.glob("*.pdf"))
            
            driver.get("about:blank")
            
            download_script = f"""
            var link = document.createElement('a');
            link.href = '{url}';
            link.download = '{output_path.name}';
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            driver.execute_script(download_script)
            
            try:
                driver.get(url)
            except Exception as nav_error:
                print(f"Direct navigation failed: {nav_error}")
            
            max_wait = 60
            wait_interval = 2
            waited = 0
            
            while waited < max_wait:
                time.sleep(wait_interval)
                waited += wait_interval
                
                current_pdf_files = set(f.name for f in self.download_dir.glob("*.pdf") 
                                    if f.stat().st_mtime > download_start_time)
                new_pdf_files = current_pdf_files - initial_files
                
                if new_pdf_files:
                    break
                    
                partial_files = list(self.download_dir.glob("*.crdownload"))
                partial_files.extend(self.download_dir.glob("*.tmp"))
                
                if partial_files:
                    continue
                    
                if waited % 10 == 0:
                    print(f"Still waiting... ({waited}s)")
            
            recent_file = self._get_most_recent_file()
            
            if recent_file:
                recent_file.rename(output_path)
                print("Download successful!")
                return True
            else:
                print("No file found after download")
                return False
                
        except Exception as e:
            print(f"Error during selenium download: {e}")
            return False
        finally:
            if driver:
                driver.quit()
    
    def _get_most_recent_file(self):
        try:
            files = [f for f in self.download_dir.glob("*") 
                    if f.is_file() and f.suffix.lower() == '.pdf' 
                    and 'log' not in f.name.lower() and '.csv' not in f.name.lower()]
            
            if not files:
                return None
            
            most_recent = max(files, key=lambda x: x.stat().st_mtime)
            file_age = time.time() - most_recent.stat().st_mtime
            
            if file_age < 60:
                return most_recent
            else:
                return None
                
        except Exception as e:
            print(f"Error finding recent file: {e}")
            return None

class RequestsDownloader(ResourceDownloader):
    def download(self, url: str, output_path: Path) -> bool:
        try:
            time.sleep(random.uniform(2, 5))
            response = requests.get(url, allow_redirects=True, timeout=60)
            response.raise_for_status() 
            with open(output_path, 'wb') as f:
                f.write(response.content)
                return True
            
        except Exception as e:
            print(f"Error during requests download: {e}")
            return False
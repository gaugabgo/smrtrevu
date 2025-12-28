import pandas as pd
from bs4 import BeautifulSoup
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

class TEIXMLParser:
    
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logging(log_level)
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'successful_files': [],
            'failed_files': [],
            'total_content_items': 0
        }
        
    def _setup_logging(self, log_level):
        """Setup logging configuration."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger('TEIXMLParser')
        logger.setLevel(log_level)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'tei_parser_{timestamp}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
        
    def parse_xml_file(self, xml_file_path: str) -> Optional[Dict]:
        """Parse TEI XML using BeautifulSoup."""
        filename = Path(xml_file_path).name
        self.logger.info(f"Starting to parse: {filename}")
        
        try:
            if not Path(xml_file_path).exists():
                raise FileNotFoundError(f"File not found: {xml_file_path}")
            
            file_size = Path(xml_file_path).stat().st_size
            if file_size == 0:
                self.logger.warning(f"File {filename} is empty")
                return None
            elif file_size > 50 * 1024 * 1024:  # 50MB
                self.logger.warning(f"File {filename} is very large ({file_size/1024/1024:.1f}MB)")
            
            with open(xml_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if not content.strip():
                self.logger.warning(f"File {filename} has no content after reading")
                return None
            
            soup = BeautifulSoup(content, 'xml')
            
            if not soup or not soup.find():
                self.logger.error(f"Failed to parse XML structure in {filename}")
                return None
            
            self.logger.debug(f"Parsing metadata for {filename}")
            title = self._extract_title(soup)
            pub_date = self._extract_pub_date(soup)
            authors = self._extract_authors(soup)
            
            self.logger.debug(f"Extracting body content for {filename}")
            body_content = self._extract_body_content(soup)
            
            result = {
                'title': title,
                'pub_date': pub_date,
                'authors': authors,
                'body_content': body_content
            }
            
            self.logger.info(f"Successfully parsed {filename}: "
                           f"title='{title[:50]}...', "
                           f"authors={len(authors)}, "
                           f"content_items={len(body_content)}")
            
            self.processing_stats['files_processed'] += 1
            self.processing_stats['successful_files'].append(filename)
            self.processing_stats['total_content_items'] += len(body_content)
            
            return result
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filename} - {str(e)}")
            self._record_failure(filename, str(e))
            return None
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error in {filename}: {str(e)}")
            self._record_failure(filename, f"Encoding error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing {filename}: {str(e)}", exc_info=True)
            self._record_failure(filename, f"Unexpected error: {str(e)}")
            return None
    
    def _record_failure(self, filename: str, error_msg: str):
        """Record a failed file processing."""
        self.processing_stats['files_failed'] += 1
        self.processing_stats['failed_files'].append({
            'filename': filename,
            'error': error_msg
        })
    
    def _extract_title(self, soup) -> str:
        """Extract article title."""
        try:
            # Look for title in fileDesc/titleStmt/title with level="a"
            title_elem = soup.find('title', {'level': 'a'})
            
            # Fallback to any title in titleStmt
            if not title_elem:
                title_stmt = soup.find('titleStmt')
                if title_stmt:
                    title_elem = title_stmt.find('title')
            
            # Last fallback to any title
            if not title_elem:
                title_elem = soup.find('title')
            
            title = title_elem.get_text().strip() if title_elem else "Unknown Title"
            
            if title == "Unknown Title":
                self.logger.warning("Could not extract title, using 'Unknown Title'")
            
            return title
            
        except Exception as e:
            self.logger.warning(f"Error extracting title: {str(e)}")
            return "Unknown Title"
    
    def _extract_pub_date(self, soup) -> str:
        """Extract publication date."""
        try:
            # Look for date in publicationStmt
            pub_stmt = soup.find('publicationStmt')
            if pub_stmt:
                date_elem = pub_stmt.find('date')
                if date_elem:
                    date_value = date_elem.get('when') or date_elem.get_text().strip()
                    if date_value:
                        return date_value
            
            # Additional fallback locations for dates
            for date_elem in soup.find_all('date'):
                date_value = date_elem.get('when') or date_elem.get_text().strip()
                if date_value:
                    self.logger.debug(f"Found date in alternative location: {date_value}")
                    return date_value
            
            self.logger.warning("Could not extract publication date")
            return "Unknown Date"
            
        except Exception as e:
            self.logger.warning(f"Error extracting publication date: {str(e)}")
            return "Unknown Date"
    
    def _extract_authors(self, soup) -> List[str]:
        """Extract authors."""
        authors = []
        
        try:
            author_elems = soup.find_all('author')
            
            for author in author_elems:
                author_name = self._extract_author_name(author)
                if author_name and author_name.strip():
                    authors.append(author_name.strip())
            
            if not authors:
                self.logger.warning("No authors found")
                return ["Unknown Author"]
                
            self.logger.debug(f"Extracted {len(authors)} authors: {authors}")
            return authors
            
        except Exception as e:
            self.logger.warning(f"Error extracting authors: {str(e)}")
            return ["Unknown Author"]
    
    def _extract_author_name(self, author_elem) -> Optional[str]:
        """Extract individual author name from author element."""
        try:
            persname = author_elem.find('persName')
            if persname:
                forename = persname.find('forename')
                surname = persname.find('surname')
                
                name_parts = []
                if forename:
                    forename_text = forename.get_text().strip()
                    if forename_text:
                        name_parts.append(forename_text)
                if surname:
                    surname_text = surname.get_text().strip()
                    if surname_text:
                        name_parts.append(surname_text)
                
                if name_parts:
                    return ' '.join(name_parts)
                else:
                    # Fallback to persName text content
                    persname_text = persname.get_text().strip()
                    return persname_text if persname_text else None
            
            # Fallback to author element text content
            author_text = author_elem.get_text().strip()
            return author_text if author_text else None
            
        except Exception as e:
            self.logger.debug(f"Error extracting individual author name: {str(e)}")
            return None
    
    def _extract_body_content(self, soup) -> List[Dict]:
        """Extract body content from the document."""
        body_content = []
        
        try:
            body = soup.find('body')
            if not body:
                self.logger.warning("No body element found in document")
                return body_content
            
            # Process all top-level divs in the body
            top_level_divs = body.find_all('div', recursive=False)
            
            if not top_level_divs:
                self.logger.warning("No div elements found in body")
                paragraphs = body.find_all('p', recursive=False)
                if paragraphs:
                    self.logger.info(f"Found {len(paragraphs)} direct paragraphs in body")
                    paragraph_texts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                    if paragraph_texts:
                        concatenated_text = ' '.join(paragraph_texts)
                        body_content.append({
                            'parent_header': None,
                            'content_type': "section_text",
                            'content': concatenated_text 
                        })
            else:
                for i, div in enumerate(top_level_divs):
                    self.logger.debug(f"Processing div {i+1}/{len(top_level_divs)}")
                    self._process_div(div, body_content, parent_header=None)
            
            self.logger.debug(f"Extracted {len(body_content)} content items from body")
            return body_content
            
        except Exception as e:
            self.logger.error(f"Error extracting body content: {str(e)}")
            return []
    
    def _process_div(self, div, body_content: List[Dict], parent_header: Optional[str] = None):
        """Process a div element and extract its content."""
        try:
            # Extract header from this div (direct child only)
            head = div.find('head', recursive=False)
            current_header = head.get_text().strip() if head and head.get_text().strip() else None

            effective_parent = current_header if current_header else parent_header

            # Extract all paragraphs in this div
            all_paragraphs = div.find_all('p', recursive=True)
            
            if all_paragraphs:
                paragraph_texts = [p.get_text().strip() for p in all_paragraphs if p.get_text().strip()]
                
                if paragraph_texts:
                    concatenated_text = ' '.join(paragraph_texts)
                    if concatenated_text.strip():  
                        body_content.append({
                            'parent_header': current_header,
                            'content_type': "section_text",
                            'content': concatenated_text.strip()
                        })
            
            # Process other text elements (e.g., lists)
            self._process_other_elements(div, body_content, effective_parent)
            
        except Exception as e:
            self.logger.warning(f"Error processing div: {str(e)}")
    
    def _process_other_elements(self, div, body_content: List[Dict], parent_header: Optional[str]):
        """Process other elements like lists, figures, etc."""
        try:
            lists = div.find_all(['list'], recursive=False)
            for lst in lists:
                list_items = lst.find_all('item')
                for item in list_items:
                    text = item.get_text().strip()
                    if text:
                        body_content.append({
                            'parent_header': parent_header or "No Header",
                            'content_type': 'list_item',
                            'content': text
                        })
        except Exception as e:
            self.logger.warning(f"Error processing other elements: {str(e)}")
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def save_processing_report(self, output_path: str = None):
        """Save a detailed processing report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"logs/processing_report_{timestamp}.txt"
        
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("TEI XML Processing Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            stats = self.processing_stats
            f.write(f"Files Successfully Processed: {stats['files_processed']}\n")
            f.write(f"Files Failed: {stats['files_failed']}\n")
            f.write(f"Total Content Items Extracted: {stats['total_content_items']}\n\n")
            
            if stats['successful_files']:
                f.write("Successfully Processed Files:\n")
                f.write("-" * 30 + "\n")
                for filename in stats['successful_files']:
                    f.write(f"  ✓ {filename}\n")
                f.write("\n")
            
            if stats['failed_files']:
                f.write("Failed Files:\n")
                f.write("-" * 15 + "\n")
                for failure in stats['failed_files']:
                    f.write(f"  ✗ {failure['filename']}: {failure['error']}\n")
                f.write("\n")
        
        self.logger.info(f"Processing report saved to: {output_path}")

def process_tei_files(directory_path: str, log_level=logging.INFO) -> pd.DataFrame:
    """
    Process all TEI XML files in a directory and return a DataFrame.
    
    Args:
        directory_path: Path to directory containing TEI XML files
        log_level: Logging level (default: INFO)
    
    Returns:
        pandas.DataFrame with columns: filename, title, pub_date, authors, 
        parent_header, content_type, content
    """
    parser = TEIXMLParser(log_level=log_level)
    all_data = []
    
    dir_path = Path(directory_path)
    if not dir_path.exists():
        parser.logger.error(f"Directory does not exist: {directory_path}")
        return pd.DataFrame()
    
    if not dir_path.is_dir():
        parser.logger.error(f"Path is not a directory: {directory_path}")
        return pd.DataFrame()
    
    xml_files = list(dir_path.glob("*.xml"))
    
    if not xml_files:
        parser.logger.warning(f"No XML files found in {directory_path}")
        return pd.DataFrame()
    
    parser.logger.info(f"Found {len(xml_files)} XML files to process in {directory_path}")
    
    for xml_file in xml_files:
        parsed_data = parser.parse_xml_file(str(xml_file))
        
        if parsed_data:
            if parsed_data['body_content']:
                for content_item in parsed_data['body_content']:
                    all_data.append({
                        'filename': xml_file.name,
                        'title': parsed_data['title'],
                        'pub_date': parsed_data['pub_date'],
                        'authors': '; '.join(parsed_data['authors']), 
                        'parent_header': content_item['parent_header'],
                        'content_type': content_item['content_type'],
                        'content': content_item['content']
                    })
            else:
                parser.logger.warning(f"No body content extracted from {xml_file.name}")
                all_data.append({
                    'filename': xml_file.name,
                    'title': parsed_data['title'],
                    'pub_date': parsed_data['pub_date'],
                    'authors': '; '.join(parsed_data['authors']), 
                    'parent_header': None,
                    'content_type': 'none',
                    'content': ''
                })
    
    df = pd.DataFrame(all_data)
    parser.logger.info(f"Created DataFrame with {len(df)} rows from {len(xml_files)} files")
    parser.save_processing_report()
    
    stats = parser.get_processing_stats()
    parser.logger.info(f"Processing complete - Success: {stats['files_processed']}, "
                      f"Failed: {stats['files_failed']}, "
                      f"Total content items: {stats['total_content_items']}")
    
    return df

def save_results(df: pd.DataFrame, output_path: str = None):
    """Save DataFrame to CSV file."""
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"tei_extracted_data_{timestamp}.csv"
    
    try:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False

def display_summary(df: pd.DataFrame):
    """Display summary information about the extracted data."""
    if df.empty:
        print("No data extracted.")
        return
    
    print("=== EXTRACTION SUMMARY ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique files processed: {df['filename'].nunique()}")
    print(f"Unique articles: {df['title'].nunique()}")
    
    print(f"\nContent types:")
    print(df['content_type'].value_counts())
    
    print(f"\nFiles with content:")
    files_with_content = df[df['content_type'] != 'no_content']['filename'].nunique()
    print(f"  Files with body content: {files_with_content}")
    print(f"  Files without body content: {df['filename'].nunique() - files_with_content}")
    
    print(f"\nSample of extracted data:")
    print(df.head(10))


if __name__ == "__main__":
    xml_directory = "data/ft_output/pdf_xml_out"
    log_level = logging.INFO  # INFO or DEBUG 
    
    df = process_tei_files(xml_directory, log_level=log_level)
    
    if not df.empty:
        save_results(df)
        display_summary(df)
    else:
        print("No data was extracted. Check the logs for details.")

import requests
import pandas as pd
import time
from typing import List, Dict, Optional
import json

class CitationCounter:
    """
    Get citation counts using Crossref API with Semantic Scholar fallback
    """
    
    def __init__(self, email: str = "your-email@example.com"):
        """
        Initialize citation counter
        
        Args:
            email: Your email for Crossref API (polite usage)
        """
        self.session = requests.Session()
        self.email = email
        self.crossref_headers = {
            'User-Agent': f'Citation-Counter/1.0 (mailto:{email})'
        }
        
    def get_crossref_citation_count(self, doi: str) -> Dict:
        """
        Get citation count and metadata from Crossref API
        """
        clean_doi = doi.replace('https://doi.org/', '').replace('doi:', '').strip()
        url = f"https://api.crossref.org/works/{clean_doi}"
        
        try:
            time.sleep(0.1)  # Rate limiting - be polite to Crossref
            response = self.session.get(url, headers=self.crossref_headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                work = data.get('message', {})
                
                # Extract publication date
                pub_date = None
                if work.get('published-print'):
                    date_parts = work['published-print'].get('date-parts', [[]])[0]
                elif work.get('published-online'):
                    date_parts = work['published-online'].get('date-parts', [[]])[0]
                else:
                    date_parts = []
                
                if date_parts:
                    pub_date = date_parts[0] if len(date_parts) > 0 else None
                
                # Extract authors
                authors = []
                for author in work.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                return {
                    'doi': clean_doi,
                    'citation_count': work.get('is-referenced-by-count', 0),
                    'title': work.get('title', [''])[0] if work.get('title') else '',
                    'authors': authors,
                    'journal': work.get('container-title', [''])[0] if work.get('container-title') else '',
                    'publisher': work.get('publisher', ''),
                    'year': pub_date,
                    'type': work.get('type', ''),
                    'url': work.get('URL', ''),
                    'source': 'crossref',
                    'success': True
                }
            
            elif response.status_code == 404:
                return {
                    'doi': clean_doi,
                    'success': False,
                    'error': 'Not found in Crossref',
                    'source': 'crossref'
                }
            else:
                return {
                    'doi': clean_doi,
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'source': 'crossref'
                }
                
        except Exception as e:
            return {
                'doi': clean_doi,
                'success': False,
                'error': str(e),
                'source': 'crossref'
            }
    
    def get_semantic_scholar_citation_count(self, doi: str, title: str = None) -> Dict:
        """
        Get citation count from Semantic Scholar API
        Can search by DOI or fall back to title search
        """
        clean_doi = doi.replace('https://doi.org/', '').replace('doi:', '').strip()
        
        # Try DOI first
        search_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}"
        
        params = {
            'fields': 'title,authors,year,citationCount,influentialCitationCount,journal,publicationTypes,externalIds,url'
        }
        
        try:
            time.sleep(1.0)  # Rate limiting for Semantic Scholar
            response = self.session.get(search_url, params=params, timeout=30)
            
            if response.status_code == 200:
                paper = response.json()
                
                # Extract authors
                authors = []
                for author in paper.get('authors', []):
                    if author.get('name'):
                        authors.append(author['name'])
                
                # Extract journal info
                journal_info = paper.get('journal', {})
                journal = journal_info.get('name', '') if journal_info else ''
                
                return {
                    'doi': clean_doi,
                    'citation_count': paper.get('citationCount', 0),
                    'influential_citation_count': paper.get('influentialCitationCount', 0),
                    'title': paper.get('title', ''),
                    'authors': authors,
                    'journal': journal,
                    'year': paper.get('year'),
                    'semantic_scholar_id': paper.get('paperId', ''),
                    'url': paper.get('url', ''),
                    'source': 'semantic_scholar',
                    'success': True
                }
            
            elif response.status_code == 404:
                # Try title search if DOI search failed and title is provided
                if title and title.strip():
                    return self._search_semantic_scholar_by_title(title, clean_doi)
                else:
                    return {
                        'doi': clean_doi,
                        'success': False,
                        'error': 'Not found in Semantic Scholar',
                        'source': 'semantic_scholar'
                    }
            else:
                return {
                    'doi': clean_doi,
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'source': 'semantic_scholar'
                }
                
        except Exception as e:
            return {
                'doi': clean_doi,
                'success': False,
                'error': str(e),
                'source': 'semantic_scholar'
            }
    
    def _search_semantic_scholar_by_title(self, title: str, doi: str) -> Dict:
        """Search Semantic Scholar by title when DOI search fails"""
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            'query': title.strip(),
            'limit': 5,
            'fields': 'title,authors,year,citationCount,influentialCitationCount,journal,externalIds,url'
        }
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                
                if papers:
                    # Take the first result (most relevant)
                    paper = papers[0]
                    
                    authors = []
                    for author in paper.get('authors', []):
                        if author.get('name'):
                            authors.append(author['name'])
                    
                    journal_info = paper.get('journal', {})
                    journal = journal_info.get('name', '') if journal_info else ''
                    
                    return {
                        'doi': doi,
                        'citation_count': paper.get('citationCount', 0),
                        'influential_citation_count': paper.get('influentialCitationCount', 0),
                        'title': paper.get('title', ''),
                        'authors': authors,
                        'journal': journal,
                        'year': paper.get('year'),
                        'semantic_scholar_id': paper.get('paperId', ''),
                        'url': paper.get('url', ''),
                        'source': 'semantic_scholar_title_search',
                        'success': True,
                        'note': 'Found via title search'
                    }
                else:
                    return {
                        'doi': doi,
                        'success': False,
                        'error': 'No results from title search',
                        'source': 'semantic_scholar'
                    }
            else:
                return {
                    'doi': doi,
                    'success': False,
                    'error': f'Title search failed: HTTP {response.status_code}',
                    'source': 'semantic_scholar'
                }
                
        except Exception as e:
            return {
                'doi': doi,
                'success': False,
                'error': f'Title search error: {str(e)}',
                'source': 'semantic_scholar'
            }
    
    def get_citation_count_with_fallback(self, doi: str) -> Dict:
        """
        Get citation count using Crossref first, Semantic Scholar as fallback
        """
        print(f"Processing DOI: {doi}")
        
        # Try Crossref first
        print("  Trying Crossref...")
        crossref_result = self.get_crossref_citation_count(doi)
        
        if crossref_result.get('success'):
            print(f"  ✓ Crossref: {crossref_result.get('citation_count', 0)} citations")
            
            # Also try Semantic Scholar for additional info (influential citations)
            print("  Getting Semantic Scholar data for comparison...")
            ss_result = self.get_semantic_scholar_citation_count(
                doi, 
                crossref_result.get('title', '')
            )
            
            # Combine results
            result = crossref_result.copy()
            if ss_result.get('success'):
                result['semantic_scholar_citations'] = ss_result.get('citation_count', 0)
                result['influential_citations'] = ss_result.get('influential_citation_count', 0)
                result['semantic_scholar_id'] = ss_result.get('semantic_scholar_id', '')
                print(f"  ✓ Semantic Scholar: {ss_result.get('citation_count', 0)} citations ({ss_result.get('influential_citation_count', 0)} influential)")
            else:
                result['semantic_scholar_citations'] = 0
                result['influential_citations'] = 0
                result['semantic_scholar_id'] = ''
                print(f"  ✗ Semantic Scholar: {ss_result.get('error', 'Unknown error')}")
            
            result['primary_source'] = 'crossref'
            return result
        
        else:
            print(f"  ✗ Crossref failed: {crossref_result.get('error', 'Unknown error')}")
            print("  Trying Semantic Scholar as fallback...")
            
            ss_result = self.get_semantic_scholar_citation_count(doi)
            
            if ss_result.get('success'):
                print(f"  ✓ Semantic Scholar: {ss_result.get('citation_count', 0)} citations")
                
                # Use Semantic Scholar data as primary
                result = {
                    'doi': doi,
                    'citation_count': ss_result.get('citation_count', 0),
                    'title': ss_result.get('title', ''),
                    'authors': ss_result.get('authors', []),
                    'journal': ss_result.get('journal', ''),
                    'year': ss_result.get('year'),
                    'url': ss_result.get('url', ''),
                    'semantic_scholar_citations': ss_result.get('citation_count', 0),
                    'influential_citations': ss_result.get('influential_citation_count', 0),
                    'semantic_scholar_id': ss_result.get('semantic_scholar_id', ''),
                    'primary_source': 'semantic_scholar',
                    'success': True,
                    'note': 'Crossref failed, used Semantic Scholar'
                }
                return result
            
            else:
                print(f"  ✗ Semantic Scholar also failed: {ss_result.get('error', 'Unknown error')}")
                return {
                    'doi': doi,
                    'citation_count': 0,
                    'title': '',
                    'authors': [],
                    'journal': '',
                    'year': None,
                    'url': '',
                    'semantic_scholar_citations': 0,
                    'influential_citations': 0,
                    'semantic_scholar_id': '',
                    'primary_source': 'none',
                    'success': False,
                    'crossref_error': crossref_result.get('error', ''),
                    'semantic_scholar_error': ss_result.get('error', ''),
                    'note': 'Both Crossref and Semantic Scholar failed'
                }
    
    def process_doi_list(self, doi_list: List[str]) -> pd.DataFrame:
        """
        Process a list of DOIs and return results as DataFrame
        """
        print(f"Processing {len(doi_list)} DOIs...")
        print("=" * 50)
        
        results = []
        
        for i, doi in enumerate(doi_list, 1):
            print(f"\n[{i}/{len(doi_list)}]", end=" ")
            
            result = self.get_citation_count_with_fallback(doi)
            
            # Format for DataFrame
            row = {
                'doi': result['doi'],
                'citation_count': result.get('citation_count', 0),
                'title': result.get('title', ''),
                'authors': ', '.join(result.get('authors', [])) if isinstance(result.get('authors'), list) else str(result.get('authors', '')),
                'journal': result.get('journal', ''),
                'year': result.get('year', ''),
                'semantic_scholar_citations': result.get('semantic_scholar_citations', 0),
                'influential_citations': result.get('influential_citations', 0),
                'primary_source': result.get('primary_source', 'none'),
                'success': result.get('success', False),
                'note': result.get('note', ''),
                'url': result.get('url', ''),
                'semantic_scholar_id': result.get('semantic_scholar_id', '')
            }
            
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total DOIs processed: {len(df)}")
        print(f"Successful retrievals: {len(df[df['success'] == True])}")
        print(f"Failed retrievals: {len(df[df['success'] == False])}")
        print(f"Primary source - Crossref: {len(df[df['primary_source'] == 'crossref'])}")
        print(f"Primary source - Semantic Scholar: {len(df[df['primary_source'] == 'semantic_scholar'])}")
        print(f"No data found: {len(df[df['primary_source'] == 'none'])}")
        
        if len(df[df['success'] == True]) > 0:
            print(f"\nCitation statistics (successful retrievals only):")
            successful_df = df[df['success'] == True]
            print(f"Average citations: {successful_df['citation_count'].mean():.2f}")
            print(f"Median citations: {successful_df['citation_count'].median():.2f}")
            print(f"Max citations: {successful_df['citation_count'].max()}")
            print(f"Papers with 0 citations: {len(successful_df[successful_df['citation_count'] == 0])}")
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize citation counter with your email
    counter = CitationCounter(email="your-email@example.com")  # Replace with your email
try:
    df = pd.read_csv('data/BERTopic_abstract.csv') 
    df_test = df.head()
    doi_list_test = df_test['DOI'].dropna().tolist()
    doi_list = df['DOI'].dropna().tolist()  
    
    print(f"Loaded {len(doi_list_test)} DOIs from CSV")
    results_df = counter.process_doi_list(doi_list_test)
    results_df.to_csv("BE_Abstract_Citations.csv")
    
except FileNotFoundError:
    print("CSV file not found, using sample DOIs instead")
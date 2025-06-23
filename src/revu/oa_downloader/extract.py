import nbib
import csv
import os
import glob
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.phrases import Phraser
from gensim.models import Phrases
import umap
from umap import UMAP
import hdbscan
from gensim.models import Word2Vec
from collections import defaultdict
import matplotlib.pyplot as plt
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, PartOfSpeech, KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import rispy
import fitz  # PyMuPDF

def extract_and_save_pdf_text(pdf_directory, output_directory, extraction_method="default"):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Extraction method: {extraction_method}")
    print(f"Output directory: {output_directory}")
    print("-" * 50)
    
    extraction_stats = {
        'successful': 0,
        'failed': 0,
        'empty': 0,
        'total_chars': 0
    }
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_directory, filename)
        
        # Create output filename (replace .pdf with .txt)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_directory, txt_filename)
        
        try:
            doc = fitz.open(pdf_path)
            extracted_text = ""
            page_count = doc.page_count  # Store page count before processing
            
            # Different extraction methods
            for page_num in range(page_count):
                page = doc[page_num]
                
                if extraction_method == "default":
                    # Default text extraction
                    page_text = page.get_text()
                    
                elif extraction_method == "layout":
                    # Layout-preserving extraction
                    page_text = page.get_text("text", sort=True)
                    
                elif extraction_method == "dict":
                    # Dictionary-based extraction (more detailed)
                    text_dict = page.get_text("dict")
                    page_text = ""
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    page_text += span["text"] + " "
                                page_text += "\n"
                            page_text += "\n"
                    
                elif extraction_method == "blocks":
                    # Block-based extraction
                    blocks = page.get_text("blocks")
                    page_text = ""
                    for block in blocks:
                        if len(block) >= 5:  # Text blocks have at least 5 elements
                            page_text += block[4] + "\n\n"
                
                extracted_text += f"\n--- PAGE {page_num + 1} ---\n"
                extracted_text += page_text
                extracted_text += "\n"
            
            doc.close()
            
            # Save extracted text
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FILENAME: {filename}\n")
                f.write(f"EXTRACTION METHOD: {extraction_method}\n")
                f.write(f"TOTAL PAGES: {page_count}\n")
                f.write(f"TEXT LENGTH: {len(extracted_text)} characters\n")
                f.write("=" * 50 + "\n\n")
                f.write(extracted_text)
            
            # Update stats
            if len(extracted_text.strip()) == 0:
                extraction_stats['empty'] += 1
                status = "EMPTY"
            else:
                extraction_stats['successful'] += 1
                extraction_stats['total_chars'] += len(extracted_text)
                status = "SUCCESS"
            
            print(f"{i+1:3d}/{len(pdf_files)} - {filename[:40]:<40} [{len(extracted_text):6d} chars] - {status}")
            
        except Exception as e:
            extraction_stats['failed'] += 1
            print(f"{i+1:3d}/{len(pdf_files)} - {filename[:40]:<40} [ERROR: {str(e)[:30]}]")
            
            # Save error info
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"FILENAME: {filename}\n")
                f.write(f"ERROR OCCURRED DURING EXTRACTION\n")
                f.write(f"ERROR MESSAGE: {str(e)}\n")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(pdf_files)}")
    print(f"Successful extractions: {extraction_stats['successful']}")
    print(f"Failed extractions: {extraction_stats['failed']}")
    print(f"Empty extractions: {extraction_stats['empty']}")
    if extraction_stats['successful'] > 0:
        avg_chars = extraction_stats['total_chars'] / extraction_stats['successful']
        print(f"Average characters per successful extraction: {avg_chars:.0f}")
    print(f"\nText files saved to: {output_directory}")

def compare_extraction_methods(pdf_path, output_directory):
    """Compare different extraction methods on a single PDF."""
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    os.makedirs(output_directory, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]
    
    methods = ["default", "layout", "dict", "blocks"]
    
    print(f"Comparing extraction methods for: {filename}")
    print("-" * 50)
    
    for method in methods:
        try:
            doc = fitz.open(pdf_path)
            extracted_text = ""
            
            page_count = doc.page_count  # Store page count before processing
            
            for page_num in range(min(3, page_count)):  # Only first 3 pages for comparison
                page = doc[page_num]
                
                if method == "default":
                    page_text = page.get_text()
                elif method == "layout":
                    page_text = page.get_text("text", sort=True)
                elif method == "dict":
                    text_dict = page.get_text("dict")
                    page_text = ""
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    page_text += span["text"] + " "
                                page_text += "\n"
                            page_text += "\n"
                elif method == "blocks":
                    blocks = page.get_text("blocks")
                    page_text = ""
                    for block in blocks:
                        if len(block) >= 5:
                            page_text += block[4] + "\n\n"
                
                extracted_text += f"\n--- PAGE {page_num + 1} ---\n"
                extracted_text += page_text
                extracted_text += "\n"
            
            doc.close()
            
            # Save comparison file
            output_file = os.path.join(output_directory, f"{base_name}_{method}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"FILENAME: {filename}\n")
                f.write(f"EXTRACTION METHOD: {method}\n")
                f.write(f"FIRST 3 PAGES ONLY\n")
                f.write(f"TEXT LENGTH: {len(extracted_text)} characters\n")
                f.write("=" * 50 + "\n\n")
                f.write(extracted_text)
            
            print(f"{method.upper():8s} method: {len(extracted_text):6d} characters extracted")
            
        except Exception as e:
            print(f"{method.upper():8s} method: ERROR - {str(e)}")

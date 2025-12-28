import pymupdf  
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import os

def main():
    pdf_doc = pymupdf.open("oa_pdfs/10.1007_978-3-319-56091-5_11.pdf")
    for page in pdf_doc:
        text = page.get_text("xml")
    print(text)

    with open('oa_pdfs/10.1002_ajcp.12160.html') as fp:
        soup = BeautifulSoup(fp, 'xml')
    print(soup.prettify())

if __name__== "__main__":
    main()
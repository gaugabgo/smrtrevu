import tempfile
import os
from pathlib import Path
from revu.oa_downloader.extract_XML import TEIXMLParser, process_tei_files

def create_test_xml():
    """Create a minimal TEI XML file for testing"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a">Test Article Title</title>
            </titleStmt>
            <publicationStmt>
                <date when="2023-01-15">January 2023</date>
            </publicationStmt>
        </fileDesc>
        <profileDesc>
            <textClass>
                <keywords>
                    <term>test</term>
                </keywords>
            </textClass>
        </profileDesc>
    </teiHeader>
    <text>
        <front>
            <div>
                <listBibl>
                    <head>References</head>
                </listBibl>
            </div>
        </front>
        <body>
            <div>
                <head>Introduction</head>
                <p>This is the introduction paragraph.</p>
                <div>
                    <head>Background</head>
                    <p>This is background content.</p>
                    <p>Another background paragraph.</p>
                </div>
            </div>
            <div>
                <head>Methods</head>
                <p>Methods paragraph here.</p>
                <p>These are the methods.</p>
                <list>
                    <item>First method step</item>
                    <item>Second method step</item>
                </list>
            </div>
        </body>
    </text>
</TEI>"""
    return xml_content

def create_test_xml_with_authors():
    """Create TEI XML with author information"""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a">Multi-Author Paper</title>
                <author>
                    <persName>
                        <forename>John</forename>
                        <surname>Smith</surname>
                    </persName>
                </author>
                <author>
                    <persName>
                        <forename>Jane</forename>
                        <surname>Doe</surname>
                    </persName>
                </author>
            </titleStmt>
            <publicationStmt>
                <date when="2023-12-01">December 2023</date>
            </publicationStmt>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <div>
                <head>Abstract</head>
                <p>Abstract content goes here.</p>
            </div>
        </body>
    </text>
</TEI>"""
    return xml_content

def test_basic_parsing():
    parser = TEIXMLParser()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(create_test_xml())
        temp_path = f.name
    
    try:
        result = parser.parse_xml_file(temp_path)
        
        is_not_equal(None, result)
        is_equal("Test Article Title", result['title'])
        is_equal("2023-01-15", result['pub_date'])
        is_equal(["Unknown Author"], result['authors'])
        
        body_content = result['body_content']
        is_equal(True, len(body_content) > 0)
        
        headers = [item for item in body_content if item['content_type'] == 'header']
        is_equal(2, len(headers))
        
        sections = [item for item in body_content if item['content_type'] == 'section']
        is_equal(2, len(sections))
        
        list_items = [item for item in body_content if item['content_type'] == 'list_item']
        is_equal(2, len(list_items))
        
    finally:
        os.unlink(temp_path)

def is_equal(expected, actual):
    assert expected == actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

def is_not_equal(expected, actual):
    assert expected != actual, f"Expected: {expected}, Actual: {actual}"
    print(f"✓ Assertion passed")

if __name__ == '__main__':
    test_basic_parsing()
    print("All tests passed!")
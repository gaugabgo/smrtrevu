import os

# Look at one of your output files
xml_files = os.listdir("pdf_xml_out")
if xml_files:
    sample_file = xml_files[0]
    with open(f"pdf_xml_out/{sample_file}", 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    print("First 1000 characters:")
    print(content[:1000])
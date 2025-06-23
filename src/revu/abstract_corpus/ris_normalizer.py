import os
import glob
import re

def preprocess_and_save(input_folder, output_folder):
    """Preprocess EBSCO RIS files by normalizing dates and save them to a new folder."""
    os.makedirs(output_folder, exist_ok=True)
    ris_files = glob.glob(os.path.join(input_folder, "*.ris"))

    for filepath in ris_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as infile:
            raw_text = infile.read()

        lines = raw_text.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("Y1  -"):
                match = re.search(r'(\d{4})', line)
                if match:
                    year = match.group(1)
                    new_lines.append(f"PY  - {year}")
                    continue
            new_lines.append(line)

        normalized_text = "\n".join(new_lines)

        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(normalized_text)

import csv

import pytest

from revu.abstract_corpus.abstract_parser import parse_nbib_folder, parse_ris_folder

@pytest.fixture
def sample_nbib_file(tmp_path):
    nbib_content = """\
PMID- 12345678
TI  - Sample NBIB Title
AB  - Sample abstract content.
FAU - Doe, John
AU  - Doe J
DP  - 2020
JT  - Sample Journal
AID - 10.1234/nbib.doi [doi]

"""
    nbib_path = tmp_path / "sample.nbib"
    nbib_path.write_text(nbib_content, encoding="utf-8")
    return nbib_path


@pytest.fixture
def sample_ris_file(tmp_path):
    ris_content = """
TY  - JOUR
TI  - Sample RIS Title
AU  - Doe J
PY  - 2021
JO  - RIS Journal
AB  - A RIS abstract here.
DO  - 10.5678/ris.doi
ER  -
"""
    ris_path = tmp_path / "sample.ris"
    ris_path.write_text(ris_content.strip(), encoding="utf-8")
    return ris_path


def test_parse_nbib_folder(tmp_path, sample_nbib_file):
    output_csv = tmp_path / "output_nbib.csv"
    parse_nbib_folder(str(tmp_path), str(output_csv))

    with output_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["TI"] == "Sample NBIB Title"
    assert "Doe" in row["AU"]
    assert row["DP"] == "2020"
    assert row["JT"] == "Sample Journal"
    assert row["DOI"] == "10.1234/nbib.doi"


def test_parse_ris_folder(tmp_path, sample_ris_file):
    output_csv = tmp_path / "output_ris.csv"
    parse_ris_folder(str(tmp_path), str(output_csv))

    with output_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["TI"] == "Sample RIS Title"
    assert "Doe" in row["AU"]
    assert row["PY"] == "2021"
    assert row["T2"] == "RIS Journal"
    assert row["DOI"] == "10.5678/ris.doi"

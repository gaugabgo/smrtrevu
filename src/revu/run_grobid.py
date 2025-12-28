from grobid_client.grobid_client import GrobidClient

# GROBID (lightweight image) used for extracting text from pdf articles
# Need to open Docker to run grobid servers and specify extraction params in config.json
# type into command line: docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.2
# Last, run the python script to process documents

if __name__ == "__main__":
    client = GrobidClient(config_path="src/revu/config.json")
    client.process("processFulltextDocument", "data/ft_output/2nd_round_pdfs", output="pdf_xml_out", consolidate_citations=False, tei_coordinates=True, force=True)

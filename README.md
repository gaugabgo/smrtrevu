# 📚 Revu CLI Toolkit

**Revu** is your all-in-one command-line toolkit for:

- 🧾 Parsing citation files (NBIB & RIS)  
- 🧹 Cleaning & merging metadata  
- 🧠 Running topic models with BERTopic  
- 📥 Downloading and extracting Open Access (OA) PDFs  

Whether you're a researcher, developer, or text mining enthusiast — Revu helps you wrangle your literature data with style.

---

## 🚀 Installation

```bash
pip install -e .
```

> Make sure you're in the root directory of the `revu` project with a valid `setup.py` or `pyproject.toml`.

---

## 🧾 Citation Parsing

```bash
# Parse NBIB files
revu parse nbib data/abstract_import/nbib_raw data/abstract_output/nbib_parsed nbibparsed.csv

# Parse RIS files
revu parse ris data/abstract_import/ris_raw data/abstract_output/ris_parsed risparsed.csv
```

---

## 🧼 Citation Preprocessing

```bash
# Normalize RIS files
revu preprocess data/abstract_import/raw_EBSCO_ris data/abstract_output/normalized_ris

# Merge multiple parsed CSVs
revu merge

# Deduplicate merged citations
revu deduplicate
```

---

## 🧠 Topic Modeling (BERTopic FTW)

```bash
# Train a BERTopic model
revu model run --input_csv parsed.csv --output_csv topics.csv

# Visualize your topic landscape
revu model visualize --input_csv topics.csv
```

---

## 📂 Open Access (OA) Workflow

```bash
# Download PDFs via Unpaywall
revu oa download -i dois.txt -o metadata.csv --email you@example.com

# Extract text from downloaded PDFs
revu oa extract --pdf-dir oa_pdfs --output-dir oa_texts
```

---

## 🤖 CLI Structure

Revu is modular and click-powered:

```bash
revu [parse|preprocess|merge|deduplicate|model|oa]
```

- `parse` – Parse `.nbib` or `.ris` citation files  
- `preprocess` – Normalize messy citation formats  
- `merge` – Combine multiple CSVs  
- `deduplicate` – Remove redundant records  
- `model` – Run and visualize topic modeling  
- `oa` – Download and extract Open Access content  

---

## 🧙‍♂️ Pro Tips

- Use `--help` with any subcommand for options.  
- Plug it into a pipeline for scalable workflows.  
- Works great with `make`, `snakemake`, or a dash of Python scripting.

---

## 💬 Feedback

We love ideas, bugs, and pull requests! 🐛  
Feel free to open an issue or start a discussion.

---

## 🧾 License

MIT (or your preferred license)

---

✨ Happy parsing, modeling, and discovering!

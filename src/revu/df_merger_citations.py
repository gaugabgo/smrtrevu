import pandas as pd

def main():
    df_original = pd.read_csv("data/preprocess_out/processed_articles_BE.csv")
    df_citations = pd.read_csv("data/BERTopic_abstract_with_citations_onlycitations.csv")

    # Clean DOIs first (remove whitespace, standardize format)
    df_original['DOI'] = df_original['DOI'].str.strip()
    df_citations['DOI'] = df_citations['DOI'].str.strip()

    # Check for duplicates in citation data
    print("Duplicates in citations:", df_citations['DOI'].duplicated().sum())
    if df_citations['DOI'].duplicated().any():
        # Keep first occurrence or handle as needed
        df_citations = df_citations.drop_duplicates(subset='DOI', keep='first')

    # Merge
    df_merged = df_original.merge(df_citations, on='DOI', how='left')
    df_merged.to_csv("BE_processed_withcitations.csv")

    # Check the merge results
    print(f"Original rows: {len(df_original)}")
    print(f"Merged rows: {len(df_merged)}")
    print(f"Rows with citation data: {df_merged['citation_count'].notna().sum()}")

    # See which DOIs didn't get matched
    unmatched = df_merged[df_merged['citation_count'].isna()]['DOI']
    print(f"Unmatched DOIs: {len(unmatched)}")

    # Check for DOIs in citations that weren't in original
    extra_dois = set(df_citations['DOI']) - set(df_original['DOI'])
    print(f"Extra DOIs in citation data: {len(extra_dois)}")

if __name__ == "__main__":
    main()
    print("DF merger completed")
import pandas as pd
import os

def deduplicate_csv(input_file, output_file, log_file):
    """
    Remove duplicates from CSV based on DOI and Title columns.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        log_file (str): Path to log file
    """
    # Read and prepare data
    df = pd.read_csv(input_file, dtype=str).fillna('')

    # Strip any quotes from column names
    df.columns = df.columns.str.strip().str.strip("'")
    
    # Track counts and create log
    counts = {'original': len(df)}
    
    # Sort by Abstract so rows with text come first (non-empty strings sort before empty)
    # Using key=lambda to ensure rows with abstracts are prioritized
    df['has_abstract'] = df['abstract'].astype(str).str.strip().ne('')
    df = df.sort_values('has_abstract', ascending=False)

    # Filter to only English language entries
    before_lang_filter = len(df)
    df = df[df['language'].str.lower() == 'en']
    counts['language_filter'] = before_lang_filter - len(df)

    # Remove duplicates by ID, then DOI, then by Title
    # Now it will keep the first instance (which will be one with abstract if available)
    for col in ['id', 'doi', 'title']:
        before = len(df)
        df = df.drop_duplicates(subset=col, keep='first')
        counts[col.lower()] = before - len(df)

    # Remove the helper column
    df = df.drop(columns=['has_abstract'])
    
    counts['final'] = len(df)
    
    # Generate log messages
    log_messages = [
        f"Original entries: {counts['original']}",
        f"Non-English entries removed: {counts['language_filter']}",
        f"Duplicates removed based on OpenAlex ID: {counts['id']}",
        f"Duplicates removed based on DOI: {counts['doi']}",
        f"Duplicates removed based on Title: {counts['title']}",
        f"Final entries after deduplication: {counts['final']}"
    ]
    
    # Save results and write log
    df.to_csv(output_file, index=False)

    # If log_file is a directory, create a log file inside it
    if os.path.isdir(log_file):
        log_file = os.path.join(log_file, 'deduplication_log.txt')

    with open(log_file, 'w') as f:
        for msg in log_messages:
            print(msg)
            f.write(msg + '\n')

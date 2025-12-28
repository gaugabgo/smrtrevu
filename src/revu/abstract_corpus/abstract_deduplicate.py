import pandas as pd

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
    
    # Track counts and create log
    counts = {'original': len(df)}
    
    # Remove duplicates by DOI, then by Title
    for col in ['DOI', 'Title']:
        before = len(df)
        df = df.drop_duplicates(subset=col, keep='first')
        counts[col.lower()] = before - len(df)
    
    counts['final'] = len(df)
    
    # Generate log messages
    log_messages = [
        f"Original entries: {counts['original']}",
        f"Duplicates removed based on DOI: {counts['doi']}",
        f"Duplicates removed based on Title: {counts['title']}",
        f"Final entries after deduplication: {counts['final']}"
    ]
    
    # Save results and write log
    df.to_csv(output_file, index=False)
    
    with open(log_file, 'w') as f:
        for msg in log_messages:
            print(msg)
            f.write(msg + '\n')

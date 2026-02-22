import pandas as pd
import os


def extract_and_deduplicate_metadata(
    original_csv_files,
    metadata_csv,
    metadata_deduplicated_csv,
    metadata_columns,
    topic_model_csv=None,
    log_file=None
):
    """
    Extract specified metadata columns and create deduplicated metadata file.

    This function:
    1. Extracts specified metadata columns from original CSV files
    2. Saves merged metadata to CSV
    3. Applies deduplication logic prioritizing records with full abstract information
    4. Saves deduplicated metadata to CSV
    5. Validates ID matching with topic model data (if provided)

    Args:
        original_csv_files (list): List of paths to original CSV files with full metadata
        metadata_csv (str): Path to output merged metadata CSV file
        metadata_deduplicated_csv (str): Path to output deduplicated metadata CSV file
        metadata_columns (list): List of metadata columns to extract (should include ID, DOI, Title)
        topic_model_csv (str, optional): Path to CSV file containing topic model assignments for ID validation
        log_file (str, optional): Path to log file. If None, no log is written.

    Returns:
        pd.DataFrame: Deduplicated metadata DataFrame
    """

    print("\n" + "="*70)
    print("METADATA EXTRACTION AND DEDUPLICATION")
    print("="*70)

    # ========================================
    # STEP 1: Load topic model IDs (if provided)
    # ========================================
    topic_ids_filter = None
    if topic_model_csv:
        print("\n" + "="*70)
        print("STEP 1: Loading topic model IDs for filtering")
        print("="*70)

        print(f"Reading topic model data from: {topic_model_csv}")
        topic_df = pd.read_csv(topic_model_csv, dtype=str)
        print(f"✓ Loaded {len(topic_df)} records from topic model CSV")

        # Find ID column in topic model
        topic_id_col = None
        for col in topic_df.columns:
            if col.lower() == 'id':
                topic_id_col = col
                break

        if topic_id_col is None:
            print("  Warning: No 'id' column found in topic model CSV")
            print("  Proceeding without ID filtering")
        else:
            topic_ids_filter = set(topic_df[topic_id_col].astype(str).str.strip())
            print(f"✓ Extracted {len(topic_ids_filter)} unique IDs from topic model")
            print(f"  Metadata will be filtered to match only these IDs")

    # ========================================
    # STEP 2: Validate metadata columns
    # ========================================
    print("\n" + "="*70)
    step_num = 2 if topic_model_csv else 1
    print(f"STEP {step_num}: Validating metadata columns specification")
    print("="*70)

    if not metadata_columns:
        raise ValueError("metadata_columns must be specified and cannot be empty")

    # Check that required columns are present (case-insensitive)
    required_cols = ['id', 'doi', 'title']
    metadata_cols_lower = [col.lower() for col in metadata_columns]

    missing_required = []
    for req in required_cols:
        if req not in metadata_cols_lower:
            missing_required.append(req.upper())

    if missing_required:
        print(f"  Warning: Recommended columns missing: {', '.join(missing_required)}")
        print(f"  These columns are needed for deduplication logic")
    else:
        print(f"  ✓ All recommended columns present: ID, DOI, Title")

    print(f"\nMetadata columns to extract ({len(metadata_columns)}):")
    for col in metadata_columns:
        print(f"  - {col}")

    # ========================================
    # STEP 3: Extract metadata from original files
    # ========================================
    print("\n" + "="*70)
    step_num = 3 if topic_model_csv else 2
    print(f"STEP {step_num}: Extracting metadata from original files")
    print("="*70)

    if not original_csv_files:
        raise ValueError("No original CSV file paths provided")

    print(f"Processing {len(original_csv_files)} original CSV files...")

    # Read and extract metadata from each file
    metadata_dfs = []
    for i, path in enumerate(original_csv_files, 1):
        print(f"\n  [{i}/{len(original_csv_files)}] Reading: {path}")

        df = pd.read_csv(path, dtype=str).fillna('')

        # Strip quotes from column names
        df.columns = df.columns.str.strip().str.strip("'")

        print(f"      Total records: {len(df)}")
        print(f"      Available columns: {len(df.columns)}")

        # Map requested columns to actual columns (case-insensitive)
        available_cols_lower = {col.lower(): col for col in df.columns}
        cols_to_extract = []

        for col in metadata_columns:
            col_lower = col.lower()
            if col_lower in available_cols_lower:
                cols_to_extract.append(available_cols_lower[col_lower])
            else:
                print(f"      Warning: Column '{col}' not found, skipping")

        # Also include Abstract/AB for deduplication prioritization (even if not requested)
        for abstract_col in ['Abstract', 'AB']:
            if abstract_col in df.columns and abstract_col not in cols_to_extract:
                cols_to_extract.append(abstract_col)
                print(f"      Adding '{abstract_col}' for deduplication prioritization")

        # Extract only specified columns
        df_extracted = df[cols_to_extract].copy()
        print(f"      Extracted columns: {len(df_extracted.columns)}")

        metadata_dfs.append(df_extracted)

    # ========================================
    # STEP 4: Merge extracted metadata
    # ========================================
    print("\n" + "="*70)
    step_num = 4 if topic_model_csv else 3
    print(f"STEP {step_num}: Merging extracted metadata")
    print("="*70)

    merged_metadata = pd.concat(metadata_dfs, ignore_index=True)
    print(f"✓ Merged metadata from {len(original_csv_files)} files")
    print(f"  Total records: {len(merged_metadata)}")
    print(f"  Total columns: {len(merged_metadata.columns)}")
    print(f"  Columns: {', '.join(merged_metadata.columns)}")

    # Track counts for logging
    counts = {'original': len(merged_metadata)}

    # ========================================
    # STEP 5: Filter to topic model IDs (if provided)
    # ========================================
    if topic_ids_filter is not None:
        print("\n" + "="*70)
        print("STEP 5: Filtering metadata to match topic model IDs")
        print("="*70)

        # Find ID column in metadata
        metadata_id_col = None
        for col in merged_metadata.columns:
            if col.lower() == 'id':
                metadata_id_col = col
                break

        if metadata_id_col is None:
            print("  Warning: No 'id' column found in metadata, skipping ID filtering")
        else:
            before_filter = len(merged_metadata)
            # Filter to only IDs that exist in topic model
            merged_metadata[metadata_id_col] = merged_metadata[metadata_id_col].astype(str).str.strip()
            merged_metadata = merged_metadata[merged_metadata[metadata_id_col].isin(topic_ids_filter)]
            after_filter = len(merged_metadata)
            counts['topic_model_filter'] = before_filter - after_filter

            print(f"  ✓ Filtered metadata to match topic model IDs")
            print(f"    Records before filter: {before_filter}")
            print(f"    Records after filter: {after_filter}")
            print(f"    Records removed: {counts['topic_model_filter']}")

    # Save merged metadata
    merged_metadata.to_csv(metadata_csv, index=False)
    print(f"\n✓ Merged metadata saved to: {metadata_csv}")

    # ========================================
    # STEP 6: Apply deduplication logic
    # ========================================
    print("\n" + "="*70)
    step_num = 6 if topic_model_csv else 4
    print(f"STEP {step_num}: Applying deduplication logic")
    print("="*70)
    print("Prioritizing records with abstracts and removing duplicates...")

    # Identify abstract column (case-insensitive)
    abstract_col = None
    for col in merged_metadata.columns:
        if col.lower() == 'abstract' or col.lower() == 'ab':
            abstract_col = col
            break

    if abstract_col:
        # Sort by Abstract so rows with text come first
        merged_metadata['has_abstract'] = merged_metadata[abstract_col].astype(str).str.strip().ne('')
        merged_metadata = merged_metadata.sort_values('has_abstract', ascending=False)
        print(f"  ✓ Using '{abstract_col}' column for prioritization")

        records_with_abstract = merged_metadata['has_abstract'].sum()
        records_without_abstract = len(merged_metadata) - records_with_abstract
        print(f"    - Records with abstract: {records_with_abstract}")
        print(f"    - Records without abstract: {records_without_abstract}")
    else:
        print("  Warning: No abstract column found, skipping abstract prioritization")

    # Identify deduplication columns (ID, DOI, Title)
    dedup_cols = []
    for col_name in ['ID', 'DOI', 'Title']:
        for col in merged_metadata.columns:
            if col.lower() == col_name.lower():
                dedup_cols.append((col_name, col))
                break

    if not dedup_cols:
        print("  Warning: No deduplication columns (ID, DOI, Title) found")
    else:
        print(f"\n  Deduplication sequence:")
        # Remove duplicates by ID, DOI, Title
        for col_name, col in dedup_cols:
            before = len(merged_metadata)
            merged_metadata = merged_metadata.drop_duplicates(subset=col, keep='first')
            removed = before - len(merged_metadata)
            counts[col_name.lower()] = removed
            print(f"    {col_name}: Removed {removed} duplicates")

    # Remove helper column if it exists
    if 'has_abstract' in merged_metadata.columns:
        merged_metadata = merged_metadata.drop(columns=['has_abstract'])

    # Remove Abstract/AB column if it wasn't originally requested
    metadata_cols_lower = [col.lower() for col in metadata_columns]
    for abstract_col in ['Abstract', 'AB']:
        if abstract_col in merged_metadata.columns:
            if abstract_col.lower() not in metadata_cols_lower:
                merged_metadata = merged_metadata.drop(columns=[abstract_col])
                print(f"  ✓ Removed '{abstract_col}' column (used only for prioritization)")

    counts['deduplicated'] = len(merged_metadata)
    print(f"\n✓ Deduplication complete")
    print(f"  Original records: {counts['original']}")
    print(f"  Final records: {counts['deduplicated']}")
    print(f"  Total removed: {counts['original'] - counts['deduplicated']}")

    # Save deduplicated metadata
    merged_metadata.to_csv(metadata_deduplicated_csv, index=False)
    print(f"\n✓ Deduplicated metadata saved to: {metadata_deduplicated_csv}")

    # ========================================
    # STEP 7: Validate ID matching with topic model (optional)
    # ========================================
    if topic_ids_filter is not None:
        print("\n" + "="*70)
        print("STEP 7: Validating ID matching with topic model data")
        print("="*70)

        # Find ID column in metadata
        metadata_id_col = None
        for col in merged_metadata.columns:
            if col.lower() == 'id':
                metadata_id_col = col
                break

        if metadata_id_col is None:
            print("  Warning: No 'id' column found in metadata, skipping validation")
        else:
            # Compare IDs
            metadata_ids = set(merged_metadata[metadata_id_col].astype(str).str.strip())

            print(f"\n  ID Matching Analysis:")
            print(f"    Topic model IDs: {len(topic_ids_filter)}")
            print(f"    Metadata IDs: {len(metadata_ids)}")

            # Find matching and non-matching IDs
            matching_ids = topic_ids_filter & metadata_ids
            topic_only_ids = topic_ids_filter - metadata_ids
            metadata_only_ids = metadata_ids - topic_ids_filter

            print(f"\n  Matching Results:")
            print(f"    ✓ Matching IDs: {len(matching_ids)} ({len(matching_ids)/len(topic_ids_filter)*100:.1f}% of topic model)")

            if topic_only_ids:
                print(f"    ⚠ IDs in topic model but NOT in metadata: {len(topic_only_ids)}")
                if len(topic_only_ids) <= 10:
                    print(f"      {list(topic_only_ids)[:10]}")
                else:
                    print(f"      First 10: {list(topic_only_ids)[:10]}")
            else:
                print(f"    ✓ All topic model IDs found in metadata")

            if metadata_only_ids:
                print(f"    ⚠ IDs in metadata but NOT in topic model: {len(metadata_only_ids)}")
                print(f"      This should not happen - metadata was filtered to topic model IDs")
            else:
                print(f"    ✓ No extra metadata IDs")

            # Calculate match percentage
            match_pct = len(matching_ids) / len(topic_ids_filter) * 100 if topic_ids_filter else 0
            if match_pct == 100:
                print(f"\n  ✅ PERFECT MATCH: All topic model IDs can be linked with metadata")
            elif match_pct >= 95:
                print(f"\n  ✓ GOOD MATCH: {match_pct:.1f}% of topic model IDs can be linked")
            else:
                print(f"\n  ⚠ WARNING: Only {match_pct:.1f}% of topic model IDs can be linked")
                print(f"     Some topic model IDs are missing from original metadata files")

    # ========================================
    # STEP 8: Write log file
    # ========================================
    if log_file:
        print("\n" + "="*70)
        step_num = 8 if topic_model_csv else 5
        print(f"STEP {step_num}: Writing log file")
        print("="*70)

        # If log_file is a directory, create a log file inside it
        if os.path.isdir(log_file):
            log_file = os.path.join(log_file, 'metadata_extraction_log.txt')

        log_messages = [
            "="*70,
            "METADATA EXTRACTION AND DEDUPLICATION LOG",
            "="*70,
            "",
            "INPUT FILES:",
            f"  Original CSV files: {len(original_csv_files)}",
        ]

        for i, path in enumerate(original_csv_files, 1):
            log_messages.append(f"    {i}. {path}")

        log_messages.extend([
            "",
            "METADATA COLUMNS REQUESTED:",
        ])

        for col in metadata_columns:
            log_messages.append(f"  - {col}")

        log_messages.extend([
            "",
            "PROCESSING SUMMARY:",
            f"  Original entries (merged): {counts['original']}",
        ])

        # Add topic model filtering info if applicable
        if 'topic_model_filter' in counts:
            log_messages.extend([
                "",
                "TOPIC MODEL FILTERING:",
                f"  Topic model CSV: {topic_model_csv}",
                f"  Topic model IDs: {len(topic_ids_filter)}",
                f"  Records filtered to match topic model: {counts['topic_model_filter']}",
                f"  Records after filtering: {counts['original'] - counts['topic_model_filter']}",
            ])

        log_messages.extend([
            "",
            "DEDUPLICATION:",
        ])

        for col_name, col in dedup_cols:
            if col_name.lower() in counts:
                log_messages.append(f"  Duplicates removed based on {col_name}: {counts[col_name.lower()]}")

        log_messages.extend([
            f"  Final entries after deduplication: {counts['deduplicated']}",
        ])

        # Add ID matching results if available
        if topic_ids_filter is not None and metadata_id_col:
            log_messages.extend([
                "",
                "ID MATCHING VALIDATION:",
                f"  Topic model IDs: {len(topic_ids_filter)}",
                f"  Metadata IDs: {len(metadata_ids)}",
                f"  Matching IDs: {len(matching_ids)} ({len(matching_ids)/len(topic_ids_filter)*100:.1f}%)",
                f"  IDs in topic model only: {len(topic_only_ids)}",
                f"  IDs in metadata only: {len(metadata_only_ids)}",
            ])

        log_messages.extend([
            "",
            "OUTPUT FILES:",
            f"  Merged metadata: {metadata_csv}",
            f"  Deduplicated metadata: {metadata_deduplicated_csv}",
            "="*70,
        ])

        with open(log_file, 'w') as f:
            for msg in log_messages:
                f.write(msg + '\n')

        print(f"✓ Log file saved to: {log_file}")

    print("\n" + "="*70)
    print("✅ METADATA EXTRACTION AND DEDUPLICATION COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. Merged metadata: {metadata_csv}")
    print(f"  2. Deduplicated metadata: {metadata_deduplicated_csv}")
    if log_file:
        print(f"  3. Log file: {log_file}")
    print("")

    return merged_metadata
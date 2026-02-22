## Step 1

Flatten data from each csv file. Each row becomes its own csv file with the column labels remaining intact. The file name becomes the text from the 'id' column. 

Sample csv data input:

### Step 2

Remove duplicate records by identifying multiples with the same id (filename). Only records with unique IDs kept. Log all duplicates and removals.

### Step 3

Create primary database. Data frame containing data from the 'id', 'doi', 'title', and 'abstract' columns from each of the unique record files. Allow metadata from flattened data (Step 1) to be linked to primary database based on id.

Sample csv output:


### Step 4

Check for duplicates based on 'doi' and 'title' texts. Keep only records with unique DOI and title. Log duplicates identified and removed based on title. Check primary database for missing abstracts. Log missing abstracts and remove rows for records with missing abstracts.

### Step 5

Use primary database to run 'revu preprocess-texts' pipeline. Implement batching to accomodate for large datasets. Save dataset with "processed_text" column added.

### Step 6

Use primary dataset to run 'revu model run' pipeline, keeping only the bertopic and model saving code. Implement batching to accomodate for large datasets. 

### Step 7

Use saved topic model data to generate visualization code from 'revu model run' pipeline. Use metadata from flattened data (Step 1) linked using 'id' text to generate 'hover text' information. 


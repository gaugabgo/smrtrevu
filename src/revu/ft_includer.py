import pandas as pd

"""
Article inclusion and exclusion based on cluster allocation.
Exclusion values represent topic cluster numbers which are not relevant to review.
Note this is the main topic - excluded values may be secondary/tertiary topics.
"""
# Load your CSV file
df = pd.read_csv('/Users/gabriellegauthier/Projects/smrtrevu_workspace/smrtrevu/BERTopic_abstract.csv')

# Define exclusion values for the main_topic column
exclusion_values = [-1, 3, 4, 7, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 36, 40, 41, 42, 43]  

# Step 1: Make a copy of the dataframe
df_copy = df.copy()

# Step 2: Filter out rows where main_topic is in exclusion_values
df_filtered = df_copy[~df_copy['main_topic'].isin(exclusion_values)]

# Debug: Check the data type and unique values
print("Data type of main_topic column:", df_copy['main_topic'].dtype)
print("Unique values in main_topic:", sorted(df_copy['main_topic'].unique()))
print("Sample values:", df_copy['main_topic'].head(10).tolist())

# Verify filtering worked by checking remaining unique values
remaining_topics = sorted(df_filtered['main_topic'].unique())
print(f"Remaining topics after filtering: {remaining_topics}")

# Save the filtered dataframe to a new CSV
df_filtered.to_csv('BERTopic_includedabstracts.csv', index=False)

# Optional: Display first few rows of filtered data
#print("\nFirst 5 rows of filtered data:")
#print(df_filtered.head())
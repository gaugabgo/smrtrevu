import pandas as pd

# Load your CSV file
df = pd.read_csv('metadata_BE.csv')

# Define exclusion values for the main_topic column
exclusion_value = [False]

#Copy df
df_copy = df.copy()

# Filter out rows where main_topic is in exclusion_values
df_filtered = df_copy[~df_copy['downloaded'].isin(exclusion_value)]

# Save the filtered dataframe to a new CSV
df_filtered.to_csv('BERTopic_ftdownloaded.csv', index=False)
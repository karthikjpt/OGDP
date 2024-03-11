import pandas as pd
import numpy as np
import os

# Define the path to the directory containing the CSV files
directory_path = '/home/ogdp/ogdp/data/earthquake/'

# List of CSV files in the directory
file_names = ['depth_energy.csv', 'latitude_energy.csv', 'longitude_energy.csv', 'magnitude_energy.csv']

# Iterate through each file
for file_name in file_names:
    # Construct the full path to the CSV file
    file_path = os.path.join(directory_path, file_name)

    # Read the CSV file using pandas
    df = pd.read_csv(file_path, index_col=0)

    # Exclude the first row and apply the modified formula to specific cells
    for col in df.columns[0:]:
        # Remove specific words from column names
        col_name_without_prefix = col.replace('depth', '').replace('latitude', '').replace('longitude', '').replace('magnitude', '')
        # Apply the formula to the column
        df[col_name_without_prefix] = df[col].map(lambda x: round((np.log10(x) - 4.4) / 1.5, 1) if pd.notnull(x) and x > 0 else x)
        # Drop the original column
        df.drop(col, axis=1, inplace=True)

    # Save the updated DataFrame to the same CSV file
    df.to_csv(file_path)

# Print a success message after processing all files
print("Successfully done.")

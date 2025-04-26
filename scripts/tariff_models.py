import pandas as pd
import numpy as np
import os

# Define file paths
input_folder = "data"
output_folder = "data"
input_file = os.path.join(input_folder, "CMO-Historical-Data-Monthly.xlsx")
output_file = os.path.join(output_folder, "CMO-Historical-Data-Monthly-CLEANED.xlsx")

# Load the raw data
print("Loading data from:", input_file)
raw_data = pd.read_excel(input_file, 
                        sheet_name="Monthly Prices", 
                        skiprows=4)

# Step 1: Clean column names
print("Cleaning column names...")
raw_data = raw_data.rename(columns={raw_data.columns[0]: "Date"})
raw_data.columns = [col.replace("**", "").strip() for col in raw_data.columns]

# Step 2: Convert date column to proper date format (no time)
print("Processing dates...")
raw_data['Date'] = raw_data['Date'].astype(str)
raw_data['Date'] = raw_data['Date'].str.replace('M', '-')
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m', errors='coerce').dt.date

# Drop rows with invalid dates
raw_data = raw_data.dropna(subset=['Date'])

# Step 3: Set Date as index
raw_data.set_index("Date", inplace=True)

# Step 4: Clean data values
print("Cleaning numeric data...")
raw_data = raw_data.replace('...', np.nan)
raw_data = raw_data.replace(r'^\s*$', np.nan, regex=True)

# Convert all numeric columns to float
for col in raw_data.columns:
    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

# Step 5: Remove empty columns
print("Removing empty columns...")
threshold = len(raw_data) * 0.5
raw_data = raw_data.dropna(axis=1, thresh=threshold)

# Step 6: Fill missing values
print("Handling missing values...")
raw_data = raw_data.ffill().bfill()

# Step 7: Save cleaned data
print("Saving cleaned data to:", output_file)
with pd.ExcelWriter(output_file, 
                   engine='openpyxl',
                   date_format='mm/dd/yyyy') as writer:
    raw_data.to_excel(writer, sheet_name="Cleaned Data")

print("\nCleaning complete!")
print("Original shape:", raw_data.shape)
print("Columns kept:", list(raw_data.columns))

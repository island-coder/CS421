import os, csv
import pandas as pd

def initialize_csv_with_headers(file_path,header=['label', 'n_images', 'timestamp']):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(header)

def append_rows_to_csv(file_path, rows):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write each row
        writer.writerows(rows)

def read_rows_from_csv(file_path):
    rows = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        rows = [row for row in reader]
    return rows

def check_file_exists(file_path):
    return os.path.isfile(file_path)

# 1. Initialize or Load the DataFrame
def initialize_dataframe(file_path: str) -> pd.DataFrame:
    columns = ['id', 'title', 'content', 'images', 
               'metadata', 'entities', 'triples', 
               'face-reid-status', 'grounding-status']
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)
    
    return df

# 2. Add a New Entry
def add_entry(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    entry_df = pd.DataFrame([entry])  # Convert the dictionary to a DataFrame
    df = pd.concat([df, entry_df], ignore_index=True)  # Use pd.concat to add the new row
    return df

# 3. Check if Row with Particular ID Exists and Retrieve It
def get_row_by_id(df: pd.DataFrame, entry_id: int) -> pd.Series:
    row = df[df['id'] == entry_id]
    if not row.empty:
        return row.iloc[0]  # Return the first matching row
    else:
        return None

# 4. Update Particular Row Values
def update_row_by_id(df: pd.DataFrame, entry_id: int, updates: dict) -> pd.DataFrame:
    index = df.index[df['id'] == entry_id].tolist()
    if index:
        df.loc[index[0], updates.keys()] = updates.values()
    else:
        print(f"No entry found with id: {entry_id}")
    return df

# 5. Save the DataFrame to a File
def save_dataframe(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)

# 6. Delete an Entry by ID
def delete_entry(df: pd.DataFrame, entry_id: int) -> pd.DataFrame:
    df = df[df['id'] != entry_id]
    return df

# 4. Save the DataFrame to a File
def save_dataframe(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)

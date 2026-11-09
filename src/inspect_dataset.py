"""
Comprehensive dataset inspection script for ShuttleSet in Badminton Pressure Analysis.
"""

import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "shuttleset")

def inspect_dataset():
    match_dirs = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    csv_files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)

    print("=================================================================")
    print("DATASET OVERVIEW")
    print("=================================================================")
    print(f"Dataset root: {RAW_DIR}")
    print(f"Total match directories: {len(match_dirs)}")
    print(f"Total set CSV files: {len(csv_files)}")

    # Check schemas
    columns_set = set()
    file_schemas = {}
    total_shots = 0
    all_lose_reasons = set()
    all_win_reasons = set()
    all_shot_types = set()
    all_players = set()
    all_getpoint_players = set()
    
    match_stats = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            total_shots += len(df)
            file_schemas[f] = list(df.columns)
            columns_set.update(df.columns)
            
            if 'lose_reason' in df.columns:
                all_lose_reasons.update(df['lose_reason'].dropna().unique())
            if 'win_reason' in df.columns:
                all_win_reasons.update(df['win_reason'].dropna().unique())
            if 'type' in df.columns:
                all_shot_types.update(df['type'].dropna().unique())
            if 'player' in df.columns:
                all_players.update(df['player'].dropna().unique())
            if 'getpoint_player' in df.columns:
                all_getpoint_players.update(df['getpoint_player'].dropna().unique())
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Total shots (records across all sets): {total_shots}")

    # Check if all files have the same columns
    first_cols = list(file_schemas.values())[0]
    all_same = all(cols == first_cols for cols in file_schemas.values())
    print(f"Are all CSV schemas identical? {all_same}")
    print(f"Total unique columns across all files: {len(columns_set)}")
    print(f"Columns: {first_cols}")

    # Inspect a sample set in depth
    sample_file = csv_files[0]
    sample_df = pd.read_csv(sample_file)
    print("\n=================================================================")
    print(f"SAMPLE FILE DETAILS: {os.path.basename(os.path.dirname(sample_file))}/{os.path.basename(sample_file)}")
    print("=================================================================")
    print("Data types and non-null counts:")
    info_df = pd.DataFrame({
        "column": sample_df.columns,
        "dtype": sample_df.dtypes.astype(str),
        "non_null_count": sample_df.notnull().sum(),
        "null_count": sample_df.isnull().sum(),
        "sample_val": [sample_df[c].dropna().iloc[0] if len(sample_df[c].dropna()) > 0 else np.nan for c in sample_df.columns]
    })
    print(info_df.to_string())

    print("\n=================================================================")
    print("CATEGORICAL DOMAIN VALUES")
    print("=================================================================")
    print(f"Unique 'lose_reason' values: {all_lose_reasons}")
    print(f"Unique 'win_reason' values: {all_win_reasons}")
    print(f"Unique 'player' values: {all_players}")
    print(f"Unique 'getpoint_player' values: {all_getpoint_players}")
    print(f"Number of unique 'type' (shot types): {len(all_shot_types)}")
    print(f"Sample 'type' values: {list(all_shot_types)[:10]}")

    # Inspect match directories naming convention
    print("\n=================================================================")
    print("MATCH DIRECTORY NAMING SAMPLES (FIRST 10)")
    print("=================================================================")
    for m in match_dirs[:10]:
        print(f"  {m}")

if __name__ == "__main__":
    inspect_dataset()


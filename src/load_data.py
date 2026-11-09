"""
Data loading module for ShuttleSet badminton dataset.
Loads match registry and individual set CSV files with structured metadata.
"""

import os
import sys
import glob
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

def load_match_registry(raw_data_dir: str = config.RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads match.csv and adds category (Men's Singles vs Women's Singles)
    and resolved directory paths.
    """
    match_file = os.path.join(raw_data_dir, "match.csv")
    if not os.path.exists(match_file):
        raise FileNotFoundError(f"Match registry not found at: {match_file}")

    df = pd.read_csv(match_file)
    
    # Classify match category (Men's vs Women's Singles) based on player roster
    def classify_category(row):
        w = str(row.get("winner", ""))
        l = str(row.get("loser", ""))
        if w in config.WOMEN_SINGLES_PLAYERS or l in config.WOMEN_SINGLES_PLAYERS:
            return "Women's Singles"
        return "Men's Singles"

    df["category"] = df.apply(classify_category, axis=1)

    # Locate match folder on disk (handling potential trailing whitespace or case differences)
    existing_dirs = {d.strip().lower(): d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))}
    
    def resolve_folder(row):
        vid = str(row.get("video", "")).strip().lower()
        if vid in existing_dirs:
            return os.path.join(raw_data_dir, existing_dirs[vid])
        # Direct check
        direct = os.path.join(raw_data_dir, str(row.get("video", "")))
        if os.path.exists(direct):
            return direct
        return None

    df["folder_path"] = df.apply(resolve_folder, axis=1)
    return df

def get_set_files(folder_path: str) -> List[Tuple[int, str]]:
    """
    Finds all set CSV files in a match folder, sorted by set number.
    Returns list of (set_number, file_path).
    """
    if not folder_path or not os.path.exists(folder_path):
        return []
    
    files = glob.glob(os.path.join(folder_path, "set*.csv"))
    results = []
    for f in files:
        basename = os.path.basename(f)
        # set1.csv, set2.csv, set3.csv
        num_str = "".join([c for c in basename if c.isdigit()])
        if num_str:
            results.append((int(num_str), f))
    results.sort(key=lambda x: x[0])
    return results

def load_raw_shots_for_match(match_row: pd.Series) -> List[Tuple[int, pd.DataFrame]]:
    """
    Loads all sets for a given match, returning list of (set_num, df).
    """
    folder = match_row.get("folder_path")
    set_files = get_set_files(folder)
    sets_data = []
    for set_num, fpath in set_files:
        df = pd.read_csv(fpath)
        sets_data.append((set_num, df))
    return sets_data

def load_all_raw_data(raw_data_dir: str = config.RAW_DATA_DIR) -> Dict:
    """
    Loads all matches and their sets into a structured dictionary.
    """
    registry = load_match_registry(raw_data_dir)
    matches_dict = {}
    total_shots = 0
    total_sets = 0

    for _, row in registry.iterrows():
        mid = row["id"]
        sets = load_raw_shots_for_match(row)
        total_sets += len(sets)
        for _, s_df in sets:
            total_shots += len(s_df)
        matches_dict[mid] = {
            "metadata": row.to_dict(),
            "sets": sets
        }

    return {
        "registry": registry,
        "matches": matches_dict,
        "total_matches": len(registry),
        "total_sets": total_sets,
        "total_shots": total_shots,
    }

if __name__ == "__main__":
    data = load_all_raw_data()
    print(f"Loaded {data['total_matches']} matches, {data['total_sets']} sets, {data['total_shots']} shots.")
    print(data['registry'][['id', 'tournament', 'category', 'winner', 'loser']].head())

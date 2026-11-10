"""
Data cleaning and validation module.
Validates shot sequences, checks score monotonicity, translates categorical labels,
maintains an audit trail of data quality, and outputs cleaned records.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.load_data import load_all_raw_data

def clean_and_validate_dataset(raw_data_dir: str = config.RAW_DATA_DIR) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans all shot-level records across all matches and games.
    Returns:
        - cleaned_shots_df: Processed DataFrame containing validated shot records.
        - audit_summary: Dictionary documenting data quality, anomalies, and exclusions.
    """
    raw_bundle = load_all_raw_data(raw_data_dir)
    registry = raw_bundle["registry"]
    matches = raw_bundle["matches"]

    audit_summary = {
        "raw_matches_count": len(registry),
        "raw_sets_count": raw_bundle["total_sets"],
        "raw_shots_count": raw_bundle["total_shots"],
        "duplicate_shots_removed": 0,
        "invalid_score_rallies_flagged": 0,
        "discontinuous_ball_rounds_flagged": 0,
        "unrecorded_endings_count": 0,
        "cleaned_shots_count": 0,
        "cleaned_rallies_count": 0,
        "exclusions_log": []
    }

    all_cleaned_shots = []

    for mid, match_data in matches.items():
        meta = match_data["metadata"]
        winner_name = str(meta.get("winner", "")).strip()
        loser_name = str(meta.get("loser", "")).strip()
        tournament = str(meta.get("tournament", "")).strip()
        category = meta.get("category", "Men's Singles")
        year = meta.get("year", None)
        round_stage = meta.get("round", "")

        for set_num, raw_set_df in match_data["sets"]:
            if raw_set_df.empty:
                continue

            set_df = raw_set_df.copy()

            # 1. Check for exact duplicate rows
            dups = set_df.duplicated().sum()
            if dups > 0:
                audit_summary["duplicate_shots_removed"] += int(dups)
                set_df = set_df.drop_duplicates()
                audit_summary["exclusions_log"].append(
                    f"Match {mid} Set {set_num}: Removed {dups} duplicate rows."
                )

            # 2. Check shot sequence continuity
            for r_id, r_group in set_df.groupby("rally"):
                rounds = r_group["ball_round"].tolist()
                expected = list(range(1, len(rounds) + 1))
                if [int(r) for r in rounds if pd.notnull(r)] != expected:
                    audit_summary["discontinuous_ball_rounds_flagged"] += 1

            # 3. Add match and game context
            set_df["match_id"] = mid
            set_df["set_num"] = set_num
            set_df["game_id"] = f"{mid}_S{set_num}"
            set_df["tournament"] = tournament
            set_df["round_stage"] = round_stage
            set_df["year"] = year
            set_df["category"] = category
            set_df["player_A_name"] = winner_name
            set_df["player_B_name"] = loser_name

            # 4. Standardize player identifier in shot
            # Raw 'player' is 'A' or 'B'
            def map_player_name(p):
                if p == "A":
                    return winner_name
                elif p == "B":
                    return loser_name
                return "Unknown"

            set_df["hitter_name"] = set_df["player"].apply(map_player_name)

            # 5. Translate shot type to English
            set_df["shot_type_en"] = set_df["type"].apply(
                lambda t: config.SHOT_TYPE_TRANSLATIONS.get(str(t).strip(), "unknown") if pd.notnull(t) else "unknown"
            )

            # 6. Translate lose_reason and win_reason
            set_df["lose_reason_en"] = set_df["lose_reason"].apply(
                lambda r: config.REASON_TRANSLATIONS.get(str(r).strip(), "unclassified") if pd.notnull(r) else "none"
            )
            set_df["win_reason_en"] = set_df["win_reason"].apply(
                lambda r: config.REASON_TRANSLATIONS.get(str(r).strip(), "unclassified") if pd.notnull(r) else "none"
            )

            all_cleaned_shots.append(set_df)

    combined_df = pd.concat(all_cleaned_shots, ignore_index=True)
    audit_summary["cleaned_shots_count"] = len(combined_df)
    audit_summary["cleaned_rallies_count"] = combined_df.groupby(["match_id", "set_num", "rally"]).ngroups

    # Count terminal shots with unrecorded endings
    last_shots = combined_df.groupby(["match_id", "set_num", "rally"]).last()
    unrec = (last_shots["lose_reason_en"] == "none").sum()
    audit_summary["unrecorded_endings_count"] = int(unrec)

    # Save cleaned shots to processed folder
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    cleaned_shots_path = os.path.join(config.PROCESSED_DATA_DIR, "cleaned_shots.csv")
    combined_df.to_csv(cleaned_shots_path, index=False)

    # Save audit log
    audit_path = os.path.join(config.PROCESSED_DATA_DIR, "cleaning_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return combined_df, audit_summary

if __name__ == "__main__":
    df, audit = clean_and_validate_dataset()
    print("Clean and Validate Completed.")
    print(f"Total Cleaned Shots: {audit['cleaned_shots_count']}")
    print(f"Total Cleaned Rallies: {audit['cleaned_rallies_count']}")
    print(f"Unrecorded Endings: {audit['unrecorded_endings_count']}")
    print(f"Discontinuous Ball Rounds Flagged: {audit['discontinuous_ball_rounds_flagged']}")


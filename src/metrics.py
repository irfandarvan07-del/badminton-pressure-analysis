"""
Metrics calculation module for Badminton Pressure Analysis.
Computes rally length distributions, unforced error rates, and stratified metrics
with explicit denominators and zero-division safeguards.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

def calculate_rate(numerator: int, denominator: int, round_digits: int = 4) -> float:
    """
    Computes percentage/rate safely.
    Returns 0.0 if denominator is 0.
    """
    if denominator is None or denominator <= 0:
        return 0.0
    if numerator is None or numerator < 0:
        return 0.0
    return round(float(numerator) / float(denominator), round_digits)

def summarize_numeric_series(series: pd.Series) -> Dict[str, float]:
    """
    Computes count, mean, std, median, IQR, min, max, p25, p75 for a numeric series.
    """
    clean_s = series.dropna()
    n = len(clean_s)
    if n == 0:
        return {
            "count": 0, "mean": 0.0, "std": 0.0,
            "median": 0.0, "iqr": 0.0, "min": 0.0,
            "max": 0.0, "p25": 0.0, "p75": 0.0
        }
    
    p25 = float(clean_s.quantile(0.25))
    p75 = float(clean_s.quantile(0.75))
    
    return {
        "count": int(n),
        "mean": round(float(clean_s.mean()), 2),
        "std": round(float(clean_s.std(ddof=1) if n > 1 else 0.0), 2),
        "median": round(float(clean_s.median()), 2),
        "iqr": round(p75 - p25, 2),
        "min": int(clean_s.min()),
        "max": int(clean_s.max()),
        "p25": round(p25, 2),
        "p75": round(p75, 2)
    }

def compute_rally_length_metrics(rallies_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall rally length summary and breakdowns by:
    - Score State (Leading vs Trailing vs Level, from player-perspective or leader state)
    - Game Phase (Before 15 vs After 15)
    - Category (Men's Singles vs Women's Singles)
    - Cross-tabulation (Game Phase x Score State)
    """
    overall = summarize_numeric_series(rallies_df["rally_length"])
    
    by_phase = {}
    for phase, grp in rallies_df.groupby("game_phase"):
        by_phase[phase] = summarize_numeric_series(grp["rally_length"])

    by_category = {}
    for cat, grp in rallies_df.groupby("category"):
        by_category[cat] = summarize_numeric_series(grp["rally_length"])

    return {
        "overall": overall,
        "by_phase": by_phase,
        "by_category": by_category,
    }

def compute_player_perspective_rally_length(perspective_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes rally length metrics by score state from player-perspective dataset.
    """
    by_state = {}
    for state in ["Leading", "Level", "Trailing"]:
        sub = perspective_df[perspective_df["score_state"] == state]
        by_state[state] = summarize_numeric_series(sub["rally_length"])

    by_state_and_phase = {}
    for (state, phase), grp in perspective_df.groupby(["score_state", "game_phase"]):
        key = f"{state} - {phase}"
        by_state_and_phase[key] = summarize_numeric_series(grp["rally_length"])

    return {
        "by_score_state": by_state,
        "by_score_state_and_phase": by_state_and_phase
    }

def compute_unforced_error_metrics(rallies_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates unforced error rates and counts with transparent denominators.
    
    Definitions:
        - Total Rallies: N
        - Classifiable Rallies: Rallies ending in net, out, fault, or winner
        - Unforced Errors: Rallies ending in net, out, or fault
        - Rate on Classifiable: unforced_errors / classifiable_rallies (Standard)
        - Rate on Total: unforced_errors / total_rallies
    """
    total = len(rallies_df)
    classifiable_df = rallies_df[rallies_df["ending_type"] != "unknown"]
    classifiable_count = len(classifiable_df)
    
    ue_mask = rallies_df["unforced_error"] == True
    ue_count = int(ue_mask.sum())
    winners_count = int((rallies_df["ending_type"] == "winner").sum())
    unrec_count = total - classifiable_count

    rate_classifiable = calculate_rate(ue_count, classifiable_count)
    rate_total = calculate_rate(ue_count, total)

    # Breakdowns by Game Phase
    phase_metrics = {}
    for phase in ["Before 15", "After 15"]:
        p_df = rallies_df[rallies_df["game_phase"] == phase]
        p_total = len(p_df)
        p_class = len(p_df[p_df["ending_type"] != "unknown"])
        p_ue = int((p_df["unforced_error"] == True).sum())
        p_rate = calculate_rate(p_ue, p_class)
        phase_metrics[phase] = {
            "total_rallies": p_total,
            "classifiable_rallies": p_class,
            "unforced_errors": p_ue,
            "unforced_error_rate": p_rate,
            "denominator_type": "classifiable_rallies_in_phase"
        }

    # Breakdowns by Category (Men's vs Women's)
    cat_metrics = {}
    for cat in ["Men's Singles", "Women's Singles"]:
        c_df = rallies_df[rallies_df["category"] == cat]
        c_total = len(c_df)
        c_class = len(c_df[c_df["ending_type"] != "unknown"])
        c_ue = int((c_df["unforced_error"] == True).sum())
        c_rate = calculate_rate(c_ue, c_class)

        # Also get phase breakdown within category
        c_phase = {}
        for phase in ["Before 15", "After 15"]:
            cp_df = c_df[c_df["game_phase"] == phase]
            cp_class = len(cp_df[cp_df["ending_type"] != "unknown"])
            cp_ue = int((cp_df["unforced_error"] == True).sum())
            c_phase[phase] = {
                "total_rallies": len(cp_df),
                "classifiable_rallies": cp_class,
                "unforced_errors": cp_ue,
                "unforced_error_rate": calculate_rate(cp_ue, cp_class),
            }

        cat_metrics[cat] = {
            "total_rallies": c_total,
            "classifiable_rallies": c_class,
            "unforced_errors": c_ue,
            "unforced_error_rate": c_rate,
            "by_phase": c_phase
        }

    # Ending type details
    ending_breakdown = rallies_df["ending_type"].value_counts().to_dict()

    return {
        "total_rallies": total,
        "classifiable_rallies": classifiable_count,
        "unforced_errors": ue_count,
        "clean_winners": winners_count,
        "unclassified_endings": unrec_count,
        "overall_unforced_error_rate_classifiable": rate_classifiable,
        "overall_unforced_error_rate_total": rate_total,
        "by_phase": phase_metrics,
        "by_category": cat_metrics,
        "ending_type_counts": ending_breakdown
    }

def generate_full_metrics_summary(
    canonical_df: Optional[pd.DataFrame] = None,
    perspective_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Computes all analytical metrics across dataset.
    """
    if canonical_df is None:
        c_path = os.path.join(config.PROCESSED_DATA_DIR, "rallies.csv")
        canonical_df = pd.read_csv(c_path)
    if perspective_df is None:
        p_path = os.path.join(config.PROCESSED_DATA_DIR, "rallies_player_perspective.csv")
        perspective_df = pd.read_csv(p_path)

    r_metrics = compute_rally_length_metrics(canonical_df)
    p_metrics = compute_player_perspective_rally_length(perspective_df)
    ue_metrics = compute_unforced_error_metrics(canonical_df)

    # Coverage counts
    coverage = {
        "matches_count": int(canonical_df["match_id"].nunique()),
        "games_count": int(canonical_df["game_id"].nunique()),
        "rallies_count": int(len(canonical_df)),
        "shots_count": int(canonical_df["rally_length"].sum()),
    }

    return {
        "coverage": coverage,
        "rally_length": r_metrics,
        "player_perspective_length": p_metrics,
        "unforced_errors": ue_metrics
    }

if __name__ == "__main__":
    summary = generate_full_metrics_summary()
    print("=" * 60)
    print("BADMINTON PRESSURE METRICS SUMMARY")
    print("=" * 60)
    print("Coverage:", summary["coverage"])
    print("\nOverall Rally Length:", summary["rally_length"]["overall"])
    print("\nRally Length by Game Phase:")
    for k, v in summary["rally_length"]["by_phase"].items():
        print(f"  {k}: Median={v['median']}, Mean={v['mean']}, IQR={v['iqr']}, N={v['count']}")
    print("\nRally Length by Score State (Player Perspective):")
    for k, v in summary["player_perspective_length"]["by_score_state"].items():
        print(f"  {k}: Median={v['median']}, Mean={v['mean']}, IQR={v['iqr']}, N={v['count']}")
    print("\nUnforced Error Rates by Phase:")
    for k, v in summary["unforced_errors"]["by_phase"].items():
        print(f"  {k}: Rate={v['unforced_error_rate']*100:.1f}%, UEs={v['unforced_errors']}/{v['classifiable_rallies']}")
    print("\nUnforced Error Rates by Category:")
    for k, v in summary["unforced_errors"]["by_category"].items():
        print(f"  {k}: Rate={v['unforced_error_rate']*100:.1f}%, UEs={v['unforced_errors']}/{v['classifiable_rallies']}")

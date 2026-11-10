"""
src/predictor.py
Pattern-based predictor that derives rally context statistics
from historical ShuttleSet data for a given score/pressure situation.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import config

# ---------------------------------------------------------------------------
# Data loader (cached at module level so it is read once per process)
# ---------------------------------------------------------------------------
_RALLY_PERSPECTIVE: pd.DataFrame | None = None
_RALLIES: pd.DataFrame | None = None


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    global _RALLY_PERSPECTIVE, _RALLIES
    if _RALLY_PERSPECTIVE is None:
        path = Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv"
        _RALLY_PERSPECTIVE = pd.read_csv(path)
    if _RALLIES is None:
        path = Path(config.PROCESSED_DATA_DIR) / "rallies.csv"
        _RALLIES = pd.read_csv(path)
    return _RALLY_PERSPECTIVE, _RALLIES


# ---------------------------------------------------------------------------
# Score-state helpers (mirrors src/score_state.py but dependency-free)
# ---------------------------------------------------------------------------

def classify_score_state(player_score: int, opponent_score: int) -> str:
    if player_score > opponent_score:
        return "Leading"
    elif player_score < opponent_score:
        return "Trailing"
    else:
        return "Level"


def classify_game_phase(player_score: int, opponent_score: int,
                         threshold: int = 15) -> str:
    leader_score = max(player_score, opponent_score)
    return "After 15" if leader_score >= threshold else "Before 15"


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_rally_context(
    player_score: int,
    opponent_score: int,
    category: str = "Men's Singles",
    game_phase_threshold: int = 15,
) -> dict:
    """
    Given a pre-rally score, return historical pressure statistics.

    Parameters
    ----------
    player_score      : pre-rally score of the focal player
    opponent_score    : pre-rally score of the opponent
    category          : "Men's Singles" | "Women's Singles" | "All"
    game_phase_threshold : default 15 (from config)

    Returns
    -------
    dict with keys:
      score_state, game_phase,
      median_rally_length, mean_rally_length,
      error_rate, ci_low, ci_high, sample_size,
      all_states_summary (DataFrame row for reference)
    """
    rp, _ = _load_data()
    threshold = game_phase_threshold or config.GAME_PHASE_THRESHOLD

    score_state = classify_score_state(player_score, opponent_score)
    game_phase  = classify_game_phase(player_score, opponent_score, threshold)

    # Build filter mask
    mask = (rp["score_state"] == score_state) & (rp["game_phase"] == game_phase)
    if category != "All":
        mask &= (rp["category"] == category)

    subset = rp[mask].copy()
    n = len(subset)

    if n == 0:
        # Fallback: relax game_phase
        mask2 = rp["score_state"] == score_state
        if category != "All":
            mask2 &= rp["category"] == category
        subset = rp[mask2].copy()
        n = len(subset)
        fallback = True
    else:
        fallback = False

    if n == 0:
        return {
            "score_state": score_state,
            "game_phase": game_phase,
            "median_rally_length": None,
            "mean_rally_length": None,
            "error_rate": None,
            "ci_low": None,
            "ci_high": None,
            "sample_size": 0,
            "fallback": True,
            "warning": "No historical data for this combination.",
        }

    lengths   = subset["rally_length"].dropna()
    med_len   = float(lengths.median())
    mean_len  = float(lengths.mean())

    # Error rate among classifiable rallies
    classifiable = subset[subset["classifiable_ending"] == True]
    nc = len(classifiable)
    errors = int(classifiable["unforced_error"].sum()) if nc > 0 else 0
    error_rate = errors / nc if nc > 0 else None

    # 95 % Wilson CI for error rate
    if nc > 0 and error_rate is not None:
        z = 1.96
        centre = (errors + z**2 / 2) / (nc + z**2)
        margin  = z * np.sqrt(errors * (nc - errors) / nc + z**2 / 4) / (nc + z**2)
        ci_low  = max(0.0, centre - margin)
        ci_high = min(1.0, centre + margin)
    else:
        ci_low = ci_high = None

    return {
        "score_state":        score_state,
        "game_phase":         game_phase,
        "median_rally_length": med_len,
        "mean_rally_length":  mean_len,
        "error_rate":         error_rate,
        "ci_low":             ci_low,
        "ci_high":            ci_high,
        "sample_size":        n,
        "classifiable_n":     nc,
        "fallback":           fallback,
        "warning":            "Game phase relaxed (no data for exact phase)." if fallback else None,
    }


# ---------------------------------------------------------------------------
# Comparative summary: all score states for a given phase + category
# ---------------------------------------------------------------------------

def get_all_states_summary(
    category: str = "Men's Singles",
    game_phase: str | None = None,
) -> pd.DataFrame:
    """
    Returns a summary DataFrame with one row per (score_state, game_phase)
    combination, showing median rally length, mean rally length, error_rate,
    and sample_size.
    """
    rp, _ = _load_data()

    mask = pd.Series([True] * len(rp), index=rp.index)
    if category != "All":
        mask &= rp["category"] == category
    if game_phase:
        mask &= rp["game_phase"] == game_phase

    subset = rp[mask]

    rows = []
    for (ss, gp), grp in subset.groupby(["score_state", "game_phase"]):
        cl = grp[grp["classifiable_ending"] == True]
        nc = len(cl)
        er = cl["unforced_error"].sum() / nc if nc > 0 else None
        rows.append({
            "Score State":    ss,
            "Game Phase":     gp,
            "Median Rally":   round(grp["rally_length"].median(), 1),
            "Mean Rally":     round(grp["rally_length"].mean(), 1),
            "Error Rate (%)": round(er * 100, 1) if er is not None else None,
            "N (rallies)":    len(grp),
        })

    return pd.DataFrame(rows).sort_values(["Game Phase", "Score State"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shot sequence tracker  (simple rally-in-progress helper)
# ---------------------------------------------------------------------------

PRESSURE_THRESHOLDS = {
    "Trailing": {"error_boost": 1.22,  "length_factor": 0.95},
    "Level":    {"error_boost": 1.00,  "length_factor": 1.00},
    "Leading":  {"error_boost": 0.85,  "length_factor": 1.05},
}


def score_pressure_level(error_rate: float | None, score_state: str) -> str:
    """Return a qualitative pressure label."""
    if error_rate is None:
        return "Unknown"
    pct = error_rate * 100
    if pct >= 35:
        return "🔴 High"
    elif pct >= 25:
        return "🟠 Moderate"
    else:
        return "🟢 Low"


def get_match_history_trends(match_log: list[dict]) -> pd.DataFrame:
    """
    Convert a list of rally log dicts into a DataFrame suitable for
    displaying a live match trend chart.

    Each dict should have: rally_num, score_a, score_b, winner (A/B), rally_length
    """
    if not match_log:
        return pd.DataFrame(columns=["Rally", "Score A", "Score B", "Winner", "Length"])
    df = pd.DataFrame(match_log)
    df.rename(columns={
        "rally_num": "Rally", "score_a": "Score A",
        "score_b": "Score B", "winner": "Winner", "rally_length": "Length"
    }, inplace=True)
    return df

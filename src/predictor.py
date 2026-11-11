"""
src/predictor.py
Pattern-based predictor that derives rally context statistics
from historical ShuttleSet data for a given score/pressure situation.

Metric definitions
------------------
unforced_error_rate :=
    (# rallies in this cell where THIS PLAYER committed an unforced error)
    / (# rallies in this cell with a classifiable ending)

Unforced errors = out of bounds, net fault, service fault.
Rallies with missing or unclassifiable endings are EXCLUDED from the
denominator. The fraction excluded is reported as `unclassifiable_pct`.

Score state and game phase are computed from PRE-RALLY scores (the
previous rally's post-score; 0-0 for the first rally of each set).
This avoids lookahead bias. See tests/test_lookahead_bias.py.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
_RALLY_PERSPECTIVE: "pd.DataFrame | None" = None
_RALLIES: "pd.DataFrame | None" = None


def _load_data() -> tuple:
    global _RALLY_PERSPECTIVE, _RALLIES
    if _RALLY_PERSPECTIVE is None:
        path = Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv"
        _RALLY_PERSPECTIVE = pd.read_csv(path)
    if _RALLIES is None:
        path = Path(config.PROCESSED_DATA_DIR) / "rallies.csv"
        _RALLIES = pd.read_csv(path)
    return _RALLY_PERSPECTIVE, _RALLIES


# ---------------------------------------------------------------------------
# Score-state helpers (stateless — mirrors src/score_state.py)
# ---------------------------------------------------------------------------

def classify_score_state(player_score: int, opponent_score: int) -> str:
    if player_score > opponent_score:
        return "Leading"
    elif player_score < opponent_score:
        return "Trailing"
    return "Level"


def classify_game_phase(player_score: int, opponent_score: int,
                         threshold: int = 15) -> str:
    leader_score = max(player_score, opponent_score)
    return "After 15" if leader_score >= threshold else "Before 15"


def classify_deuce(player_score: int, opponent_score: int) -> bool:
    """True when both players have >= 20 points and are tied (true deuce)."""
    return player_score == opponent_score and player_score >= 20


# ---------------------------------------------------------------------------
# Wilson CI (kept as secondary reference alongside bootstrap)
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a proportion. Returns (lo, hi)."""
    if n == 0:
        return None, None
    centre = (successes + z ** 2 / 2) / (n + z ** 2)
    margin = z * np.sqrt(successes * (n - successes) / n + z ** 2 / 4) / (n + z ** 2)
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Score-margin bins (P1.7)
# ---------------------------------------------------------------------------

def _margin_bin(score_diff: int) -> str:
    ad = abs(score_diff)
    if score_diff == 0:
        return "Tied"
    elif 1 <= ad <= 2:
        return "±1-2"
    elif 3 <= ad <= 5:
        return "±3-5"
    return "±6+"


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict_rally_context(
    player_score: int,
    opponent_score: int,
    category: str = "Men's Singles",
    game_phase_threshold: int = 15,
    min_n_margin: int = 100,
) -> dict:
    """
    Given a PRE-RALLY score, return historical pressure statistics.

    Parameters
    ----------
    player_score      : pre-rally score of the focal player
    opponent_score    : pre-rally score of the opponent
    category          : "Men's Singles" | "Women's Singles" | "All"
    game_phase_threshold : default 15 (from config)
    min_n_margin      : minimum N to use margin-bin level; falls back to
                        coarser cell if below this threshold.

    Returns
    -------
    dict — see field list in docstring body.
    """
    rp, _ = _load_data()
    threshold = game_phase_threshold or config.GAME_PHASE_THRESHOLD

    score_state   = classify_score_state(player_score, opponent_score)
    game_phase    = classify_game_phase(player_score, opponent_score, threshold)
    is_deuce      = classify_deuce(player_score, opponent_score)
    score_diff    = player_score - opponent_score
    margin_bin    = _margin_bin(score_diff)

    # ── Try margin-bin level first ──────────────────────────────────────────
    rp2 = rp.copy()
    rp2["margin_bin"] = rp2["score_diff"].apply(_margin_bin)

    def _base_mask(df):
        m = (df["score_state"] == score_state) & (df["game_phase"] == game_phase)
        if category != "All":
            m &= df["category"] == category
        return m

    mask_fine = _base_mask(rp2) & (rp2["margin_bin"] == margin_bin)
    subset_fine = rp2[mask_fine]

    if len(subset_fine) >= min_n_margin:
        subset = subset_fine.copy()
        resolution = "margin_bin"
        fallback_used = False
        fallback_cell = None
    else:
        # ── Fallback: coarse (score_state × game_phase) ─────────────────────
        mask_coarse = _base_mask(rp)
        subset = rp[mask_coarse].copy()
        resolution = "score_state×phase"
        fallback_used = (len(subset_fine) > 0 and len(subset_fine) < min_n_margin) or (len(subset_fine) == 0)
        fallback_cell = f"{score_state} × {game_phase}" if fallback_used else None

    n = len(subset)

    if n == 0:
        # Last resort: relax game_phase
        mask_relax = rp["score_state"] == score_state
        if category != "All":
            mask_relax &= rp["category"] == category
        subset = rp[mask_relax].copy()
        n = len(subset)
        resolution = "score_state_only"
        fallback_used = True
        fallback_cell = f"{score_state} (phase relaxed)"

    if n == 0:
        return {
            "score_state": score_state, "game_phase": game_phase,
            "is_deuce": is_deuce, "margin_bin": margin_bin,
            "median_rally_length": None, "rally_iqr": None,
            "mean_rally_length": None, "rally_skew": None,
            "error_rate": None, "ci_low": None, "ci_high": None,
            "wilson_lo": None, "wilson_hi": None,
            "sample_size": 0, "classifiable_n": 0,
            "unclassifiable_pct": None,
            "n_matches": 0,
            "resolution": "no_data", "fallback_used": True,
            "fallback_cell": "No data at all",
            "low_confidence": True,
            "warning": "No historical data for this combination.",
        }

    # ── Compute statistics ──────────────────────────────────────────────────
    lengths   = subset["rally_length"].dropna()
    med_len   = float(lengths.median())
    mean_len  = float(lengths.mean())
    iqr       = float(lengths.quantile(0.75) - lengths.quantile(0.25))
    skew      = float(lengths.skew())

    n_total = len(subset)
    classifiable = subset[subset["classifiable_ending"] == True]
    nc = len(classifiable)
    n_unclassifiable = n_total - nc
    unclassifiable_pct = round(n_unclassifiable / n_total * 100, 1) if n_total else None

    errors_raw = pd.to_numeric(classifiable["unforced_error"], errors="coerce")
    errors = int(errors_raw.sum()) if nc > 0 else 0
    error_rate = errors / nc if nc > 0 else None

    wilson_lo, wilson_hi = _wilson_ci(errors, nc)

    n_matches = int(subset["match_id"].nunique())
    low_confidence = (nc < 100)

    # Deuce-specific warning
    warning = None
    if is_deuce:
        warning = f"True deuce (both ≥ 20, tied). N={nc} classifiable rallies — treat with caution."
    elif low_confidence:
        warning = f"Low N (classifiable={nc} < 100). Estimates unreliable; widen to All category for more data."
    elif fallback_used:
        warning = f"Margin-bin N too small ({len(subset_fine)}). Fell back to: {fallback_cell}."

    return {
        "score_state":          score_state,
        "game_phase":           game_phase,
        "is_deuce":             is_deuce,
        "margin_bin":           margin_bin,
        "median_rally_length":  med_len,
        "rally_iqr":            iqr,
        "mean_rally_length":    mean_len,
        "rally_skew":           skew,
        "error_rate":           error_rate,
        "ci_low":               wilson_lo,       # Wilson (secondary)
        "ci_high":              wilson_hi,
        "wilson_lo":            wilson_lo,
        "wilson_hi":            wilson_hi,
        "sample_size":          n,
        "classifiable_n":       nc,
        "unclassifiable_pct":   unclassifiable_pct,
        "n_matches":            n_matches,
        "resolution":           resolution,
        "fallback_used":        fallback_used,
        "fallback_cell":        fallback_cell,
        "low_confidence":       low_confidence,
        "warning":              warning,
    }


# ---------------------------------------------------------------------------
# Full-cell summary (now uses label "Tied, after 15" for level/after-15 cell)
# ---------------------------------------------------------------------------

def get_all_states_summary(
    category: str = "Men's Singles",
    game_phase: "str | None" = None,
) -> pd.DataFrame:
    """
    Returns one row per (score_state, game_phase) with point estimates.
    The "Level · After 15" cell is relabelled "Tied, after 15".
    True deuce (both >= 20, tied) is flagged separately.
    """
    rp, _ = _load_data()

    mask = pd.Series(True, index=rp.index)
    if category != "All":
        mask &= rp["category"] == category
    if game_phase:
        mask &= rp["game_phase"] == game_phase

    subset = rp[mask].copy()
    subset["margin_bin"] = subset["score_diff"].apply(_margin_bin)

    rows = []
    for (ss, gp), grp in subset.groupby(["score_state", "game_phase"]):
        # Rename Level × After 15
        display_state = "Tied, after 15" if (ss == "Level" and gp == "After 15") else ss

        n_total = len(grp)
        cl = grp[grp["classifiable_ending"] == True]
        nc = len(cl)
        errors = int(pd.to_numeric(cl["unforced_error"], errors="coerce").sum()) if nc else 0
        er = errors / nc if nc else None
        wlo, whi = _wilson_ci(errors, nc)

        q1 = grp["rally_length"].quantile(0.25)
        q3 = grp["rally_length"].quantile(0.75)

        rows.append({
            "Score State":          display_state,
            "Game Phase":           gp,
            "Median Rally":         round(grp["rally_length"].median(), 1),
            "IQR":                  round(q3 - q1, 1),
            "Mean Rally":           round(grp["rally_length"].mean(), 1),
            "Skew":                 round(grp["rally_length"].skew(), 2),
            "Error Rate (%)":       round(er * 100, 1) if er is not None else None,
            "Wilson CI":            f"[{wlo*100:.1f}–{whi*100:.1f}%]" if wlo else "—",
            "N (rallies)":          n_total,
            "N (classifiable)":     nc,
            "Unclassifiable (%)":   round((n_total - nc) / n_total * 100, 1) if n_total else None,
            "N Matches":            grp["match_id"].nunique(),
            "Low Confidence":       nc < 100,
        })

    # Also add true-deuce row
    deuce_mask = mask & (rp["player_score"] >= 20) & (rp["score_diff"] == 0)
    if deuce_mask.sum() > 0:
        dg = rp[deuce_mask]
        cl_d = dg[dg["classifiable_ending"] == True]
        nc_d = len(cl_d)
        er_d = int(pd.to_numeric(cl_d["unforced_error"], errors="coerce").sum()) / nc_d if nc_d else None
        wlo_d, whi_d = _wilson_ci(int(pd.to_numeric(cl_d["unforced_error"], errors="coerce").sum()), nc_d) if nc_d else (None, None)
        rows.append({
            "Score State":          "⚠️ True Deuce (≥20–≥20)",
            "Game Phase":           "After 15",
            "Median Rally":         round(dg["rally_length"].median(), 1),
            "IQR":                  round(dg["rally_length"].quantile(0.75) - dg["rally_length"].quantile(0.25), 1),
            "Mean Rally":           round(dg["rally_length"].mean(), 1),
            "Skew":                 round(dg["rally_length"].skew(), 2),
            "Error Rate (%)":       round(er_d * 100, 1) if er_d else None,
            "Wilson CI":            f"[{wlo_d*100:.1f}–{whi_d*100:.1f}%]" if wlo_d else "—",
            "N (rallies)":          len(dg),
            "N (classifiable)":     nc_d,
            "Unclassifiable (%)":   round((len(dg) - nc_d) / len(dg) * 100, 1),
            "N Matches":            dg["match_id"].nunique(),
            "Low Confidence":       True,
        })

    return pd.DataFrame(rows).sort_values(["Game Phase", "Score State"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Match log helper (rally tracker)
# ---------------------------------------------------------------------------

def get_match_history_trends(match_log: list) -> pd.DataFrame:
    if not match_log:
        return pd.DataFrame(columns=["Rally", "Score A", "Score B", "Winner", "Length"])
    df = pd.DataFrame(match_log)
    df.rename(columns={
        "rally_num": "Rally", "score_a": "Score A",
        "score_b": "Score B", "winner": "Winner", "rally_length": "Length"
    }, inplace=True)
    return df

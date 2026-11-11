"""
src/player_strength.py
Controls for player strength and match-level clustering when estimating
the score state effect on unforced errors.

Implements P2.8:
Model: error ~ score_state + game_phase + category + (1 | player) + (1 | match)
using regularized logistic regression (L2 penalization on player and match intercepts,
equivalent to Gaussian random effects), with cluster-bootstrap confidence intervals
and Odds Ratios.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import config

RNG_SEED = 42

def fit_player_strength_model(
    rp: pd.DataFrame,
    n_boot: int = 500,
    seed: int = RNG_SEED,
) -> dict:
    """
    Fits regularized logistic regression controlling for player strength and match context.
    Evaluates whether the Trailing/Level vs Leading score state effect remains statistically
    significant after controlling for player identity and match selection bias.
    """
    cl = rp[rp["classifiable_ending"] == True].copy()
    cl["y"] = pd.to_numeric(cl["unforced_error"], errors="coerce").fillna(0).astype(int)

    # Reference levels:
    # score_state: Leading (reference)
    # game_phase: Before 15 (reference)
    # category: Men's Singles (reference)
    cl["is_trailing"] = (cl["score_state"] == "Trailing").astype(int)
    cl["is_level"] = (cl["score_state"] == "Level").astype(int)
    cl["is_after_15"] = (cl["game_phase"] == "After 15").astype(int)
    cl["is_ws"] = (cl["category"] == "Women's Singles").astype(int)

    core_cols = ["is_trailing", "is_level", "is_after_15", "is_ws"]
    
    # One-hot encode player_id and match_id to control for player skill and match intercepts
    player_dummies = pd.get_dummies(cl["player_id"], prefix="player", drop_first=True, dtype=float)
    match_dummies = pd.get_dummies(cl["match_id"], prefix="match", drop_first=True, dtype=float)

    X_core = cl[core_cols].astype(float)
    X_full = pd.concat([X_core, player_dummies, match_dummies], axis=1)
    y = cl["y"].values

    # Fit L2 regularized logistic regression
    # C=1.0 puts a standard Gaussian prior on dummy coefficients
    clf = LogisticRegression(C=1.0, max_iter=1000, penalty="l2", solver="lbfgs", random_state=seed)
    clf.fit(X_full, y)

    coef_dict = dict(zip(core_cols, clf.coef_[0][:len(core_cols)]))
    or_dict = {k: float(np.exp(v)) for k, v in coef_dict.items()}

    # Cluster-bootstrap over matches to get robust 95% CIs for core coefficients
    unique_matches = cl["match_id"].unique()
    n_matches = len(unique_matches)
    rng = np.random.default_rng(seed)

    boot_coefs = {col: [] for col in core_cols}

    # Group dataframe indices by match for fast resampling
    match_indices = {m: cl.index[cl["match_id"] == m].to_numpy() for m in unique_matches}

    for _ in range(n_boot):
        sampled_m = rng.choice(unique_matches, size=n_matches, replace=True)
        sampled_idx = np.concatenate([match_indices[m] for m in sampled_m])
        X_boot = X_full.loc[sampled_idx]
        y_boot = y[cl.index.get_indexer(sampled_idx)]
        
        # Fit with low tolerance for speed
        b_clf = LogisticRegression(C=1.0, max_iter=300, penalty="l2", solver="lbfgs", tol=1e-3)
        b_clf.fit(X_boot, y_boot)
        for i, col in enumerate(core_cols):
            boot_coefs[col].append(b_clf.coef_[0][i])

    results = []
    names = {
        "is_trailing": "Score State: Trailing vs Leading",
        "is_level": "Score State: Level vs Leading",
        "is_after_15": "Game Phase: After 15 vs Before 15",
        "is_ws": "Category: Women's vs Men's Singles",
    }

    for col in core_cols:
        arr = np.array(boot_coefs[col])
        coef_val = float(coef_dict[col])
        or_val = float(or_dict[col])
        ci_coef_low = float(np.percentile(arr, 2.5))
        ci_coef_high = float(np.percentile(arr, 97.5))
        or_low = float(np.exp(ci_coef_low))
        or_high = float(np.exp(ci_coef_high))

        # Check if CI excludes 0 for coef (or 1.0 for OR)
        sig = bool(ci_coef_low > 0 or ci_coef_high < 0)

        results.append({
            "Predictor": names[col],
            "Coefficient": round(coef_val, 3),
            "Odds Ratio (OR)": round(or_val, 3),
            "95% CI (OR)": f"[{or_low:.3f}, {or_high:.3f}]",
            "Significant (excludes 1.0)": sig,
        })

    res_df = pd.DataFrame(results)
    return {
        "summary_table": res_df,
        "n_obs": len(cl),
        "n_matches": n_matches,
        "n_players": cl["player_id"].nunique(),
        "trailing_or": or_dict["is_trailing"],
        "trailing_survives": bool(res_df.loc[res_df["Predictor"].str.contains("Trailing"), "Significant (excludes 1.0)"].values[0]),
    }

if __name__ == "__main__":
    rp = pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv")
    res = fit_player_strength_model(rp, n_boot=200)
    print("\n=== Player Strength Control Model ===")
    print(res["summary_table"].to_string(index=False))
    print(f"\nTrailing effect survives control: {res['trailing_survives']}")


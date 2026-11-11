"""
src/holdout_eval.py
P1.4 — Leave-one-match-out holdout evaluation.

Compares three baselines against the lookup-table predictor:
  (a) Global error rate
  (b) Category-only rate
  (c) score_state × game_phase lookup (our predictor)

Metrics: log loss, Brier score (rally-level binary prediction).
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

import config

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-7

def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yp = np.clip(y_pred, _EPS, 1 - _EPS)
    return float(-np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp)))


def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Lookup table builder (trained on train set)
# ─────────────────────────────────────────────────────────────────────────────

def _build_lookup(train_df: pd.DataFrame, category: str) -> dict:
    """
    Returns dict[(score_state, game_phase)] -> error_rate
    trained only on train_df (classifiable rows, optionally filtered by category).
    """
    if category != "All":
        train_df = train_df[train_df["category"] == category]
    cl = train_df[train_df["classifiable_ending"] == True].copy()
    cl["unforced_error"] = pd.to_numeric(cl["unforced_error"], errors="coerce")
    lookup = {}
    for (ss, gp), grp in cl.groupby(["score_state", "game_phase"]):
        lookup[(ss, gp)] = float(grp["unforced_error"].mean())
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-match-out CV
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_match_out(rp: pd.DataFrame) -> pd.DataFrame:
    """
    For each held-out match, train on remaining matches, predict on
    classifiable rows of held-out match, compare models.

    Returns a DataFrame with one row per held-out rally (classifiable only).
    """
    rp = rp.copy()
    rp["unforced_error_num"] = pd.to_numeric(rp["unforced_error"], errors="coerce")
    classifiable = rp[rp["classifiable_ending"] == True].copy()

    matches = classifiable["match_id"].unique()
    result_rows = []

    for held_match in matches:
        train = classifiable[classifiable["match_id"] != held_match]
        test  = classifiable[classifiable["match_id"] == held_match]
        if len(test) == 0:
            continue

        # Baselines
        global_rate  = float(train["unforced_error_num"].mean())
        cat_rates    = train.groupby("category")["unforced_error_num"].mean().to_dict()
        lookup_ms    = _build_lookup(train, "Men's Singles")
        lookup_ws    = _build_lookup(train, "Women's Singles")
        lookup_all   = _build_lookup(train, "All")

        for _, row in test.iterrows():
            cat    = row["category"]
            ss     = row["score_state"]
            gp     = row["game_phase"]
            y_true = row["unforced_error_num"]
            if np.isnan(y_true):
                continue

            # (a) Global baseline
            pred_global = global_rate

            # (b) Category baseline
            pred_cat = cat_rates.get(cat, global_rate)

            # (c) Lookup: use category-specific lookup, fall back to All
            cat_lookup = lookup_ms if "Men" in cat else lookup_ws
            pred_lookup_cat = cat_lookup.get((ss, gp), pred_cat)
            pred_lookup_all = lookup_all.get((ss, gp), global_rate)

            result_rows.append({
                "match_id":        row["match_id"],
                "category":        cat,
                "score_state":     ss,
                "game_phase":      gp,
                "y_true":          y_true,
                "pred_global":     pred_global,
                "pred_category":   pred_cat,
                "pred_lookup_cat": pred_lookup_cat,
                "pred_lookup_all": pred_lookup_all,
            })

    return pd.DataFrame(result_rows)


def evaluate_holdout(rp: pd.DataFrame, save_calibration_plot: bool = True) -> dict:
    """
    Run LOMO CV and return summary metrics for all models,
    optionally saving a calibration plot.
    """
    results = leave_one_match_out(rp)
    if results.empty:
        return {}

    y = results["y_true"].values
    models = {
        "Global baseline":           results["pred_global"].values,
        "Category baseline":         results["pred_category"].values,
        "Lookup (category-specific)": results["pred_lookup_cat"].values,
        "Lookup (All categories)":   results["pred_lookup_all"].values,
    }

    summary = {}
    for name, preds in models.items():
        summary[name] = {
            "log_loss":    round(_log_loss(y, preds), 4),
            "brier_score": round(_brier_score(y, preds), 4),
            "n_rallies":   int(len(y)),
        }

    if save_calibration_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")

        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfectly calibrated")
        colors = {
            "Global baseline": "#94a3b8",
            "Category baseline": "#38bdf8",
            "Lookup (category-specific)": "#34d399",
            "Lookup (All categories)": "#fbbf24",
        }

        for name, preds in models.items():
            prob_true, prob_pred = calibration_curve(y, preds, n_bins=5, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name, color=colors.get(name, "#e2e8f0"))

        ax.set_xlabel("Mean Predicted Probability", color="#e2e8f0", fontsize=10)
        ax.set_ylabel("Fraction of Positives (Observed Error Rate)", color="#e2e8f0", fontsize=10)
        ax.set_title("Calibration Curve (Leave-One-Match-Out)", color="#f1f5f9", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        fig.patch.set_facecolor("#0f172a")
        plt.tight_layout()
        out_path = Path(config.FIGURES_DIR) / "7_holdout_calibration.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    return summary


def print_holdout_report(rp: pd.DataFrame) -> None:
    summary = evaluate_holdout(rp)
    print("\n" + "="*65)
    print("  HOLDOUT EVALUATION  (Leave-One-Match-Out)")
    print("="*65)
    print(f"  {'Model':<35} {'Log Loss':>9} {'Brier':>9}")
    print("-"*65)
    best_ll = min(v["log_loss"] for v in summary.values())
    for name, m in summary.items():
        marker = " ←" if m["log_loss"] == best_ll else ""
        print(f"  {name:<35} {m['log_loss']:>9.4f} {m['brier_score']:>9.4f}{marker}")
    print(f"\n  Evaluated on {list(summary.values())[0]['n_rallies']:,} classifiable rallies")
    print("="*65)


if __name__ == "__main__":
    rp = pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv")
    print_holdout_report(rp)


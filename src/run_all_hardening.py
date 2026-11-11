"""
src/run_all_hardening.py
Master script to execute and verify all hardening requirements (P0 - P3).
Generates bootstrap distributions, holdout cross-validation, player strength models,
and saves tables to reports/tables/ and figures to reports/figures/.
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import config
from src.bootstrap import (
    build_bootstrap_summary,
    build_margin_summary,
    bootstrap_claims_test_error_rate,
    bootstrap_claims_test_median_length,
)
from src.holdout_eval import evaluate_holdout
from src.player_strength import fit_player_strength_model
from src.predictor import get_all_states_summary

def main():
    print("Loading data...")
    rp = pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv")
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    print("\n--- 1. Cluster-Bootstrap Summary (2,000 iterations) ---")
    boot_ms = build_bootstrap_summary(rp, category="Men's Singles", n_boot=2000)
    boot_ws = build_bootstrap_summary(rp, category="Women's Singles", n_boot=2000)
    boot_all = build_bootstrap_summary(rp, category="All", n_boot=2000)

    boot_ms.to_csv(Path(config.TABLES_DIR) / "bootstrap_summary_ms.csv", index=False)
    boot_ws.to_csv(Path(config.TABLES_DIR) / "bootstrap_summary_ws.csv", index=False)
    boot_all.to_csv(Path(config.TABLES_DIR) / "bootstrap_summary_all.csv", index=False)
    print("Men's Singles Bootstrap Summary:")
    print(boot_ms[["Score State","Game Phase","N (rallies)","N (classifiable)","Error Rate","Boot CI Low (ER%)","Boot CI High (ER%)","Wilson CI Low (ER%)","Wilson CI High (ER%)","Median Rally","Med CI Low","Med CI High","Confidence"]].to_string(index=False))

    print("\n--- 2. Holdout Evaluation (Leave-One-Match-Out) ---")
    holdout_res = evaluate_holdout(rp, save_calibration_plot=True)
    for model_name, metrics in holdout_res.items():
        print(f"  {model_name:<30}: Log Loss = {metrics['log_loss']:.4f}, Brier = {metrics['brier_score']:.4f}")

    with open(Path(config.TABLES_DIR) / "holdout_evaluation.json", "w") as f:
        json.dump(holdout_res, f, indent=2)

    print("\n--- 3. Testing Empirical Claims with Cluster-Bootstrap (2,000 iterations) ---")
    ms_cl = rp[(rp["category"] == "Men's Singles") & (rp["classifiable_ending"] == True)]
    trailing_b15 = ms_cl[(ms_cl["score_state"] == "Trailing") & (ms_cl["game_phase"] == "Before 15")]
    leading_b15  = ms_cl[(ms_cl["score_state"] == "Leading")  & (ms_cl["game_phase"] == "Before 15")]
    trailing_a15 = ms_cl[(ms_cl["score_state"] == "Trailing") & (ms_cl["game_phase"] == "After 15")]
    leading_a15  = ms_cl[(ms_cl["score_state"] == "Leading")  & (ms_cl["game_phase"] == "After 15")]
    
    ms_all = rp[rp["category"] == "Men's Singles"]
    ws_all = rp[rp["category"] == "Women's Singles"]

    # Claim 1: Trailing > Leading (Before 15, MS)
    t1 = bootstrap_claims_test_error_rate(trailing_b15, leading_b15, n_boot=2000)
    # Claim 2: Trailing > Leading (After 15, MS)
    t2 = bootstrap_claims_test_error_rate(trailing_a15, leading_a15, n_boot=2000)
    # Claim 3: Leaders get cleaner after 15 (Leading B15 - Leading A15, MS)
    t3 = bootstrap_claims_test_error_rate(leading_b15, leading_a15, n_boot=2000)
    # Claim 4: MS rallies longer than WS median (All phases)
    t4 = bootstrap_claims_test_median_length(ms_all, ws_all, n_boot=2000)

    claims_data = [
        {
            "Claim": "Trailing vs Leading error rate (MS, Before 15)",
            "Observed Diff": f"{t1['observed_diff']*100:+.1f} pp",
            "95% Cluster-Bootstrap CI": f"[{t1['ci_low_diff']*100:.1f}, {t1['ci_high_diff']*100:.1f}] pp",
            "Bootstrap p-value": f"{t1['p_value']:.4f}",
            "CI Excludes Zero": t1['ci_excludes_zero'],
            "Supported": "Supported (association)" if t1['ci_excludes_zero'] else "Inconclusive",
        },
        {
            "Claim": "Trailing vs Leading error rate (MS, After 15)",
            "Observed Diff": f"{t2['observed_diff']*100:+.1f} pp",
            "95% Cluster-Bootstrap CI": f"[{t2['ci_low_diff']*100:.1f}, {t2['ci_high_diff']*100:.1f}] pp",
            "Bootstrap p-value": f"{t2['p_value']:.4f}",
            "CI Excludes Zero": t2['ci_excludes_zero'],
            "Supported": "Supported (association)" if t2['ci_excludes_zero'] else "Inconclusive",
        },
        {
            "Claim": "Leaders error rate drops after 15 (MS)",
            "Observed Diff": f"{t3['observed_diff']*100:+.1f} pp",
            "95% Cluster-Bootstrap CI": f"[{t3['ci_low_diff']*100:.1f}, {t3['ci_high_diff']*100:.1f}] pp",
            "Bootstrap p-value": f"{t3['p_value']:.4f}",
            "CI Excludes Zero": t3['ci_excludes_zero'],
            "Supported": "Supported (association)" if t3['ci_excludes_zero'] else "Inconclusive",
        },
        {
            "Claim": "MS median rally length longer than WS",
            "Observed Diff": f"{t4['observed_diff']:+.1f} shots",
            "95% Cluster-Bootstrap CI": f"[{t4['ci_low_diff']:.1f}, {t4['ci_high_diff']:.1f}] shots",
            "Bootstrap p-value": f"{t4['p_value']:.4f}",
            "CI Excludes Zero": t4['ci_excludes_zero'],
            "Supported": "Supported (association)" if t4['ci_excludes_zero'] else "Inconclusive",
        },
    ]

    claims_df = pd.DataFrame(claims_data)
    claims_df.to_csv(Path(config.TABLES_DIR) / "claims_verification.csv", index=False)
    print(claims_df.to_string(index=False))

    print("\n--- 4. Player Strength & Selection Bias Control Model ---")
    ps_model = fit_player_strength_model(rp, n_boot=500)
    ps_model["summary_table"].to_csv(Path(config.TABLES_DIR) / "player_strength_model.csv", index=False)
    print(ps_model["summary_table"].to_string(index=False))
    print(f"Trailing effect survives player/match control: {ps_model['trailing_survives']}")

    print("\n--- 5. Score Margin Bins & Deuce Cell ---")
    margin_ms = build_margin_summary(rp, category="Men's Singles", min_n=100)
    margin_ms.to_csv(Path(config.TABLES_DIR) / "margin_bins_ms.csv", index=False)
    print(margin_ms.to_string(index=False))

    print("\nAll hardening computations completed successfully.")

if __name__ == "__main__":
    main()

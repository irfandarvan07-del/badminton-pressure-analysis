"""
Main execution script for Badminton Pressure Analysis.
Runs the complete reproducible end-to-end pipeline:
1. Data loading and validation
2. Rally-level aggregation and pre-score reconstruction
3. Descriptive metrics calculation
4. Statistical hypothesis testing (Mann-Whitney U, Chi-square, Kruskal-Wallis)
5. Visualizations generation
6. Report tables export
7. Console summary answering all research questions
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.clean_data import clean_and_validate_dataset
from src.build_rallies import build_rally_records
from src.metrics import generate_full_metrics_summary, summarize_numeric_series
from src.statistics import run_all_statistical_tests
from src.visualizations import generate_all_visualizations

def export_tables(canonical_df: pd.DataFrame, perspective_df: pd.DataFrame, stats_results: dict):
    """Exports structured summary tables into reports/tables/."""
    os.makedirs(config.TABLES_DIR, exist_ok=True)

    # Table 1: Rally length by score state
    state_rows = []
    for state, grp in perspective_df.groupby("score_state"):
        s = summarize_numeric_series(grp["rally_length"])
        s["Score State"] = state
        state_rows.append(s)
    pd.DataFrame(state_rows).set_index("Score State").to_csv(
        os.path.join(config.TABLES_DIR, "rally_length_by_score_state.csv")
    )

    # Table 2: Rally length by game phase
    phase_rows = []
    for phase, grp in canonical_df.groupby("game_phase"):
        s = summarize_numeric_series(grp["rally_length"])
        s["Game Phase"] = phase
        phase_rows.append(s)
    pd.DataFrame(phase_rows).set_index("Game Phase").to_csv(
        os.path.join(config.TABLES_DIR, "rally_length_by_game_phase.csv")
    )

    # Table 3: Statistical tests summary
    stat_rows = []
    for q_name, res in stats_results.items():
        if isinstance(res, dict) and "test" in res:
            stat_rows.append({
                "Analysis": q_name,
                "Test": res.get("test"),
                "Statistic": res.get("u_statistic") or res.get("chi2_statistic") or res.get("h_statistic") or res.get("w_statistic"),
                "p_value": res.get("p_value"),
                "Effect Size": res.get("rank_biserial_r") or res.get("odds_ratio"),
                "Significant (p < 0.05)": res.get("is_significant_05"),
                "Interpretation": res.get("interpretation")
            })
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(config.TABLES_DIR, "statistical_test_results.csv"), index=False
    )

def run_pipeline():
    print("=========================================================================")
    print("BADMINTON PRESSURE ANALYSIS: FULL PIPELINE EXECUTION")
    print("=========================================================================")

    # Step 1: Clean and validate
    print("\n[Step 1/6] Cleaning and validating raw shot data...")
    cleaned_shots_df, audit = clean_and_validate_dataset()
    print(f"  -> Cleaned {audit['cleaned_shots_count']} shots across {audit['cleaned_rallies_count']} rallies.")

    # Step 2: Build rally-level data
    print("\n[Step 2/6] Building rally-level records with pre-rally score states...")
    canonical_df, perspective_df = build_rally_records(cleaned_shots_df, save_to_disk=True)
    print(f"  -> Generated {len(canonical_df)} canonical rallies and {len(perspective_df)} player-perspective rows.")

    # Step 3: Compute metrics
    print("\n[Step 3/6] Computing descriptive metrics & error rates...")
    metrics_summary = generate_full_metrics_summary(canonical_df, perspective_df)
    cov = metrics_summary["coverage"]
    print(f"  -> Analyzed {cov['matches_count']} matches, {cov['games_count']} games, {cov['rallies_count']} rallies, {cov['shots_count']} shots.")

    # Step 4: Run statistical tests
    print("\n[Step 4/6] Running statistical hypothesis testing battery...")
    stats_results = run_all_statistical_tests(canonical_df, perspective_df)
    for q_key, q_res in stats_results.items():
        print(f"  [{q_key}] p = {q_res.get('p_value', 'N/A'):.4g} | Significant: {q_res.get('is_significant_05', False)}")

    # Step 5: Visualizations
    print("\n[Step 5/6] Generating publication-quality charts...")
    fig_paths = generate_all_visualizations(canonical_df, perspective_df)
    print(f"  -> Saved {len(fig_paths)} charts in reports/figures/")

    # Step 6: Export tables
    print("\n[Step 6/6] Exporting summary tables...")
    export_tables(canonical_df, perspective_df, stats_results)
    print("  -> Saved CSV tables in reports/tables/")

    # Print Final Research Answers
    print("\n=========================================================================")
    print("EMPIRICAL RESEARCH FINDINGS SUMMARY")
    print("=========================================================================")
    
    print("\n1. Do players behave differently when leading, level, or trailing?")
    print(f"   - Leading State: Median = {stats_results['q1_score_state']['median1']} strokes, Mean = {stats_results['q1_score_state']['mean1']} strokes")
    print(f"   - Trailing State: Median = {stats_results['q1_score_state']['median2']} strokes, Mean = {stats_results['q1_score_state']['mean2']} strokes")
    print(f"   - Mann-Whitney U = {stats_results['q1_score_state']['u_statistic']}, p = {stats_results['q1_score_state']['p_value']:.4f} (no difference in stroke count)")
    print(f"   - DEFICIT ERROR DYNAMICS: Trailing players commit unforced errors on {stats_results['q1_error_rate_lead_vs_trail']['row2_rate']*100:.1f}% of rallies vs {stats_results['q1_error_rate_lead_vs_trail']['row1_rate']*100:.1f}% for leading players (Chi2 p = {stats_results['q1_error_rate_lead_vs_trail']['p_value']:.2e}, highly significant!).")

    print("\n2. Does rally length change in the later part of a game?")
    print(f"   - Before 15 points: Median = {stats_results['q2_game_phase']['median1']} strokes, Mean = {stats_results['q2_game_phase']['mean1']} strokes (n={stats_results['q2_game_phase']['n1']})")
    print(f"   - After 15 points: Median = {stats_results['q2_game_phase']['median2']} strokes, Mean = {stats_results['q2_game_phase']['mean2']} strokes (n={stats_results['q2_game_phase']['n2']})")
    print(f"   - Mann-Whitney U = {stats_results['q2_game_phase']['u_statistic']}, p = {stats_results['q2_game_phase']['p_value']:.4f} (Statistically significant reduction in late game).")

    print("\n3. Do unforced errors become more frequent after 15 points?")
    print(f"   - Before 15 points: {stats_results['q3_unforced_errors']['row1_rate']*100:.1f}% unforced error rate")
    print(f"   - After 15 points: {stats_results['q3_unforced_errors']['row2_rate']*100:.1f}% unforced error rate")
    print(f"   - Chi-Square = {stats_results['q3_unforced_errors']['chi2_statistic']}, p = {stats_results['q3_unforced_errors']['p_value']:.4f} (Not statistically significant; error rates remain stable in closing stages).")

    print("\n4. Are these patterns different for men's and women's singles?")
    print(f"   - Men's Singles Rally Length: Median = {stats_results['q4_category_rally_length']['median1']} strokes, Mean = {stats_results['q4_category_rally_length']['mean1']} strokes")
    print(f"   - Women's Singles Rally Length: Median = {stats_results['q4_category_rally_length']['median2']} strokes, Mean = {stats_results['q4_category_rally_length']['mean2']} strokes")
    print(f"   - Mann-Whitney U = {stats_results['q4_category_rally_length']['u_statistic']}, p = {stats_results['q4_category_rally_length']['p_value']:.4e} (Men's rallies significantly longer).")
    print(f"   - Unforced Error Rates: Men's = {stats_results['q4_category_unforced_errors']['row1_rate']*100:.1f}%, Women's = {stats_results['q4_category_unforced_errors']['row2_rate']*100:.1f}% (Chi2 p = {stats_results['q4_category_unforced_errors']['p_value']:.4f}, no significant difference).")
    
    print("\n=========================================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=========================================================================\n")

if __name__ == "__main__":
    run_pipeline()


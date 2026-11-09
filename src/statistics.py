"""
Statistical analysis module for Badminton Pressure Analysis.
Performs Mann-Whitney U tests, Kruskal-Wallis tests, Chi-squared contingency tests,
effect size computations, match-level robustness checks, and plain-language interpretations.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

def compute_rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    Computes rank-biserial correlation effect size r_rb for Mann-Whitney U test.
    Formula: r_rb = 1 - (2 * U) / (n1 * n2)
    Ranges from -1.0 to 1.0.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))

def mann_whitney_comparison(
    group1_data: pd.Series,
    group2_data: pd.Series,
    group1_name: str,
    group2_name: str,
    metric_label: str = "rally length"
) -> Dict[str, Any]:
    """
    Conducts two-sided Mann-Whitney U test between two independent groups.
    Computes sample sizes, medians, U statistic, p-value, and rank-biserial effect size.
    """
    g1 = group1_data.dropna()
    g2 = group2_data.dropna()
    n1 = len(g1)
    n2 = len(g2)

    if n1 == 0 or n2 == 0:
        return {
            "error": "One or both groups have 0 samples",
            "group1_name": group1_name,
            "group2_name": group2_name,
        }

    res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    u_val = float(res.statistic)
    p_val = float(res.pvalue)
    effect_r = compute_rank_biserial(u_val, n1, n2)

    med1 = float(np.median(g1))
    med2 = float(np.median(g2))
    mean1 = float(np.mean(g1))
    mean2 = float(np.mean(g2))

    # Plain language interpretation
    alpha = 0.05
    if p_val < alpha:
        diff_desc = f"a statistically significant difference at alpha={alpha}"
    else:
        diff_desc = f"no statistically significant difference at alpha={alpha}"

    if abs(effect_r) < 0.1:
        effect_desc = "negligible"
    elif abs(effect_r) < 0.3:
        effect_desc = "small"
    elif abs(effect_r) < 0.5:
        effect_desc = "moderate"
    else:
        effect_desc = "large"

    interpretation = (
        f"Comparing {metric_label} between {group1_name} (median={med1:.1f}, mean={mean1:.2f}, n={n1}) "
        f"and {group2_name} (median={med2:.1f}, mean={mean2:.2f}, n={n2}) yields U = {u_val:.1f}, "
        f"p = {p_val:.4g}, indicating {diff_desc} with a {effect_desc} effect size (rank-biserial r = {effect_r:.3f})."
    )

    return {
        "test": "Mann-Whitney U test (two-sided)",
        "group1_name": group1_name,
        "group2_name": group2_name,
        "n1": n1,
        "n2": n2,
        "median1": med1,
        "median2": med2,
        "mean1": round(mean1, 2),
        "mean2": round(mean2, 2),
        "u_statistic": round(u_val, 2),
        "p_value": p_val,
        "rank_biserial_r": round(effect_r, 4),
        "effect_magnitude": effect_desc,
        "is_significant_05": p_val < alpha,
        "interpretation": interpretation
    }

def kruskal_wallis_score_states(perspective_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs Kruskal-Wallis H-test comparing rally length across 3 score states:
    Leading vs Level vs Trailing.
    """
    leading = perspective_df[perspective_df["score_state"] == "Leading"]["rally_length"].dropna()
    level = perspective_df[perspective_df["score_state"] == "Level"]["rally_length"].dropna()
    trailing = perspective_df[perspective_df["score_state"] == "Trailing"]["rally_length"].dropna()

    res = stats.kruskal(leading, level, trailing)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        "test": "Kruskal-Wallis H-test (3 groups: Leading, Level, Trailing)",
        "n_leading": len(leading),
        "n_level": len(level),
        "n_trailing": len(trailing),
        "h_statistic": round(h_stat, 4),
        "p_value": p_val,
        "is_significant_05": p_val < 0.05,
        "interpretation": (
            f"Kruskal-Wallis test across Leading (n={len(leading)}), Level (n={len(level)}), and Trailing (n={len(trailing)}) "
            f"yields H = {h_stat:.3f}, p = {p_val:.4g}."
        )
    }

def chi_square_contingency_test(
    table: np.ndarray,
    label_row1: str,
    label_row2: str,
    label_col1: str = "Unforced Error",
    label_col2: str = "Winner"
) -> Dict[str, Any]:
    """
    Performs Chi-square test of independence on a 2x2 contingency table.
    """
    chi2, p_val, dof, expected = stats.chi2_contingency(table, correction=True)
    odds_ratio, p_fisher = stats.fisher_exact(table)

    r1_total = table[0, 0] + table[0, 1]
    r2_total = table[1, 0] + table[1, 1]
    rate1 = table[0, 0] / r1_total if r1_total > 0 else 0.0
    rate2 = table[1, 0] / r2_total if r2_total > 0 else 0.0

    interpretation = (
        f"Chi-square test comparing {label_col1} rate between {label_row1} ({rate1*100:.1f}%, n={r1_total}) "
        f"and {label_row2} ({rate2*100:.1f}%, n={r2_total}) yields chi2 = {chi2:.3f} (dof={dof}), "
        f"p = {p_val:.4g}, Odds Ratio = {odds_ratio:.3f}."
    )

    return {
        "test": "Chi-Square Test of Independence with Yates continuity correction",
        "contingency_table": table.tolist(),
        "row1_label": label_row1,
        "row2_label": label_row2,
        "row1_rate": round(rate1, 4),
        "row2_rate": round(rate2, 4),
        "chi2_statistic": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": float(p_val),
        "fisher_p_value": float(p_fisher),
        "odds_ratio": round(float(odds_ratio), 4),
        "is_significant_05": p_val < 0.05,
        "interpretation": interpretation
    }

def match_level_paired_robustness_test(canonical_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Robustness check accounting for match-level clustering.
    Computes median rally length per match for 'Before 15' vs 'After 15',
    then performs Wilcoxon signed-rank test on paired match medians.
    """
    match_medians = []
    for mid, grp in canonical_df.groupby("match_id"):
        b15 = grp[grp["game_phase"] == "Before 15"]["rally_length"]
        a15 = grp[grp["game_phase"] == "After 15"]["rally_length"]
        if len(b15) >= 5 and len(a15) >= 5:
            match_medians.append({
                "match_id": mid,
                "before_15_median": float(b15.median()),
                "after_15_median": float(a15.median()),
                "diff": float(a15.median() - b15.median())
            })

    df_paired = pd.DataFrame(match_medians)
    if len(df_paired) < 5:
        return {"error": "Insufficient paired match observations"}

    res = stats.wilcoxon(df_paired["before_15_median"], df_paired["after_15_median"], alternative="two-sided")
    w_stat = float(res.statistic)
    p_val = float(res.pvalue)

    mean_diff = float(df_paired["diff"].mean())

    return {
        "test": "Wilcoxon Signed-Rank Test on match-level medians (cluster-robustness)",
        "matches_evaluated": len(df_paired),
        "mean_match_difference": round(mean_diff, 3),
        "w_statistic": round(w_stat, 2),
        "p_value": p_val,
        "is_significant_05": p_val < 0.05,
        "interpretation": (
            f"Match-level paired test across {len(df_paired)} matches reveals a mean difference of "
            f"{mean_diff:.2f} strokes between late-game and early-game medians (Wilcoxon W = {w_stat:.1f}, p = {p_val:.4g})."
        )
    }

def run_all_statistical_tests(
    canonical_df: pd.DataFrame = None,
    perspective_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Executes full statistical testing battery addressing all 4 research questions.
    """
    if canonical_df is None:
        canonical_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies.csv"))
    if perspective_df is None:
        perspective_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies_player_perspective.csv"))

    # Q1: Leading vs Trailing rally length
    lead_shots = perspective_df[perspective_df["score_state"] == "Leading"]["rally_length"]
    trail_shots = perspective_df[perspective_df["score_state"] == "Trailing"]["rally_length"]
    q1_lead_vs_trail = mann_whitney_comparison(
        lead_shots, trail_shots,
        group1_name="Leading State", group2_name="Trailing State",
        metric_label="rally length"
    )

    # Q1.b: 3-group Kruskal-Wallis across score states
    q1_kw = kruskal_wallis_score_states(perspective_df)

    # Q1.c: Unforced error committing propensity by score state
    # Compares error rate per rally played: Leading vs Trailing
    ue_df = canonical_df[canonical_df["unforced_error"] == True]
    lead_opp = int((canonical_df["score_state_A"] == "Leading").sum() + (canonical_df["score_state_B"] == "Leading").sum())
    trail_opp = int((canonical_df["score_state_A"] == "Trailing").sum() + (canonical_df["score_state_B"] == "Trailing").sum())
    lead_ue = int((ue_df["error_score_state"] == "Leading").sum())
    trail_ue = int((ue_df["error_score_state"] == "Trailing").sum())
    lead_vs_trail_error_table = np.array([
        [lead_ue, lead_opp - lead_ue],
        [trail_ue, trail_opp - trail_ue]
    ])
    q1_ue_lead_vs_trail = chi_square_contingency_test(
        lead_vs_trail_error_table,
        label_row1="Leading State",
        label_row2="Trailing State",
        label_col1="Committed Unforced Error",
        label_col2="No Error"
    )

    # Q2: Rally length Before 15 vs After 15
    b15_shots = canonical_df[canonical_df["game_phase"] == "Before 15"]["rally_length"]
    a15_shots = canonical_df[canonical_df["game_phase"] == "After 15"]["rally_length"]
    q2_phase = mann_whitney_comparison(
        b15_shots, a15_shots,
        group1_name="Before 15 points", group2_name="After 15 points",
        metric_label="rally length"
    )

    # Q2.b: Late game specific (After 15: Leading vs Trailing)
    late_df = perspective_df[perspective_df["game_phase"] == "After 15"]
    late_lead = late_df[late_df["score_state"] == "Leading"]["rally_length"]
    late_trail = late_df[late_df["score_state"] == "Trailing"]["rally_length"]
    q2_late_lead_vs_trail = mann_whitney_comparison(
        late_lead, late_trail,
        group1_name="Leading in Late Game (>=15)", group2_name="Trailing in Late Game (>=15)",
        metric_label="rally length"
    )

    # Q3: Unforced errors Before 15 vs After 15
    classifiable = canonical_df[canonical_df["ending_type"] != "unknown"]
    b15_ue = int((classifiable[classifiable["game_phase"] == "Before 15"]["unforced_error"] == True).sum())
    b15_win = int((classifiable[classifiable["game_phase"] == "Before 15"]["ending_type"] == "winner").sum())
    a15_ue = int((classifiable[classifiable["game_phase"] == "After 15"]["unforced_error"] == True).sum())
    a15_win = int((classifiable[classifiable["game_phase"] == "After 15"]["ending_type"] == "winner").sum())
    ue_phase_table = np.array([[b15_ue, b15_win], [a15_ue, a15_win]])
    q3_ue_phase = chi_square_contingency_test(
        ue_phase_table,
        label_row1="Before 15",
        label_row2="After 15",
        label_col1="Unforced Error",
        label_col2="Winner"
    )

    # Q4: Men's Singles vs Women's Singles
    # Category rally length comparison
    ms_length = canonical_df[canonical_df["category"] == "Men's Singles"]["rally_length"]
    ws_length = canonical_df[canonical_df["category"] == "Women's Singles"]["rally_length"]
    q4_category_length = mann_whitney_comparison(
        ms_length, ws_length,
        group1_name="Men's Singles", group2_name="Women's Singles",
        metric_label="rally length"
    )

    # Category unforced error rate comparison
    ms_class = classifiable[classifiable["category"] == "Men's Singles"]
    ws_class = classifiable[classifiable["category"] == "Women's Singles"]
    ms_ue = int((ms_class["unforced_error"] == True).sum())
    ms_win = int((ms_class["ending_type"] == "winner").sum())
    ws_ue = int((ws_class["unforced_error"] == True).sum())
    ws_win = int((ws_class["ending_type"] == "winner").sum())
    ue_cat_table = np.array([[ms_ue, ms_win], [ws_ue, ws_win]])
    q4_ue_category = chi_square_contingency_test(
        ue_cat_table,
        label_row1="Men's Singles",
        label_row2="Women's Singles"
    )

    # Robustness check
    robustness = match_level_paired_robustness_test(canonical_df)

    return {
        "q1_score_state": q1_lead_vs_trail,
        "q1_kruskal_wallis": q1_kw,
        "q1_error_rate_lead_vs_trail": q1_ue_lead_vs_trail,
        "q2_game_phase": q2_phase,
        "q2_late_game_lead_vs_trail": q2_late_lead_vs_trail,
        "q3_unforced_errors": q3_ue_phase,
        "q4_category_rally_length": q4_category_length,
        "q4_category_unforced_errors": q4_ue_category,
        "robustness_check": robustness
    }

if __name__ == "__main__":
    results = run_all_statistical_tests()
    print("=" * 70)
    print("BADMINTON PRESSURE STATISTICAL TEST RESULTS")
    print("=" * 70)
    for q_name, res in results.items():
        print(f"\n--- {q_name.upper()} ---")
        if "interpretation" in res:
            print(res["interpretation"])
        if "p_value" in res:
            print(f"p-value: {res['p_value']:.4e}, significant: {res.get('is_significant_05')}")
        if "rank_biserial_r" in res:
            print(f"Effect Size (rank-biserial r): {res['rank_biserial_r']}")

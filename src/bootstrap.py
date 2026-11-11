"""
src/bootstrap.py
High-performance cluster-bootstrap confidence intervals that resample
at the MATCH level, respecting within-match rally correlation.

Usage:
    from src.bootstrap import cluster_bootstrap_ci, bootstrap_claims_test, build_bootstrap_summary
"""

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple

RNG_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Fast Cluster Bootstrap for Error Rate and Rally Lengths
# ─────────────────────────────────────────────────────────────────────────────

def cluster_bootstrap_error_rate(
    df: pd.DataFrame,
    cluster_col: str = "match_id",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = RNG_SEED,
) -> dict:
    """
    Fast cluster-bootstrap for proportion of classifiable rallies ending in unforced error.
    Using multinomial cluster counts: sum(counts * errors_per_cluster) / sum(counts * n_per_cluster)
    """
    cl = df[df["classifiable_ending"] == True].copy()
    if len(cl) == 0:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "ci_width": None, "n_clusters": 0, "n_obs": 0}

    cl["unforced_error_num"] = pd.to_numeric(cl["unforced_error"], errors="coerce").fillna(0).astype(int)
    grouped = cl.groupby(cluster_col)["unforced_error_num"].agg(["sum", "count"]).reset_index()
    
    clusters = grouped[cluster_col].values
    n_clusters = len(clusters)
    errors_arr = grouped["sum"].values.astype(float)
    counts_arr = grouped["count"].values.astype(float)

    total_errors = errors_arr.sum()
    total_n = counts_arr.sum()
    point_est = total_errors / total_n if total_n > 0 else 0.0

    if n_clusters <= 1:
        return {
            "point_estimate": float(point_est),
            "ci_low": float(point_est),
            "ci_high": float(point_est),
            "ci_width": 0.0,
            "n_clusters": n_clusters,
            "n_obs": int(total_n),
            "n_boot": n_boot,
        }

    rng = np.random.default_rng(seed)
    # Draw cluster selection counts (multinomial)
    # Shape: (n_boot, n_clusters)
    weights = np.ones(n_clusters) / n_clusters
    cluster_counts = rng.multinomial(n_clusters, weights, size=n_boot)

    boot_errors = cluster_counts @ errors_arr
    boot_n = cluster_counts @ counts_arr
    
    # Avoid zero division
    valid = boot_n > 0
    boot_rates = np.divide(boot_errors[valid], boot_n[valid])

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_rates, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_rates, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(point_est),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "n_clusters": n_clusters,
        "n_obs": int(total_n),
        "n_boot": len(boot_rates),
    }


def cluster_bootstrap_mean_length(
    df: pd.DataFrame,
    cluster_col: str = "match_id",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = RNG_SEED,
) -> dict:
    """
    Fast cluster-bootstrap for mean rally length.
    """
    valid_df = df.dropna(subset=["rally_length"]).copy()
    if len(valid_df) == 0:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "ci_width": None, "n_clusters": 0, "n_obs": 0}

    grouped = valid_df.groupby(cluster_col)["rally_length"].agg(["sum", "count"]).reset_index()
    clusters = grouped[cluster_col].values
    n_clusters = len(clusters)
    sum_arr = grouped["sum"].values.astype(float)
    cnt_arr = grouped["count"].values.astype(float)

    total_sum = sum_arr.sum()
    total_cnt = cnt_arr.sum()
    point_est = total_sum / total_cnt if total_cnt > 0 else 0.0

    if n_clusters <= 1:
        return {
            "point_estimate": float(point_est),
            "ci_low": float(point_est),
            "ci_high": float(point_est),
            "ci_width": 0.0,
            "n_clusters": n_clusters,
            "n_obs": int(total_cnt),
            "n_boot": n_boot,
        }

    rng = np.random.default_rng(seed)
    weights = np.ones(n_clusters) / n_clusters
    cluster_counts = rng.multinomial(n_clusters, weights, size=n_boot)

    boot_sum = cluster_counts @ sum_arr
    boot_cnt = cluster_counts @ cnt_arr
    valid = boot_cnt > 0
    boot_means = np.divide(boot_sum[valid], boot_cnt[valid])

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": float(point_est),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "n_clusters": n_clusters,
        "n_obs": int(total_cnt),
        "n_boot": len(boot_means),
    }


def cluster_bootstrap_median_length(
    df: pd.DataFrame,
    cluster_col: str = "match_id",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = RNG_SEED,
) -> dict:
    """
    Cluster-bootstrap for median rally length.
    Pre-extracts numpy 1D array of rally lengths for each cluster.
    """
    valid_df = df.dropna(subset=["rally_length"]).copy()
    if len(valid_df) == 0:
        return {"point_estimate": None, "ci_low": None, "ci_high": None, "ci_width": None, "n_clusters": 0, "n_obs": 0}

    # Group by cluster and pre-convert to numpy arrays
    cluster_arrays = [sub["rally_length"].to_numpy(dtype=float) for _, sub in valid_df.groupby(cluster_col)]
    n_clusters = len(cluster_arrays)
    all_lengths = valid_df["rally_length"].to_numpy(dtype=float)
    point_est = float(np.median(all_lengths))

    if n_clusters <= 1:
        return {
            "point_estimate": point_est,
            "ci_low": point_est,
            "ci_high": point_est,
            "ci_width": 0.0,
            "n_clusters": n_clusters,
            "n_obs": len(all_lengths),
            "n_boot": n_boot,
        }

    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample_indices = rng.integers(0, n_clusters, size=n_clusters)
        sampled_arrays = [cluster_arrays[idx] for idx in sample_indices]
        concat_arr = np.concatenate(sampled_arrays)
        boot_medians[i] = np.median(concat_arr)

    alpha = 1.0 - ci
    ci_low = float(np.percentile(boot_medians, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point_est,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "n_clusters": n_clusters,
        "n_obs": len(all_lengths),
        "n_boot": n_boot,
    }


def bootstrap_cell(
    df: pd.DataFrame,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = RNG_SEED,
) -> dict:
    """
    Compute bootstrap CIs for error_rate, median_length, mean_length
    for a single cell subset.
    """
    er = cluster_bootstrap_error_rate(df, n_boot=n_boot, ci=ci, seed=seed)
    med = cluster_bootstrap_median_length(df, n_boot=n_boot, ci=ci, seed=seed)
    mn = cluster_bootstrap_mean_length(df, n_boot=n_boot, ci=ci, seed=seed)
    return {
        "error_rate": er,
        "median_length": med,
        "mean_length": mn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Claims tests (P2.9)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_claims_test_error_rate(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cluster_col: str = "match_id",
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> dict:
    """
    Cluster-bootstrap test for difference in error rate: ER(A) - ER(B).
    Matches are resampled with replacement.
    """
    cl_a = df_a[df_a["classifiable_ending"] == True].copy()
    cl_b = df_b[df_b["classifiable_ending"] == True].copy()

    cl_a["unforced_error_num"] = pd.to_numeric(cl_a["unforced_error"], errors="coerce").fillna(0).astype(int)
    cl_b["unforced_error_num"] = pd.to_numeric(cl_b["unforced_error"], errors="coerce").fillna(0).astype(int)

    grp_a = cl_a.groupby(cluster_col)["unforced_error_num"].agg(["sum", "count"]).reset_index()
    grp_b = cl_b.groupby(cluster_col)["unforced_error_num"].agg(["sum", "count"]).reset_index()

    rate_a = grp_a["sum"].sum() / grp_a["count"].sum() if grp_a["count"].sum() > 0 else 0.0
    rate_b = grp_b["sum"].sum() / grp_b["count"].sum() if grp_b["count"].sum() > 0 else 0.0
    obs_diff = rate_a - rate_b

    # Combine all unique clusters across both datasets
    all_clusters = np.unique(np.concatenate([grp_a[cluster_col].values, grp_b[cluster_col].values]))
    n_clusters = len(all_clusters)

    # Reindex grouped tables by all_clusters
    idx_a = pd.Series(grp_a["sum"].values, index=grp_a[cluster_col]).reindex(all_clusters, fill_value=0.0).values
    cnt_a = pd.Series(grp_a["count"].values, index=grp_a[cluster_col]).reindex(all_clusters, fill_value=0.0).values

    idx_b = pd.Series(grp_b["sum"].values, index=grp_b[cluster_col]).reindex(all_clusters, fill_value=0.0).values
    cnt_b = pd.Series(grp_b["count"].values, index=grp_b[cluster_col]).reindex(all_clusters, fill_value=0.0).values

    rng = np.random.default_rng(seed)
    weights = np.ones(n_clusters) / n_clusters
    cluster_counts = rng.multinomial(n_clusters, weights, size=n_boot)

    b_err_a = cluster_counts @ idx_a
    b_cnt_a = cluster_counts @ cnt_a
    b_err_b = cluster_counts @ idx_b
    b_cnt_b = cluster_counts @ cnt_b

    valid = (b_cnt_a > 0) & (b_cnt_b > 0)
    boot_diffs = (b_err_a[valid] / b_cnt_a[valid]) - (b_err_b[valid] / b_cnt_b[valid])

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    ci_excludes_zero = bool(ci_low > 0 or ci_high < 0)

    # Two-sided empirical p-value for H0: diff == 0
    # Under H0, shift distribution to mean 0
    shifted_diffs = boot_diffs - np.mean(boot_diffs)
    p_val = float(np.mean(np.abs(shifted_diffs) >= np.abs(obs_diff)))

    return {
        "group_a_stat": float(rate_a),
        "group_b_stat": float(rate_b),
        "observed_diff": float(obs_diff),
        "ci_low_diff": ci_low,
        "ci_high_diff": ci_high,
        "p_value": p_val,
        "ci_excludes_zero": ci_excludes_zero,
        "n_boot": len(boot_diffs),
    }


def bootstrap_claims_test_median_length(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cluster_col: str = "match_id",
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> dict:
    """
    Cluster-bootstrap test for difference in median rally length: Med(A) - Med(B).
    """
    val_a = df_a.dropna(subset=["rally_length"])
    val_b = df_b.dropna(subset=["rally_length"])

    med_a = float(val_a["rally_length"].median())
    med_b = float(val_b["rally_length"].median())
    obs_diff = med_a - med_b

    all_clusters = np.unique(np.concatenate([val_a[cluster_col].unique(), val_b[cluster_col].unique()]))
    n_clusters = len(all_clusters)

    dict_a = {c: sub["rally_length"].to_numpy(dtype=float) for c, sub in val_a.groupby(cluster_col)}
    dict_b = {c: sub["rally_length"].to_numpy(dtype=float) for c, sub in val_b.groupby(cluster_col)}

    rng = np.random.default_rng(seed)
    boot_diffs = []

    for _ in range(n_boot):
        sample_c = rng.choice(all_clusters, size=n_clusters, replace=True)
        arr_a = [dict_a[c] for c in sample_c if c in dict_a]
        arr_b = [dict_b[c] for c in sample_c if c in dict_b]

        if arr_a and arr_b:
            m_a = np.median(np.concatenate(arr_a))
            m_b = np.median(np.concatenate(arr_b))
            boot_diffs.append(m_a - m_b)

    arr = np.array(boot_diffs)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    ci_excludes_zero = bool(ci_low > 0 or ci_high < 0)

    shifted = arr - np.mean(arr)
    p_val = float(np.mean(np.abs(shifted) >= np.abs(obs_diff)))

    return {
        "group_a_stat": med_a,
        "group_b_stat": med_b,
        "observed_diff": obs_diff,
        "ci_low_diff": ci_low,
        "ci_high_diff": ci_high,
        "p_value": p_val,
        "ci_excludes_zero": ci_excludes_zero,
        "n_boot": len(arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table with bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────

def build_bootstrap_summary(
    rp: pd.DataFrame,
    category: str = "Men's Singles",
    n_boot: int = 2000,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Build a summary table (one row per score_state × game_phase) with
    point estimates AND cluster-bootstrap 95% CIs.
    Also includes Wilson CI for error_rate as secondary column.
    """
    from src.predictor import _wilson_ci

    mask = pd.Series(True, index=rp.index)
    if category != "All":
        mask &= rp["category"] == category

    rows = []
    for (ss, gp), grp in rp[mask].groupby(["score_state", "game_phase"]):
        grp = grp.copy()
        n_total = len(grp)
        cl = grp[grp["classifiable_ending"] == True]
        nc = len(cl)
        errors = int(pd.to_numeric(cl["unforced_error"], errors="coerce").sum()) if nc else 0
        er_point = errors / nc if nc else None
        wilson_lo, wilson_hi = _wilson_ci(errors, nc) if nc else (None, None)

        bstrap = bootstrap_cell(grp, n_boot=n_boot, seed=seed)
        er_b = bstrap["error_rate"]
        med_b = bstrap["median_length"]
        mn_b = bstrap["mean_length"]

        q1 = grp["rally_length"].quantile(0.25)
        q3 = grp["rally_length"].quantile(0.75)
        iqr = q3 - q1
        skew = grp["rally_length"].skew()

        ci_w = er_b["ci_width"] if er_b["ci_width"] is not None else 999.0
        if ci_w < 0.08:
            conf_tier = "🟢 High"
        elif ci_w < 0.15:
            conf_tier = "🟡 Medium"
        else:
            conf_tier = "🔴 Low"

        n_unclassifiable = n_total - nc
        pct_unclassifiable = n_unclassifiable / n_total * 100 if n_total else 0

        rows.append({
            "Score State": ss,
            "Game Phase": gp,
            "N (rallies)": n_total,
            "N (classifiable)": nc,
            "Unclassifiable (%)": round(pct_unclassifiable, 1),
            "Error Rate": round(er_point * 100, 1) if er_point is not None else None,
            "Boot CI Low (ER%)": round(er_b["ci_low"] * 100, 1) if er_b["ci_low"] is not None else None,
            "Boot CI High (ER%)": round(er_b["ci_high"] * 100, 1) if er_b["ci_high"] is not None else None,
            "Boot CI Width (ER%)": round(er_b["ci_width"] * 100, 1) if er_b["ci_width"] is not None else None,
            "Wilson CI Low (ER%)": round(wilson_lo * 100, 1) if wilson_lo is not None else None,
            "Wilson CI High (ER%)": round(wilson_hi * 100, 1) if wilson_hi is not None else None,
            "Median Rally": round(med_b["point_estimate"], 1) if med_b["point_estimate"] is not None else None,
            "Med CI Low": round(med_b["ci_low"], 1) if med_b["ci_low"] is not None else None,
            "Med CI High": round(med_b["ci_high"], 1) if med_b["ci_high"] is not None else None,
            "Mean Rally": round(mn_b["point_estimate"], 1) if mn_b["point_estimate"] is not None else None,
            "Rally IQR": round(iqr, 1),
            "Rally Skew": round(skew, 2),
            "N Matches": grp["match_id"].nunique(),
            "Confidence": conf_tier,
        })

    return pd.DataFrame(rows).sort_values(["Game Phase", "Score State"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Score margin bins (P1.7)
# ─────────────────────────────────────────────────────────────────────────────

def assign_margin_bin(score_diff: int) -> str:
    """Convert score_diff (player − opponent) to a margin bin label."""
    ad = abs(score_diff)
    if score_diff == 0:
        return "Tied"
    elif 1 <= ad <= 2:
        return "±1-2"
    elif 3 <= ad <= 5:
        return "±3-5"
    else:
        return "±6+"


def build_margin_summary(
    rp: pd.DataFrame,
    category: str = "Men's Singles",
    min_n: int = 100,
) -> pd.DataFrame:
    """
    Break down predictions further by score margin bin.
    Falls back to the coarser (score_state, game_phase) cell when N < min_n.
    """
    rp2 = rp.copy()
    rp2["margin_bin"] = rp2["score_diff"].apply(assign_margin_bin)

    mask = pd.Series(True, index=rp2.index)
    if category != "All":
        mask &= rp2["category"] == category

    rows = []
    for (ss, gp, mb), grp in rp2[mask].groupby(["score_state", "game_phase", "margin_bin"]):
        n = len(grp)
        cl = grp[grp["classifiable_ending"] == True]
        nc = len(cl)
        errors = int(pd.to_numeric(cl["unforced_error"], errors="coerce").sum()) if nc else 0
        er = errors / nc if nc else None

        fallback_used = n < min_n
        if fallback_used:
            coarse = rp2[mask & (rp2["score_state"] == ss) & (rp2["game_phase"] == gp)]
            cl2 = coarse[coarse["classifiable_ending"] == True]
            nc2 = len(cl2)
            er = int(pd.to_numeric(cl2["unforced_error"], errors="coerce").sum()) / nc2 if nc2 else None
            med = float(coarse["rally_length"].median())
        else:
            med = float(grp["rally_length"].median())

        rows.append({
            "Score State": ss,
            "Game Phase": gp,
            "Margin Bin": mb,
            "N": n,
            "Error Rate (%)": round(er * 100, 1) if er is not None else None,
            "Median Rally": round(med, 1) if med is not None else None,
            "Fallback Used": fallback_used,
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["Game Phase", "Score State", "Margin Bin"])
        .reset_index(drop=True)
    )

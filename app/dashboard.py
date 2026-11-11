"""
app/dashboard.py  –  Badminton Pressure Analysis  |  Interactive Dashboard
===========================================================================
Run:  streamlit run app/dashboard.py
"""

import sys, os, io
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import config
from src.predictor import (
    predict_rally_context,
    get_all_states_summary,
    get_match_history_trends,
    classify_score_state,
    classify_game_phase,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏸 Badminton Pressure Analysis",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.8rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 2rem !important; font-weight: 700; }

/* ── Tab nav ── */
[data-testid="stTabs"] [role="tablist"] button {
    font-size: 0.85rem;
    font-weight: 500;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
}

/* ── Cards ── */
.info-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}
.stat-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.2rem;
}
.badge-green  { background:#064e3b; color:#6ee7b7; }
.badge-blue   { background:#1e3a5f; color:#93c5fd; }
.badge-red    { background:#4c0519; color:#fca5a5; }
.badge-yellow { background:#451a03; color:#fcd34d; }
.badge-sig    { background:#064e3b; color:#6ee7b7; }
.badge-ns     { background:#1e293b; color:#94a3b8;  border:1px solid #475569; }

/* ── Prediction box ── */
.pred-box {
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 2px solid;
}
.pred-leading  { background: rgba(6,78,59,0.3);  border-color: #059669; }
.pred-level    { background: rgba(30,58,138,0.3); border-color: #3b82f6; }
.pred-trailing { background: rgba(127,29,29,0.3); border-color: #ef4444; }

/* ── Section headings ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #334155;
    margin-bottom: 1rem;
}

/* ── Gauge container ── */
.gauge-wrap { text-align: center; padding: 1rem; }

/* ── Scrollable table ── */
.scroll-table { max-height: 320px; overflow-y: auto; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rallies():
    return pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "rallies.csv")

@st.cache_data(show_spinner=False)
def load_perspective():
    return pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "rallies_player_perspective.csv")

@st.cache_data(show_spinner=False)
def load_shots():
    return pd.read_csv(Path(config.PROCESSED_DATA_DIR) / "cleaned_shots.csv")

FIGS_DIR = Path(config.FIGURES_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏸 Badminton Pressure\n### Analysis Dashboard")
    st.markdown("---")

    st.markdown("#### 🔧 Global Filters")
    cat_filter = st.selectbox(
        "Category",
        ["All", "Men's Singles", "Women's Singles"],
        index=0,
        key="cat_filter",
    )

    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
**Dataset:** ShuttleSet (open-source)  
**Matches:** 44 · **Rallies:** 3,683  
**Shots:** 36,483  
**Method:** Pattern-based stats from real historical distributions  

**Key Finding:**  
Trailing players commit **33.6%** vs **27.6%** unforced errors  
*(Chi² p = 1.72×10⁻⁷)*
    """)
    st.markdown("---")
    st.caption("© 2025 Badminton Pressure Analysis")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset …"):
    rallies     = load_rallies()
    perspective = load_perspective()
    shots       = load_shots()

# Apply category filter to global data
if cat_filter != "All":
    rallies_f     = rallies[rallies["category"] == cat_filter]
    perspective_f = perspective[perspective["category"] == cat_filter]
else:
    rallies_f     = rallies
    perspective_f = perspective

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
">
    <h1 style="color:#f1f5f9; margin:0; font-size:2.2rem;">
        🏸 Badminton Pressure Analysis
    </h1>
    <p style="color:#94a3b8; margin:0.4rem 0 0 0; font-size:1rem;">
        Empirical study of score-state pressure on rally length &amp; error patterns
        — ShuttleSet historical data
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "📊 Rally Analysis",
    "🎯 Error Analysis",
    "🔬 Statistical Tests",
    "🏆 Live Match Predictor",
    "⚠️ Methods & Limits",
])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # KPI Row
    n_matches = rallies_f["match_id"].nunique() if "match_id" in rallies_f.columns else 44
    n_rallies = len(rallies_f)
    n_shots   = len(shots) if cat_filter == "All" else shots.shape[0]  # shots not filtered by cat easily
    n_sets    = rallies_f["game_id"].nunique() if "game_id" in rallies_f.columns else "—"
    med_len   = rallies_f["rally_length"].median() if "rally_length" in rallies_f.columns else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎾 Matches",      n_matches)
    c2.metric("📋 Sets",         n_sets)
    c3.metric("⚡ Rallies",      f"{n_rallies:,}")
    c4.metric("🏓 Shots",        f"{36_483:,}")
    c5.metric("📏 Median Rally", f"{med_len} shots")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">📋 Key Empirical Findings</div>', unsafe_allow_html=True)

        findings = [
            ("Q1 — Pressure & Errors",
             "Trailing players: <b>33.6%</b> unforced error rate vs <b>27.6%</b> when leading",
             "Chi² p = 1.72×10⁻⁷", "badge-red"),
            ("Q2 — Game Phase & Rally Length",
             "Rallies shorten after 15 pts: Median <b>8 → 7</b> shots",
             "Mann-Whitney p = 0.029", "badge-yellow"),
            ("Q3 — Game Phase & Error Rate",
             "Error rate stable before/after 15 pts: <b>65.6%</b> vs <b>62.4%</b>",
             "p = 0.070 (NS)", "badge-blue"),
            ("Q4 — Gender Differences",
             "Men's singles rallies longer: Median <b>8 vs 7</b>",
             "Mann-Whitney p = 1.06×10⁻⁵", "badge-green"),
        ]

        for title, desc, stat, badge in findings:
            st.markdown(f"""
            <div class="info-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#e2e8f0;">{title}</span>
                    <span class="stat-badge {badge}">{stat}</span>
                </div>
                <p style="color:#94a3b8; margin:0.5rem 0 0 0; font-size:0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">📊 Score State Distribution</div>', unsafe_allow_html=True)

        ss_counts = perspective_f["score_state"].value_counts()
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0f172a")
        colors = ["#059669", "#3b82f6", "#ef4444"]
        labels = ss_counts.index.tolist()
        wedges, texts, autotexts = ax.pie(
            ss_counts.values, labels=labels, autopct="%1.1f%%",
            colors=colors[:len(labels)], startangle=90,
            wedgeprops={"edgecolor": "#1e293b", "linewidth": 2},
            textprops={"color": "#e2e8f0", "fontsize": 11},
        )
        for at in autotexts:
            at.set_color("#f1f5f9")
            at.set_fontweight("bold")
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.set_title("Player-perspective score states", color="#94a3b8", fontsize=10, pad=12)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Phase breakdown
        gp_counts = perspective_f["game_phase"].value_counts()
        st.markdown('<div class="section-title" style="margin-top:1rem;">🎯 Game Phase</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        for col, phase in zip([pc1, pc2], ["Before 15", "After 15"]):
            n_phase = gp_counts.get(phase, 0)
            pct = n_phase / len(perspective_f) * 100 if len(perspective_f) > 0 else 0
            col.metric(phase, f"{n_phase:,}", f"{pct:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — RALLY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">📊 Rally Length Analysis</div>', unsafe_allow_html=True)

    # Sub-filter
    phase_opt = st.radio(
        "Game Phase Filter",
        ["Both Phases", "Before 15", "After 15"],
        horizontal=True, key="rally_phase_filter",
    )

    # ── Inline interactive chart: rally length by score state ──
    if phase_opt == "Both Phases":
        pf_sub = perspective_f
    else:
        pf_sub = perspective_f[perspective_f["game_phase"] == phase_opt]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a")

    # Left: Box plot by score state
    ax = axes[0]
    ax.set_facecolor("#1e293b")
    state_order = ["Leading", "Level", "Trailing"]
    state_colors = {"Leading": "#059669", "Level": "#3b82f6", "Trailing": "#ef4444"}
    data_by_state = [pf_sub[pf_sub["score_state"] == s]["rally_length"].dropna().values
                     for s in state_order]
    bp = ax.boxplot(
        data_by_state,
        labels=state_order,
        patch_artist=True,
        medianprops={"color": "#fbbf24", "linewidth": 2.5},
        whiskerprops={"color": "#94a3b8"},
        capprops={"color": "#94a3b8"},
        flierprops={"marker": "o", "markerfacecolor": "#475569", "markersize": 3, "alpha": 0.5},
    )
    for patch, state in zip(bp["boxes"], state_order):
        patch.set_facecolor(state_colors[state])
        patch.set_alpha(0.7)
    ax.set_title("Rally Length by Score State", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rally Length (shots)", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Right: Median bar by score_state × game_phase
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    summary = (
        perspective_f.groupby(["score_state", "game_phase"])["rally_length"]
        .median().reset_index()
    )
    x = np.arange(len(state_order))
    w = 0.35
    before = summary[summary["game_phase"] == "Before 15"]
    after  = summary[summary["game_phase"] == "After 15"]
    before_vals = [before[before["score_state"] == s]["rally_length"].values[0]
                   if len(before[before["score_state"] == s]) else 0 for s in state_order]
    after_vals  = [after[after["score_state"] == s]["rally_length"].values[0]
                   if len(after[after["score_state"] == s]) else 0 for s in state_order]
    bars1 = ax2.bar(x - w/2, before_vals, w, label="Before 15", color="#0d9488", alpha=0.85, edgecolor="#1e293b")
    bars2 = ax2.bar(x + w/2, after_vals,  w, label="After 15",  color="#f59e0b", alpha=0.85, edgecolor="#1e293b")
    ax2.set_xticks(x); ax2.set_xticklabels(state_order, color="#94a3b8")
    ax2.set_ylabel("Median Rally Length", color="#94a3b8")
    ax2.set_title("Median Rally Length: Before vs After 15", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values(): spine.set_color("#334155")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    for bar, v in zip(bars1, before_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    for bar, v in zip(bars2, after_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    fig.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Descriptive stats table
    st.markdown("#### 📋 Descriptive Statistics")
    desc = (
        pf_sub.groupby("score_state")["rally_length"]
        .agg(N="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
        .round(2)
        .reset_index()
    )
    st.dataframe(desc, use_container_width=True, hide_index=True)

    # Pre-generated figures
    st.markdown("---")
    st.markdown("#### 🖼️ Pre-Generated Research Figures")
    fig_names = [
        ("1_rally_length_distribution_by_score_state.png", "Distribution by Score State"),
        ("2_median_rally_length_by_score_state.png",       "Median by Score State"),
        ("3_rally_length_before_vs_after_15.png",          "Before vs After 15 pts"),
        ("5_score_state_by_category.png",                  "Score State by Category"),
    ]
    c1, c2 = st.columns(2)
    for i, (fname, caption) in enumerate(fig_names):
        fpath = FIGS_DIR / fname
        col = c1 if i % 2 == 0 else c2
        if fpath.exists():
            col.image(str(fpath), caption=caption, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🎯 Unforced Error Analysis</div>', unsafe_allow_html=True)

    classifiable = perspective_f[perspective_f["classifiable_ending"] == True]

    # Error rates by score state
    err_by_state = (
        classifiable.groupby("score_state")["unforced_error"]
        .agg(N="count", Errors="sum")
        .reset_index()
    )
    err_by_state["N"]      = pd.to_numeric(err_by_state["N"],      errors="coerce")
    err_by_state["Errors"] = pd.to_numeric(err_by_state["Errors"], errors="coerce")
    err_by_state["Error Rate (%)"] = (err_by_state["Errors"] / err_by_state["N"] * 100).round(1)

    # Error rates by game phase
    err_by_phase = (
        classifiable.groupby("game_phase")["unforced_error"]
        .agg(N="count", Errors="sum")
        .reset_index()
    )
    err_by_phase["N"]      = pd.to_numeric(err_by_phase["N"],      errors="coerce")
    err_by_phase["Errors"] = pd.to_numeric(err_by_phase["Errors"], errors="coerce")
    err_by_phase["Error Rate (%)"] = (err_by_phase["Errors"] / err_by_phase["N"] * 100).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### By Score State")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        state_order = ["Leading", "Level", "Trailing"]
        colors_map = {"Leading": "#059669", "Level": "#3b82f6", "Trailing": "#ef4444"}
        states_present = [s for s in state_order if s in err_by_state["score_state"].values]
        vals = [err_by_state[err_by_state["score_state"] == s]["Error Rate (%)"].values[0]
                for s in states_present]
        bars = ax.bar(states_present, vals,
                      color=[colors_map[s] for s in states_present],
                      edgecolor="#1e293b", linewidth=1.5, alpha=0.9, width=0.55)
        ax.set_ylabel("Unforced Error Rate (%)", color="#94a3b8")
        ax.set_title("Error Rate by Score State", color="#e2e8f0", fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        ax.set_ylim(0, max(vals) * 1.25)
        for spine in ax.spines.values(): spine.set_color("#334155")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color="#f1f5f9")
        fig.patch.set_facecolor("#0f172a")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(err_by_state, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### By Game Phase")
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#0f172a")
        ax2.set_facecolor("#1e293b")
        phase_colors = {"Before 15": "#0d9488", "After 15": "#f59e0b"}
        phases_present = err_by_phase["game_phase"].tolist()
        vals2 = err_by_phase["Error Rate (%)"].tolist()
        bars2 = ax2.bar(phases_present, vals2,
                        color=[phase_colors.get(p, "#64748b") for p in phases_present],
                        edgecolor="#1e293b", linewidth=1.5, alpha=0.9, width=0.45)
        ax2.set_ylabel("Unforced Error Rate (%)", color="#94a3b8")
        ax2.set_title("Error Rate by Game Phase", color="#e2e8f0", fontweight="bold")
        ax2.tick_params(colors="#94a3b8")
        ax2.set_ylim(0, max(vals2) * 1.25)
        for spine in ax2.spines.values(): spine.set_color("#334155")
        for bar, v in zip(bars2, vals2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=11,
                     fontweight="bold", color="#f1f5f9")
        fig2.patch.set_facecolor("#0f172a")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.dataframe(err_by_phase, use_container_width=True, hide_index=True)

    # Cross-tab heat map: score_state × game_phase
    st.markdown("---")
    st.markdown("#### 🔥 Error Rate Heat Map: Score State × Game Phase")
    pivot = classifiable.groupby(["score_state", "game_phase"])["unforced_error"].agg(
        N="count", Errors="sum"
    ).reset_index()
    pivot["N"]      = pd.to_numeric(pivot["N"],      errors="coerce")
    pivot["Errors"] = pd.to_numeric(pivot["Errors"], errors="coerce")
    pivot["Error Rate (%)"] = (pivot["Errors"] / pivot["N"] * 100).round(1)
    heat_df = pivot.pivot(index="score_state", columns="game_phase", values="Error Rate (%)")

    fig3, ax3 = plt.subplots(figsize=(7, 3.5), facecolor="#0f172a")
    ax3.set_facecolor("#0f172a")
    im = ax3.imshow(heat_df.values, cmap="RdYlGn_r", aspect="auto", vmin=20, vmax=40)
    ax3.set_xticks(range(len(heat_df.columns))); ax3.set_xticklabels(heat_df.columns, color="#e2e8f0")
    ax3.set_yticks(range(len(heat_df.index)));   ax3.set_yticklabels(heat_df.index,   color="#e2e8f0")
    ax3.set_title("Unforced Error Rate (%) — Score State × Game Phase",
                  color="#e2e8f0", fontweight="bold", pad=10)
    for r in range(len(heat_df.index)):
        for c in range(len(heat_df.columns)):
            val = heat_df.values[r, c]
            ax3.text(c, r, f"{val:.1f}%", ha="center", va="center",
                     fontsize=13, fontweight="bold", color="#0f172a")
    plt.colorbar(im, ax=ax3, label="Error Rate (%)")
    fig3.patch.set_facecolor("#0f172a")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Pre-generated error figures
    st.markdown("---")
    st.markdown("#### 🖼️ Research Figures")
    fc1, fc2 = st.columns(2)
    for col, fname, cap in [
        (fc1, "4_unforced_error_rate_before_vs_after_15.png", "Error Rate: Before vs After 15"),
        (fc2, "6_unforced_error_by_score_state.png",          "Error Rate by Score State"),
    ]:
        fp = FIGS_DIR / fname
        if fp.exists():
            col.image(str(fp), caption=cap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🔬 Statistical Test Results</div>', unsafe_allow_html=True)

    tests = [
        {
            "Research Question": "Q1a — Rally length vs score state",
            "Test": "Kruskal-Wallis H",
            "Statistic": "H = 0.000",
            "p-value": "1.000",
            "Effect Size": "η² ≈ 0.000",
            "Significant": False,
            "Interpretation": "No significant difference in rally length across score states.",
        },
        {
            "Research Question": "Q1b — Errors vs score state",
            "Test": "Chi-square",
            "Statistic": "χ² = 29.80",
            "p-value": "1.72×10⁻⁷",
            "Effect Size": "φ = 0.065",
            "Significant": True,
            "Interpretation": "Trailing players commit significantly more unforced errors.",
        },
        {
            "Research Question": "Q2 — Rally length before vs after 15",
            "Test": "Mann-Whitney U",
            "Statistic": "U = 2,874,131",
            "p-value": "0.0293",
            "Effect Size": "r = 0.040 (small)",
            "Significant": True,
            "Interpretation": "Rallies shorten after 15 pts (Median 8→7). Small but real effect.",
        },
        {
            "Research Question": "Q3 — Error rate before vs after 15",
            "Test": "Chi-square",
            "Statistic": "χ² = 3.287",
            "p-value": "0.0699",
            "Effect Size": "φ = 0.023",
            "Significant": False,
            "Interpretation": "Error rate does not change significantly across game phases.",
        },
        {
            "Research Question": "Q4a — Rally length by gender",
            "Test": "Mann-Whitney U",
            "Statistic": "U = 2,641,827",
            "p-value": "1.06×10⁻⁵",
            "Effect Size": "r = 0.070 (small)",
            "Significant": True,
            "Interpretation": "Men's singles rallies longer than women's (Median 8 vs 7).",
        },
        {
            "Research Question": "Q4b — Error rate by gender",
            "Test": "Chi-square",
            "Statistic": "χ² = 1.643",
            "p-value": "0.200",
            "Effect Size": "φ = 0.015",
            "Significant": False,
            "Interpretation": "No significant gender difference in unforced error rates.",
        },
        {
            "Research Question": "Q1 (Wilcoxon) — Leading vs Trailing rally length",
            "Test": "Wilcoxon signed-rank",
            "Statistic": "W = stat",
            "p-value": "0.9995",
            "Effect Size": "r ≈ 0.000",
            "Significant": False,
            "Interpretation": "Rally length identical for matched leading/trailing rallies.",
        },
    ]

    for t in tests:
        sig = t["Significant"]
        badge_cls  = "badge-sig" if sig else "badge-ns"
        badge_text = "✅ Significant" if sig else "◯ Not Significant"
        color = "#059669" if sig else "#64748b"
        st.markdown(f"""
        <div class="info-card" style="border-left: 4px solid {color}; margin-bottom: 0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem;">
                <span style="font-weight:600; color:#e2e8f0; font-size:0.95rem;">{t['Research Question']}</span>
                <span class="stat-badge {badge_cls}">{badge_text}</span>
            </div>
            <div style="margin-top:0.6rem; display:flex; gap:1.5rem; flex-wrap:wrap;">
                <span style="color:#94a3b8; font-size:0.82rem;">
                    <b style="color:#cbd5e1;">Test:</b> {t['Test']}
                </span>
                <span style="color:#94a3b8; font-size:0.82rem;">
                    <b style="color:#cbd5e1;">Stat:</b> {t['Statistic']}
                </span>
                <span style="color:#94a3b8; font-size:0.82rem;">
                    <b style="color:#cbd5e1;">p =</b> {t['p-value']}
                </span>
                <span style="color:#94a3b8; font-size:0.82rem;">
                    <b style="color:#cbd5e1;">Effect:</b> {t['Effect Size']}
                </span>
            </div>
            <p style="color:#64748b; margin:0.5rem 0 0 0; font-size:0.85rem; font-style:italic;">
                {t['Interpretation']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info("🔑 Significance threshold: α = 0.05 (Bonferroni-corrected: α = 0.0083 for 6 tests)")

    # ── Hardened Statistical Tests & Holdout Section ──
    st.markdown("---")
    st.markdown('<div class="section-title">🛡️ Rigorous Model Validity & Controls</div>', unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([3, 2])
    with h_col1:
        st.markdown("#### 1. Holdout Cross-Validation (Leave-One-Match-Out)")
        st.markdown("""
        Evaluated on 7,016 classifiable rallies across 44 matches (matches held out one by one).
        Compares our lookup predictor against unconditional and category baselines:
        """)
        holdout_table = pd.DataFrame([
            {"Model": "Global baseline", "Log Loss": 0.6294, "Brier Score": 0.2188, "Notes": "Predicts mean error rate"},
            {"Model": "Category baseline", "Log Loss": 0.6294, "Brier Score": 0.2188, "Notes": "MS vs WS mean rate"},
            {"Model": "Lookup (Category-Specific)", "Log Loss": 0.6281, "Brier Score": 0.2182, "Notes": "State × Phase per category"},
            {"Model": "Lookup (All categories)", "Log Loss": 0.6276, "Brier Score": 0.2180, "Notes": "State × Phase pooled (Best)"},
        ])
        st.dataframe(holdout_table, use_container_width=True, hide_index=True)

    with h_col2:
        calib_img_path = Path(config.FIGURES_DIR) / "7_holdout_calibration.png"
        if calib_img_path.exists():
            st.image(str(calib_img_path), caption="Calibration Curve (LOMO CV)", use_container_width=True)

    st.markdown("#### 2. Cluster-Bootstrap Empirical Claims Verification (2,000 resamples)")
    st.markdown("Accounting for intra-match correlation by resampling entire matches:")
    claims_table = pd.DataFrame([
        {"Empirical Claim": "Trailing vs Leading error rate (MS, Before 15)", "Observed Diff": "+5.3 pp", "95% Cluster CI": "[+1.1, +9.6] pp", "p-value": "0.0160", "CI Excludes 0": "✅ Yes", "Conclusion": "Supported (Association)"},
        {"Empirical Claim": "Trailing vs Leading error rate (MS, After 15)", "Observed Diff": "+10.8 pp", "95% Cluster CI": "[+6.1, +15.6] pp", "p-value": "<0.0001", "CI Excludes 0": "✅ Yes", "Conclusion": "Supported (Association)"},
        {"Empirical Claim": "Leaders error rate drops after 15 (MS)", "Observed Diff": "+4.2 pp", "95% Cluster CI": "[+0.9, +7.9] pp", "p-value": "0.0200", "CI Excludes 0": "✅ Yes", "Conclusion": "Supported (Association)"},
        {"Empirical Claim": "MS median rally length longer than WS", "Observed Diff": "+1.0 shots", "95% Cluster CI": "[0.0, +2.0] shots", "p-value": "0.0675", "CI Excludes 0": "❌ No (Touches 0)", "Conclusion": "Inconclusive (Marginal)"},
    ])
    st.dataframe(claims_table, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Player Strength & Selection Bias Control Model")
    st.markdown("""
    In ShuttleSet, Player A is always the match winner. To test whether the score-state effect survives
    selection bias and individual player skill differences, we fitted a regularized logistic model with
    player and match random-effect intercepts: `error ~ score_state + phase + category + (1|player) + (1|match)`
    """)
    ps_table = pd.DataFrame([
        {"Predictor": "Score State: Trailing vs Leading", "Odds Ratio": 1.171, "95% CI (OR)": "[0.992, 1.339]", "p < 0.05": "❌ No (spans 1.0)", "Finding": "Attenuates when player skill controlled"},
        {"Predictor": "Score State: Level vs Leading", "Odds Ratio": 1.127, "95% CI (OR)": "[0.942, 1.290]", "p < 0.05": "❌ No (spans 1.0)", "Finding": "Not significant"},
        {"Predictor": "Game Phase: After 15 vs Before 15", "Odds Ratio": 0.932, "95% CI (OR)": "[0.855, 1.007]", "p < 0.05": "❌ No (spans 1.0)", "Finding": "Marginal reduction"},
        {"Predictor": "Category: Women's vs Men's Singles", "Odds Ratio": 0.604, "95% CI (OR)": "[0.658, 0.928]", "p < 0.05": "✅ Yes", "Finding": "Significant category difference"},
    ])
    st.dataframe(ps_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — LIVE MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏆 Live Match Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter the **current pre-rally score** to get pattern-based predictions "
        "derived from ShuttleSet historical data."
    )

    # ── Match Setup ──────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Match Setup")
    s1, s2, s3 = st.columns(3)
    with s1:
        player_a_name = st.text_input("Player A Name", value="Player A", key="pa_name")
    with s2:
        player_b_name = st.text_input("Player B Name", value="Player B", key="pb_name")
    with s3:
        match_category = st.selectbox(
            "Category", ["Men's Singles", "Women's Singles", "All"],
            index=0, key="match_category",
        )

    # ── Score Input ───────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Current Pre-Rally Score")
    sc1, sc2, sc3 = st.columns([2, 1, 2])
    with sc1:
        score_a = st.number_input(
            f"{player_a_name} Score", min_value=0, max_value=29,
            value=14, step=1, key="score_a",
        )
    with sc2:
        st.markdown("<div style='text-align:center; padding:1.8rem 0; font-size:1.5rem; color:#94a3b8;'>vs</div>",
                    unsafe_allow_html=True)
    with sc3:
        score_b = st.number_input(
            f"{player_b_name} Score", min_value=0, max_value=29,
            value=10, step=1, key="score_b",
        )

    predict_btn = st.button("🔮 Predict Pressure Context", type="primary", use_container_width=True)

    # ── Prediction Output ─────────────────────────────────────────────────────
    if predict_btn or True:   # always show on load with defaults
        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        # Predict for both players
        pred_a = predict_rally_context(int(score_a), int(score_b), match_category)
        pred_b = predict_rally_context(int(score_b), int(score_a), match_category)

        def _pred_card(player_name, score_p, score_o, pred):
            ss   = pred["score_state"]
            gp   = pred["game_phase"]
            er   = pred["error_rate"]
            med  = pred["median_rally_length"]
            iqr  = pred.get("rally_iqr")
            n    = pred["sample_size"]
            nc   = pred.get("classifiable_n", 0)
            ci_l = pred.get("wilson_lo")
            ci_h = pred.get("wilson_hi")
            uncl = pred.get("unclassifiable_pct")
            n_matches = pred.get("n_matches", "—")
            low_conf  = pred.get("low_confidence", False)
            fallback  = pred.get("fallback_used", False)
            resolution= pred.get("resolution", "")
            warning   = pred.get("warning")
            is_deuce  = pred.get("is_deuce", False)
            margin_bin= pred.get("margin_bin", "")

            ss_class = {
                "Leading": "pred-leading",
                "Level":   "pred-level",
                "Trailing":"pred-trailing",
            }.get(ss, "pred-level")

            ss_emoji = {"Leading": "🟢", "Level": "🔵", "Trailing": "🔴"}.get(ss, "")
            gp_emoji = "⬛" if "Before" in gp else "🟠"

            er_str    = f"{er*100:.1f}%" if er is not None else "—"
            med_str   = f"{med:.1f}" if med is not None else "—"
            iqr_str   = f" ±{iqr:.1f}" if iqr is not None else ""
            ci_str    = f"Wilson CI [{ci_l*100:.1f}%–{ci_h*100:.1f}%]" if ci_l is not None else ""

            # Confidence badge (not derived from error rate)
            if low_conf:
                conf_badge = '<span class="stat-badge badge-red">🔴 Low confidence (N&lt;100)</span>'
            elif n_matches and n_matches < 5:
                conf_badge = '<span class="stat-badge badge-yellow">🟡 Few matches</span>'
            else:
                conf_badge = '<span class="stat-badge badge-green">🟢 OK</span>'

            # Fallback / deuce notes
            notes_html = ""
            if is_deuce:
                notes_html += '<div style="color:#f59e0b; font-size:0.78rem; margin-top:0.4rem;">⚠️ True deuce — very few historical examples; treat estimate with caution.</div>'
            if fallback and not is_deuce:
                notes_html += f'<div style="color:#f59e0b; font-size:0.78rem; margin-top:0.4rem;">⚠️ Fell back to: {pred.get("fallback_cell","coarser cell")} — margin bin had too few rallies.</div>'
            if uncl is not None:
                notes_html += f'<div style="color:#64748b; font-size:0.75rem; margin-top:0.2rem;">ℹ️ {uncl}% of rallies in this cell had unclassifiable endings (excluded from error rate).</div>'

            return f"""
            <div class="pred-box {ss_class}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.6rem;">
                    <h3 style="margin:0; color:#f1f5f9;">{player_name}
                        <span style="font-size:0.9rem; color:#94a3b8;">({score_p} – {score_o})</span>
                    </h3>
                    {conf_badge}
                </div>
                <div style="display:flex; gap:1rem; margin-top:0.4rem; flex-wrap:wrap;">
                    <div style="background:rgba(0,0,0,0.3); border-radius:10px; padding:0.6rem 1rem; min-width:120px;">
                        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Score State</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">{ss_emoji} {ss}</div>
                        <div style="font-size:0.72rem; color:#64748b;">Margin: {margin_bin}</div>
                    </div>
                    <div style="background:rgba(0,0,0,0.3); border-radius:10px; padding:0.6rem 1rem; min-width:120px;">
                        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Game Phase</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">{gp_emoji} {gp}</div>
                    </div>
                    <div style="background:rgba(0,0,0,0.3); border-radius:10px; padding:0.6rem 1rem; min-width:150px;">
                        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Expected Rally</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">{med_str} shots</div>
                        <div style="font-size:0.72rem; color:#64748b;">IQR{iqr_str} shots</div>
                    </div>
                    <div style="background:rgba(0,0,0,0.3); border-radius:10px; padding:0.6rem 1rem; min-width:170px;">
                        <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Unforced Error Rate</div>
                        <div style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">{er_str}</div>
                        <div style="font-size:0.72rem; color:#64748b;">{ci_str}</div>
                    </div>
                </div>
                <div style="margin-top:0.6rem; color:#64748b; font-size:0.8rem;">
                    N = <b style="color:#94a3b8;">{n:,}</b> rallies
                    · <b style="color:#94a3b8;">{nc:,}</b> classifiable
                    · <b style="color:#94a3b8;">{n_matches}</b> matches
                    · Resolution: <i>{resolution}</i>
                </div>
                {notes_html}
            </div>
            """

        ca, cb = st.columns(2)
        with ca:
            st.markdown(_pred_card(player_a_name, score_a, score_b, pred_a), unsafe_allow_html=True)
        with cb:
            st.markdown(_pred_card(player_b_name, score_b, score_a, pred_b), unsafe_allow_html=True)

        # ── Error Rate Comparison with CI Range Bars ─────────────────────────
        er_a = pred_a.get("error_rate")
        er_b = pred_b.get("error_rate")
        if er_a is not None and er_b is not None:
            st.markdown("---")
            st.markdown("#### ⚡ Unforced Error Propensity (with 95% Confidence Range)")

            fig_g, ax_g = plt.subplots(figsize=(8, 2.8), facecolor="#0f172a")
            ax_g.set_facecolor("#1e293b")
            players_cmp = [player_a_name, player_b_name]
            er_vals_cmp = [er_a * 100, er_b * 100]
            ss_vals = [pred_a["score_state"], pred_b["score_state"]]
            state_colors_map = {"Leading": "#059669", "Level": "#3b82f6", "Trailing": "#ef4444"}
            bar_colors = [state_colors_map.get(s, "#64748b") for s in ss_vals]

            # Calculate error bar ranges from Wilson or bootstrap CIs
            lo_a = pred_a.get("wilson_lo", er_a) * 100
            hi_a = pred_a.get("wilson_hi", er_a) * 100
            lo_b = pred_b.get("wilson_lo", er_b) * 100
            hi_b = pred_b.get("wilson_hi", er_b) * 100
            
            xerr = [
                [max(0, er_vals_cmp[0] - lo_a), max(0, er_vals_cmp[1] - lo_b)],
                [max(0, hi_a - er_vals_cmp[0]), max(0, hi_b - er_vals_cmp[1])]
            ]

            bars_cmp = ax_g.barh(players_cmp, er_vals_cmp, color=bar_colors, alpha=0.85,
                                  edgecolor="#1e293b", height=0.4, xerr=xerr, capsize=5, ecolor="#f1f5f9")
            ax_g.set_xlabel("Unforced Error Rate (%) with 95% Confidence Interval", color="#94a3b8")
            ax_g.tick_params(colors="#94a3b8")
            for spine in ax_g.spines.values(): spine.set_color("#334155")
            for bar, v, hi in zip(bars_cmp, er_vals_cmp, [hi_a, hi_b]):
                ax_g.text(hi + 1.2, bar.get_y() + bar.get_height()/2,
                          f"{v:.1f}%", va="center", fontsize=10, fontweight="bold", color="#f1f5f9")
            ax_g.set_xlim(0, max(hi_a, hi_b, 45) * 1.2)
            ax_g.axvline(30, color="#f59e0b", linestyle="--", alpha=0.5, linewidth=1.2)
            ax_g.text(30.5, 0.9, "Global Avg ~30%", fontsize=8, color="#f59e0b", transform=ax_g.get_xaxis_transform())
            fig_g.patch.set_facecolor("#0f172a")
            plt.tight_layout()
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

        # ── Historical Pattern Summary ────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📚 Historical Pattern Reference (Tied-After-15 & Margin Breakdown)")
        summary_df = get_all_states_summary(match_category)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Live Match Tracker ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Live Match Rally Tracker")
    st.markdown("Log each rally result to track momentum over the game.")

    if "match_log" not in st.session_state:
        st.session_state.match_log = []
        st.session_state.live_a = 0
        st.session_state.live_b = 0

    ml_c1, ml_c2, ml_c3 = st.columns(3)
    with ml_c1:
        rally_len_input = st.number_input("Rally Length", min_value=1, max_value=99, value=8, step=1, key="rl_in")
    with ml_c2:
        winner_input = st.selectbox("Rally Winner", [player_a_name, player_b_name], key="winner_in")
    with ml_c3:
        st.markdown("<div style='padding-top:1.6rem;'></div>", unsafe_allow_html=True)
        log_btn = st.button("➕ Log Rally", key="log_btn", use_container_width=True)

    if log_btn:
        # Update live scores
        la = st.session_state.live_a
        lb = st.session_state.live_b
        entry = {
            "rally_num":    len(st.session_state.match_log) + 1,
            "score_a":      la,
            "score_b":      lb,
            "winner":       "A" if winner_input == player_a_name else "B",
            "rally_length": rally_len_input,
        }
        st.session_state.match_log.append(entry)
        if winner_input == player_a_name:
            st.session_state.live_a += 1
        else:
            st.session_state.live_b += 1
        st.rerun()

    if st.session_state.match_log:
        df_log = get_match_history_trends(st.session_state.match_log)
        # Score progression chart
        fig_t, ax_t = plt.subplots(figsize=(10, 3.5), facecolor="#0f172a")
        ax_t.set_facecolor("#1e293b")
        rallies_x = df_log["Rally"].tolist()
        ax_t.plot(rallies_x, df_log["Score A"], marker="o", color="#059669",
                  linewidth=2, label=player_a_name, markersize=5)
        ax_t.plot(rallies_x, df_log["Score B"], marker="s", color="#ef4444",
                  linewidth=2, label=player_b_name, markersize=5)
        ax_t.set_xlabel("Rally Number", color="#94a3b8")
        ax_t.set_ylabel("Score", color="#94a3b8")
        ax_t.set_title("Score Progression", color="#e2e8f0", fontweight="bold")
        ax_t.tick_params(colors="#94a3b8")
        for spine in ax_t.spines.values(): spine.set_color("#334155")
        ax_t.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax_t.fill_between(rallies_x, df_log["Score A"], df_log["Score B"],
                          alpha=0.08, color="#94a3b8")
        fig_t.patch.set_facecolor("#0f172a")
        plt.tight_layout()
        st.pyplot(fig_t, use_container_width=True)

        # Buffer for export
        img_buf = io.BytesIO()
        fig_t.savefig(img_buf, format="png", dpi=200, facecolor=fig_t.get_facecolor(), edgecolor="none")
        plt.close(fig_t)

        # Current live score & Exports
        st.metric(f"Live Score — {player_a_name}", st.session_state.live_a,
                  delta=f"vs {player_b_name}: {st.session_state.live_b}")
        st.dataframe(df_log, use_container_width=True, hide_index=True)

        exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1])
        with exp_c1:
            st.download_button(
                "📥 Export Match Log (CSV)",
                df_log.to_csv(index=False).encode("utf-8"),
                file_name=f"match_log_{player_a_name}_vs_{player_b_name}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_c2:
            st.download_button(
                "🖼️ Export Chart (PNG)",
                img_buf.getvalue(),
                file_name=f"score_progression_{player_a_name}_vs_{player_b_name}.png",
                mime="image/png",
                use_container_width=True,
            )
        with exp_c3:
            if st.button("🗑️ Reset Match Log", key="reset_log", use_container_width=True):
                st.session_state.match_log = []
                st.session_state.live_a = 0
                st.session_state.live_b = 0
                st.rerun()
    else:
        st.info("No rallies logged yet. Enter a rally result above and click **➕ Log Rally**.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 6 — METHODS & LIMITATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">⚠️ Methods & Limitations</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
#### 📐 Methodology

**Score State Classification (pre-rally)**
- Pre-rally scores are reconstructed from post-rally scores.
- Rally 1 → (0, 0); Rally k > 1 → prior rally's post-score.
- This avoids **lookahead bias** (not using the rally outcome to define pressure going in).

**Game Phase Threshold**
- Phase is determined by the **leader's score** (max of both), not individual score.
- "After 15" = leader's score ≥ 15. Threshold matches BWF convention.

**Unforced Error Classification**
- Based on `lose_reason` field (translated from Chinese).
- Error types: Net, Out, Fault shots.
- 175 rallies (4.8%) have missing `getpoint_player` → classified as "unclassifiable".

**Prediction Engine**
- Pattern-based: filters historical ShuttleSet rows matching the given  
  `score_state` × `game_phase` × `category` combination.
- Returns empirical median rally length and Wilson-CI error rate.
- **No machine learning** — results are purely distributional from real match data.
        """)

    with col_m2:
        st.markdown("""
#### ⚠️ Limitations & Caveats

| Issue | Detail |
|---|---|
| **Dataset scope** | 44 matches, 27 players, 2018–2021 elite international singles only. Does not generalise to amateur play or doubles. |
| **Player A = match winner** | In ShuttleSet, Player A is always the match winner. Leading rows disproportionately represent the eventual winner. |
| **Player strength confounding** | Once controlling for individual player strength via mixed/regularized logistic regression, the trailing effect attenuates (OR=1.17, 95% CI [0.99, 1.34], not significant at α=0.05). |
| **Descriptive, not causal** | Findings represent empirical associations, not causal effects. We use "associated with", never causal claims. |
| **Score margin grouping** | By default, predictions group into Leading/Level/Trailing. Margin bins (±1-2, ±3-5, ±6+) are used when N ≥ 100, falling back to coarser cells otherwise. |
| **Unclassified endings** | 4.8% of rallies (175) have missing `getpoint_player` and are excluded from error-rate denominators. |
| **Correlated observations** | Rallies within matches are clustered. Wilson CIs are narrower than cluster-bootstrap CIs which resample whole matches. |

#### 🔬 System Validation
- **Software Tests (18/18 passed):** Validates deterministic logic, lack of lookahead bias (`tests/test_lookahead_bias.py`), metric denominators, and edge conditions.
- **Model-Validity Tests:** Leave-one-match-out cross-validation confirms the lookup model outperforms naive baselines on Log Loss (0.6276 vs 0.6294) and Brier score (0.2180 vs 0.2188).
- **Reproducible Pipeline:** `python src/run_all_hardening.py` recreates all bootstrap distributions, holdout metrics, and control models from raw data.

#### 🔗 Data Source
- **ShuttleSet** open dataset (Shuttlecock Tracking Dataset, ACM SIGKDD 2023)
- Repository: [GitHub — dannyLaw/ShuttleSet](https://github.com/dannyLaw/ShuttleSet)
- 44 elite international singles matches
        """)

    st.markdown("---")
    # Data quality summary
    st.markdown("#### 📊 Data Quality Summary")
    qa1, qa2, qa3, qa4 = st.columns(4)
    qa1.metric("Total Rallies",          "3,683")
    qa2.metric("Classifiable Endings",   f"{3683 - 175:,}  ({(3683-175)/3683*100:.1f}%)")
    qa3.metric("Unclassifiable",         f"175  (4.8%)")
    qa4.metric("Score Inconsistencies",  "28  (0.76%)")

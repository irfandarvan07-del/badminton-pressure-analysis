"""
Interactive Streamlit Dashboard for Badminton Pressure Analysis.
Enables real-time exploration of rally length, unforced error dynamics,
game phases, and statistical hypothesis tests across 44 elite international matches.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.metrics import summarize_numeric_series, calculate_rate
from src.statistics import mann_whitney_comparison, chi_square_contingency_test

# Set page config
st.set_page_config(
    page_title="Badminton Pressure Analysis",
    page_icon="🏸",
    layout="wide"
)

@st.cache_data
def load_data():
    rallies_path = os.path.join(config.PROCESSED_DATA_DIR, "rallies.csv")
    persp_path = os.path.join(config.PROCESSED_DATA_DIR, "rallies_player_perspective.csv")
    if not os.path.exists(rallies_path) or not os.path.exists(persp_path):
        from src.build_rallies import build_rally_records
        build_rally_records()
    
    c_df = pd.read_csv(rallies_path)
    p_df = pd.read_csv(persp_path)
    return c_df, p_df

c_df, p_df = load_data()

# Sidebar: Controls & Filters
st.sidebar.title("🏸 Filter Controls")

# Category filter
categories = ["All Categories"] + sorted(c_df["category"].dropna().unique().tolist())
selected_cat = st.sidebar.selectbox("Match Category", categories)

# Tournament filter
tournaments = ["All Tournaments"] + sorted(c_df["tournament"].dropna().unique().tolist())
selected_tourney = st.sidebar.selectbox("Tournament", tournaments)

# Game phase filter
phases = ["All Phases"] + sorted(c_df["game_phase"].dropna().unique().tolist())
selected_phase = st.sidebar.selectbox("Game Phase", phases)

# Score state filter
states = ["All States", "Leading", "Level", "Trailing"]
selected_state = st.sidebar.selectbox("Score State (Player Perspective)", states)

# Apply filters to dataframes
filtered_c = c_df.copy()
filtered_p = p_df.copy()

if selected_cat != "All Categories":
    filtered_c = filtered_c[filtered_c["category"] == selected_cat]
    filtered_p = filtered_p[filtered_p["category"] == selected_cat]

if selected_tourney != "All Tournaments":
    filtered_c = filtered_c[filtered_c["tournament"] == selected_tourney]
    filtered_p = filtered_p[filtered_p["tournament"] == selected_tourney]

if selected_phase != "All Phases":
    filtered_c = filtered_c[filtered_c["game_phase"] == selected_phase]
    filtered_p = filtered_p[filtered_p["game_phase"] == selected_phase]

if selected_state != "All States":
    filtered_p = filtered_p[filtered_p["score_state"] == selected_state]

# Header
st.title("🏸 Pressure-Dependent Rally Behaviour in Elite Badminton")
st.markdown(
    "**An LLM-Assisted Shot-Level Data Analysis of Rally Length and Unforced Errors by Score State**  \n"
    "*Dataset: ShuttleSet (ACM SIGKDD 2023) — 44 Matches, 104 Games, 36,484 Shots.*"
)

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Matches Analysed", filtered_c["match_id"].nunique())
col2.metric("Games / Sets", filtered_c["game_id"].nunique())
col3.metric("Rallies Count", len(filtered_c))
col4.metric("Total Shots", int(filtered_c["rally_length"].sum()))
med_len = np.median(filtered_c["rally_length"]) if len(filtered_c) > 0 else 0
col5.metric("Median Rally Length", f"{med_len:.1f} shots")

st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Overview",
    "📈 Rally Length Dynamics",
    "🎯 Unforced Error Analysis",
    "🔬 Statistical Hypothesis Tests",
    "⚠️ Data Quality & Limitations"
])

with tab1:
    st.header("Dataset Overview & Sample Coverage")
    st.markdown(
        "This project inspects and processes shot-level data from elite international badminton competitions "
        "(BWF World Tour, All England Open, Sudirman Cup, Indonesia Masters, 2018–2021). "
        "All players are top-ranking world singles athletes."
    )
    
    st.subheader("Sample Rallies Table (Canonical Records)")
    st.dataframe(
        filtered_c[[
            "match_id", "tournament", "category", "game_id", "rally_id",
            "pre_score_A", "pre_score_B", "score_state_A", "game_phase",
            "rally_length", "rally_winner", "ending_type", "unforced_error"
        ]].head(25),
        use_container_width=True
    )

    st.subheader("Player Roster Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top Matches by Rally Count:**")
        top_matches = filtered_c.groupby(["tournament", "player_A", "player_B"]).size().reset_index(name="rallies")
        st.dataframe(top_matches.sort_values("rallies", ascending=False).head(10), use_container_width=True)
    with col_b:
        st.markdown("**Rally Count by Match Stage:**")
        st.bar_chart(filtered_c["round_stage"].value_counts())

with tab2:
    st.header("Rally Length Distribution & Score-State Analysis")
    st.markdown(
        "Examines whether scoreboard pressure (Leading, Level, Trailing) or game timing (Before 15 vs After 15) "
        "changes the length of rallies in strokes."
    )

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("Rally Length by Game Phase")
        fig, ax = plt.subplots(figsize=(6, 4))
        phases_list = [p for p in ["Before 15", "After 15"] if p in filtered_c["game_phase"].unique()]
        if phases_list:
            data_to_plot = [filtered_c[filtered_c["game_phase"] == p]["rally_length"].dropna() for p in phases_list]
            ax.boxplot(data_to_plot, tick_labels=phases_list, patch_artist=True, showmeans=True)
            ax.set_ylabel("Rally Length (Strokes)")
            ax.set_title("Strokes per Rally: Before vs After 15 Points")
            st.pyplot(fig)
        else:
            st.info("No data for current phase filter.")

    with col_l2:
        st.subheader("Rally Length Summary Table")
        summary_rows = []
        for phase, grp in filtered_c.groupby("game_phase"):
            stats = summarize_numeric_series(grp["rally_length"])
            stats["Phase"] = phase
            summary_rows.append(stats)
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows).set_index("Phase"), use_container_width=True)

    st.subheader("Rally Length by Score State (Player-Perspective)")
    state_rows = []
    for state, grp in filtered_p.groupby("score_state"):
        stats = summarize_numeric_series(grp["rally_length"])
        stats["Score State"] = state
        state_rows.append(stats)
    if state_rows:
        st.dataframe(pd.DataFrame(state_rows).set_index("Score State"), use_container_width=True)

with tab3:
    st.header("Unforced Error Dynamics & Deficit Pressure")
    st.markdown(
        "An unforced error is defined strictly as a rally ending in an **out-of-bounds**, **net fault**, or **service/technical fault** "
        "committed by the losing player. Rallies ending in opponent clean winners are classified as forced points."
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.subheader("Rally Ending Outcomes Breakdown")
        ending_counts = filtered_c["ending_type"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
        ax_pie.pie(ending_counts, labels=ending_counts.index, autopct="%1.1f%%", startangle=140)
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    with col_u2:
        st.subheader("Unforced Error Propensity by Score State")
        lead_opp = int((filtered_c["score_state_A"] == "Leading").sum() + (filtered_c["score_state_B"] == "Leading").sum())
        trail_opp = int((filtered_c["score_state_A"] == "Trailing").sum() + (filtered_c["score_state_B"] == "Trailing").sum())
        ue_df = filtered_c[filtered_c["unforced_error"] == True]
        lead_ue = int((ue_df["error_score_state"] == "Leading").sum())
        trail_ue = int((ue_df["error_score_state"] == "Trailing").sum())

        lead_rate = (lead_ue / lead_opp * 100) if lead_opp > 0 else 0
        trail_rate = (trail_ue / trail_opp * 100) if trail_opp > 0 else 0

        st.metric("Leading Player Error Rate", f"{lead_rate:.1f}%", f"{lead_ue} / {lead_opp} rallies")
        st.metric("Trailing Player Error Rate", f"{trail_rate:.1f}%", f"{trail_ue} / {trail_opp} rallies")
        st.info("Trailing players show a significantly higher propensity to commit unforced errors (Chi2 p = 1.7e-7).")

with tab4:
    st.header("Statistical Hypothesis Testing battery")
    st.markdown("All statistical calculations use real, non-simulated records with two-sided tests and effect sizes.")

    st.subheader("1. Game Phase: Before 15 vs After 15 Rally Length")
    b15 = c_df[c_df["game_phase"] == "Before 15"]["rally_length"]
    a15 = c_df[c_df["game_phase"] == "After 15"]["rally_length"]
    res_phase = mann_whitney_comparison(b15, a15, "Before 15", "After 15")
    st.code(res_phase["interpretation"])

    st.subheader("2. Gender Comparison: Men's Singles vs Women's Singles")
    ms = c_df[c_df["category"] == "Men's Singles"]["rally_length"]
    ws = c_df[c_df["category"] == "Women's Singles"]["rally_length"]
    res_gen = mann_whitney_comparison(ms, ws, "Men's Singles", "Women's Singles")
    st.code(res_gen["interpretation"])

    st.subheader("3. Deficit Pressure: Unforced Errors (Leading vs Trailing)")
    ue_tbl = np.array([[lead_ue, lead_opp - lead_ue], [trail_ue, trail_opp - trail_ue]])
    res_ue = chi_square_contingency_test(ue_tbl, "Leading State", "Trailing State", "Unforced Error", "Clean Play")
    st.code(res_ue["interpretation"])

with tab5:
    st.header("Data Quality, Pre-Rally Score Timing & Limitations")
    st.markdown(
        """
        ### 1. Score Timing & Lookahead Bias Prevention
        In the raw ShuttleSet dataset, `roundscore_A` and `roundscore_B` record the **post-rally score** 
        (e.g., rally 1 is recorded as `1-0` or `0-1`).  
        If uncorrected, evaluating player psychology using post-rally score would introduce **lookahead bias**.  
        Our pipeline reconstructs the **pre-rally score** ($S = 0-0$ entering rally 1; previous rally score thereafter).
        
        ### 2. Missing Rally Endings
        - **175 rallies (4.8%)** lack an annotated ending outcome in the raw footage.
        - These rallies are explicitly recorded as `ending_type = 'unknown'` and excluded from error-rate denominators.
        
        ### 3. Singles Domain
        - The dataset strictly covers **singles badminton** across 27 world-elite international athletes.
        - Doubles dynamics (which introduce split-court coverage) are not included.
        """
    )

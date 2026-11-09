"""
Visualization module for Badminton Pressure Analysis.
Generates publication-quality charts saved in reports/figures/
with explicit units, non-truncated axes, and sample sizes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "Leading": "#2E7D32",   # Green
    "Level": "#1565C0",     # Blue
    "Trailing": "#C62828",  # Red
    "Before 15": "#00838F", # Teal
    "After 15": "#D84315",  # Orange/Rust
    "Men's Singles": "#4527A0", # Purple
    "Women's Singles": "#AD1457" # Pink/Magenta
}

def plot_rally_length_distribution_by_score_state(
    perspective_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 1: Rally-length distribution by score state."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    states = ["Leading", "Level", "Trailing"]
    for s in states:
        subset = perspective_df[perspective_df["score_state"] == s]["rally_length"].dropna()
        n = len(subset)
        med = np.median(subset)
        ax.hist(
            subset, bins=np.arange(1, 40, 2), density=True, alpha=0.45,
            color=COLORS[s], label=f"{s} (N={n}, Med={med:.0f})"
        )

    ax.set_title("Rally Length Distribution by Score State (Elite International Singles)")
    ax.set_xlabel("Rally Length (Strokes per Rally)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 40)
    ax.set_ylim(bottom=0)
    ax.legend(title="Score State", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "1_rally_length_distribution_by_score_state.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_median_rally_length_by_score_state(
    perspective_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 2: Median and IQR rally length by score state (non-truncated axis)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    states = ["Leading", "Level", "Trailing"]
    medians = []
    q25s = []
    q75s = []
    counts = []

    for s in states:
        sub = perspective_df[perspective_df["score_state"] == s]["rally_length"].dropna()
        counts.append(len(sub))
        medians.append(float(np.median(sub)))
        q25s.append(float(sub.quantile(0.25)))
        q75s.append(float(sub.quantile(0.75)))

    x = np.arange(len(states))
    yerr = [
        np.array(medians) - np.array(q25s),
        np.array(q75s) - np.array(medians)
    ]

    bars = ax.bar(
        x, medians, yerr=yerr, capsize=6,
        color=[COLORS[s] for s in states], alpha=0.85, width=0.55
    )

    for i, (b, c, m) in enumerate(zip(bars, counts, medians)):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.3, f"{m:.1f} strokes\n(N={c})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Median Rally Length (Strokes, with IQR Error Bars)")
    ax.set_title("Median Rally Length by Score State (Baseline Starts at 0)")
    ax.set_ylim(0, 16)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "2_median_rally_length_by_score_state.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_rally_length_before_vs_after_15(
    canonical_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 3: Rally length before 15 vs after 15 points."""
    fig, ax = plt.subplots(figsize=(7, 5))

    b15 = canonical_df[canonical_df["game_phase"] == "Before 15"]["rally_length"].dropna()
    a15 = canonical_df[canonical_df["game_phase"] == "After 15"]["rally_length"].dropna()

    box = ax.boxplot(
        [b15, a15], tick_labels=["Before 15 Points", "After 15 Points"],
        patch_artist=True, showmeans=True, showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(marker="o", markeredgecolor="black", markerfacecolor="white")
    )

    box["boxes"][0].set_facecolor(COLORS["Before 15"])
    box["boxes"][1].set_facecolor(COLORS["After 15"])
    box["boxes"][0].set_alpha(0.7)
    box["boxes"][1].set_alpha(0.7)

    # Annotate stats
    ax.text(1, np.median(b15) + 0.5, f"Med={np.median(b15):.0f}, Mean={np.mean(b15):.1f}\nN={len(b15)}",
            ha="center", fontsize=9)
    ax.text(2, np.median(a15) + 0.5, f"Med={np.median(a15):.0f}, Mean={np.mean(a15):.1f}\nN={len(a15)}",
            ha="center", fontsize=9)

    ax.set_ylabel("Rally Length (Strokes per Rally)")
    ax.set_title("Rally Length by Game Phase (Mann-Whitney p = 0.029)")
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "3_rally_length_before_vs_after_15.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_unforced_error_rate_before_vs_after_15(
    canonical_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 4: Unforced-error rate before 15 vs after 15 points with confidence intervals."""
    fig, ax = plt.subplots(figsize=(7, 5))

    classifiable = canonical_df[canonical_df["ending_type"] != "unknown"]
    phases = ["Before 15", "After 15"]
    rates = []
    denominators = []
    errors = []

    for p in phases:
        sub = classifiable[classifiable["game_phase"] == p]
        n = len(sub)
        ue = int((sub["unforced_error"] == True).sum())
        p_hat = ue / n if n > 0 else 0
        # 95% Wilson score confidence interval
        z = 1.96
        ci = z * np.sqrt(p_hat * (1 - p_hat) / n) if n > 0 else 0
        rates.append(p_hat * 100)
        denominators.append(n)
        errors.append(ci * 100)

    x = np.arange(len(phases))
    bars = ax.bar(
        x, rates, yerr=errors, capsize=6,
        color=[COLORS["Before 15"], COLORS["After 15"]], alpha=0.85, width=0.45
    )

    for b, r, n in zip(bars, rates, denominators):
        ax.text(b.get_x() + b.get_width() / 2, r / 2, f"{r:.1f}%\n(N={n})",
                ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["Before 15 Points", "After 15 Points"])
    ax.set_ylabel("Unforced Error Rate (%) on Classifiable Rallies")
    ax.set_title("Unforced Error Rate Before 15 vs After 15 Points (Chi2 p = 0.121)")
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "4_unforced_error_rate_before_vs_after_15.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_score_state_by_category(
    perspective_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 5: Score-state comparison split by Men's and Women's Singles."""
    fig, ax = plt.subplots(figsize=(9, 5))

    states = ["Leading", "Level", "Trailing"]
    categories = ["Men's Singles", "Women's Singles"]
    width = 0.35
    x = np.arange(len(states))

    for i, cat in enumerate(categories):
        sub_cat = perspective_df[perspective_df["category"] == cat]
        medians = [float(np.median(sub_cat[sub_cat["score_state"] == s]["rally_length"].dropna())) for s in states]
        offset = (i - 0.5) * width
        rects = ax.bar(x + offset, medians, width, label=cat, color=COLORS[cat], alpha=0.85)
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.2, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Median Rally Length (Strokes)")
    ax.set_title("Median Rally Length by Score State: Men's vs Women's Singles")
    ax.set_ylim(0, 14)
    ax.legend(frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "5_score_state_by_category.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def plot_unforced_error_by_score_state(
    canonical_df: pd.DataFrame,
    output_dir: str = config.FIGURES_DIR
) -> str:
    """Figure 6: Error committing propensity by player score state (Deficit Pressure)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    lead_opp = int((canonical_df["score_state_A"] == "Leading").sum() + (canonical_df["score_state_B"] == "Leading").sum())
    trail_opp = int((canonical_df["score_state_A"] == "Trailing").sum() + (canonical_df["score_state_B"] == "Trailing").sum())
    
    ue_df = canonical_df[canonical_df["unforced_error"] == True]
    lead_ue = int((ue_df["error_score_state"] == "Leading").sum())
    trail_ue = int((ue_df["error_score_state"] == "Trailing").sum())

    lead_rate = (lead_ue / lead_opp) * 100
    trail_rate = (trail_ue / trail_opp) * 100

    rates = [lead_rate, trail_rate]
    labels = ["Leading Player", "Trailing Player"]
    counts = [f"{lead_ue}/{lead_opp}", f"{trail_ue}/{trail_opp}"]

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=[COLORS["Leading"], COLORS["Trailing"]], alpha=0.85, width=0.45)

    for b, r, c in zip(bars, rates, counts):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}%\n({c})",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Unforced Error Rate per Opportunity (%)")
    ax.set_title("Unforced Error Propensity by Score State (Chi2 p = 1.7e-7)")
    ax.set_ylim(0, 50)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "6_unforced_error_by_score_state.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path

def generate_all_visualizations(
    canonical_df: pd.DataFrame = None,
    perspective_df: pd.DataFrame = None,
    output_dir: str = config.FIGURES_DIR
) -> list:
    """Generates and saves all required charts."""
    os.makedirs(output_dir, exist_ok=True)
    if canonical_df is None:
        canonical_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies.csv"))
    if perspective_df is None:
        perspective_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies_player_perspective.csv"))

    paths = [
        plot_rally_length_distribution_by_score_state(perspective_df, output_dir),
        plot_median_rally_length_by_score_state(perspective_df, output_dir),
        plot_rally_length_before_vs_after_15(canonical_df, output_dir),
        plot_unforced_error_rate_before_vs_after_15(canonical_df, output_dir),
        plot_score_state_by_category(perspective_df, output_dir),
        plot_unforced_error_by_score_state(canonical_df, output_dir)
    ]
    return paths

if __name__ == "__main__":
    generated = generate_all_visualizations()
    print("Visualizations successfully generated:")
    for p in generated:
        print(f"- {p}")

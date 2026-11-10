# Pressure-Dependent Rally Behaviour by Score State in Elite Badminton

An empirical, software-only shot-level data analysis investigating whether elite international badminton players alter their rally length and unforced error propensities depending on score state (Leading, Level, Trailing) and game phase (Before vs. After 15 points).

---

## 1. Project Overview & Research Questions

This project answers four core empirical questions using human-annotated stroke-level match data:

1. **Do players behave differently when they are leading, level, or trailing?**
2. **Does rally length change in the later part of a game (after 15 points)?**
3. **Do unforced errors become more frequent after 15 points?**
4. **Are these behavioural patterns different between Men's and Women's Singles?**

---

## 2. Dataset

- **Source:** ShuttleSet (A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis, ACM SIGKDD 2023)
- **Repository:** `wywyWang/CoachAI-Projects`
- **Matches:** 44 elite international singles matches (33 Men's Singles, 11 Women's Singles)
- **Games / Sets:** 104 played sets
- **Rallies:** 3,683 rallies
- **Shots (strokes):** 36,484 annotated strokes
- **Athletes:** 27 world-elite singles players (e.g., Kento Momota, Viktor Axelsen, Chou Tien Chen, Anthony Sinisuka Ginting, Anders Antonsen, An Se Young, Carolina Marin, Pusarla V. Sindhu, Ratchanok Intanon).

### Key Methodological Definitions
- **Pre-Rally Score State:** Player's score entering the rally (0–0 for rally 1; prior rally post-score thereafter). Avoids target leakage / lookahead bias present in raw post-rally scores.
  - *Leading:* Player score > Opponent score
  - *Level:* Player score == Opponent score
  - *Trailing:* Player score < Opponent score
- **Game Phase:**
  - *Before 15 points:* Leading player score < 15
  - *After 15 points:* Leading player score $\ge$ 15
- **Unforced Error:** A rally ending in an out-of-bounds (`出界`), net fault (`掛網`/`未過網`), or technical/service fault (`犯規`) by the losing player. Clean winners by the opponent (`對手落地致勝`) are classified as forced points.

---

## 3. Directory Structure

```text
badminton/
├── data/
│   ├── raw/shuttleset/       # 44 match directories + match.csv (never modified)
│   ├── processed/            # cleaned_shots.csv, rallies.csv, rallies_player_perspective.csv
│   └── README.md             # Dataset documentation and conventions
├── src/
│   ├── __init__.py
│   ├── download_shuttleset.py# High-performance tree downloader
│   ├── load_data.py          # Match registry loader & file scanner
│   ├── clean_data.py         # Validation, translation, and quality audit
│   ├── build_rallies.py      # Pre-rally score tracking & rally record builder
│   ├── score_state.py        # Score state and game phase classification logic
│   ├── metrics.py            # Descriptive statistics & error rates with denominators
│   ├── statistics.py         # Mann-Whitney U, Kruskal-Wallis, Chi2, Wilcoxon tests
│   └── visualizations.py     # Publication-grade chart generation
├── app/
│   └── dashboard.py          # Interactive Streamlit dashboard
├── reports/
│   ├── figures/              # 6 publication charts
│   ├── tables/               # Exported CSV summary tables
│   └── research_report.md    # Comprehensive academic research report
├── tests/
│   ├── test_score_state.py   # Unit tests for score state & game phase
│   ├── test_metrics.py       # Unit tests for rates, zero-denominators, summaries
│   └── test_build_rallies.py # Unit tests for rally construction & stroke counting
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project README
└── main.py                   # End-to-end execution pipeline
```

---

## 4. Installation & Execution

### 4.1 Prerequisites
- Python 3.9+ (tested on Python 3.14)

### 4.2 Setup
```bash
# Optional: create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Run the Complete Pipeline
```bash
python main.py
```
This runs the full end-to-end pipeline:
1. Validates and cleans raw data
2. Builds rally records
3. Calculates descriptive metrics
4. Executes statistical tests
5. Generates figures in `reports/figures/`
6. Exports tables in `reports/tables/`
7. Prints console research answers

### 4.4 Run Tests
```bash
python -m pytest tests/
```

### 4.5 Launch Interactive Dashboard
```bash
streamlit run app/dashboard.py
```

---

## 5. Summary of Empirical Findings

| Research Question | Metric / Finding | Statistical Test | Result |
| :--- | :--- | :--- | :--- |
| **Q1: Leading vs Trailing** | Stroke lengths are identical (Med=8.0); **Trailing players commit unforced errors at 33.6% vs 27.6% for Leading players**. | Chi-square = 27.33, $p = 1.72 \times 10^{-7}$ | **Statistically Significant Deficit Pressure** |
| **Q2: Game Phase (Late Game)** | Rallies shorten after 15 points (Median: 8.0 $\to$ 7.0 strokes; Mean: 10.04 $\to$ 9.61 strokes). | Mann-Whitney $U = 1,495,430.5$, $p = 0.0293$ | **Statistically Significant Reduction** |
| **Q3: Unforced Errors Late Game** | Error rate remains stable before 15 (65.6%) vs after 15 (62.4%). | Chi-square = 3.276, $p = 0.0703$ | **No Significant Increase** |
| **Q4: Men's vs Women's Singles** | Men's rallies (Med=8.0, Mean=10.3) are longer than Women's rallies (Med=7.0, Mean=8.55). Error rates do not differ significantly (64.1% vs 66.6%). | Mann-Whitney $U = 1,301,502.5$, $p = 1.06 \times 10^{-5}$ | **Statistically Significant Difference in Rally Length** |


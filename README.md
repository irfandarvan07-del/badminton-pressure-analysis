# Pressure-Dependent Rally Behaviour by Score State in Elite Badminton

[![Tests](https://img.shields.io/badge/tests-18%2F18%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-ShuttleSet%20(KDD%202023)-orange.svg)](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](app/dashboard.py)

An empirical, software-only stroke-level data analysis investigating whether elite international badminton players alter their rally length and unforced error propensities depending on score state (**Leading**, **Level**, **Trailing**) and game phase (**Before vs. After 15 points**).

---

## 1. Project Overview & Research Questions

This project answers four core empirical questions using human-annotated stroke-level match data:

1. **Do players behave differently when they are leading, level, or trailing?**
2. **Does rally length change in the later part of a game (after 15 points)?**
3. **Do unforced errors become more frequent after 15 points?**
4. **Are these behavioural patterns different between Men's and Women's Singles?**

---

## 2. Dataset

- **Source:** ShuttleSet (*A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis*, ACM SIGKDD 2023)
- **Repository:** `wywyWang/CoachAI-Projects`
- **Matches:** 44 elite international singles matches (33 Men's Singles, 11 Women's Singles)
- **Games / Sets:** 104 played sets
- **Rallies:** 3,683 continuous rallies
- **Shots (strokes):** 36,483 annotated strokes
- **Athletes:** 27 world-elite singles players (e.g., Kento Momota, Viktor Axelsen, Chou Tien Chen, Anthony Sinisuka Ginting, Anders Antonsen, An Se Young, Carolina Marin, Pusarla V. Sindhu, Ratchanok Intanon).

### Key Methodological Definitions
- **Pre-Rally Score State:** Player's score entering the rally ($0–0$ for rally 1; prior rally post-score thereafter). Eliminates target leakage / lookahead bias present in raw post-rally scores.
  - *Leading:* Player score > Opponent score
  - *Level:* Player score == Opponent score
  - *Trailing:* Player score < Opponent score
- **Game Phase:**
  - *Before 15 points:* Leading player score < 15
  - *After 15 points:* Leading player score $\ge$ 15
- **Unforced Error Rate:** 
  $$\text{Unforced Error Rate} = \frac{\#\text{ rallies ending in this player's unforced error}}{\#\text{ classifiable rallies in this cell from this player's perspective}}$$
  Unforced errors include out-of-bounds (`出界`), net faults (`掛網`/`未過網`), and technical/service faults (`犯規`). Opponent winners (`對手落地致勝`) are classified as forced points. 175 rallies (4.8%) with missing outcome attributions are strictly excluded from denominators.

---

## 3. Directory Structure

```text
badminton/
├── data/
│   ├── raw/shuttleset/       # 44 match directories + match.csv (immutable)
│   ├── processed/            # cleaned_shots.csv, rallies.csv, rallies_player_perspective.csv
│   └── README.md             # Dataset documentation and conventions
├── src/
│   ├── __init__.py
│   ├── download_shuttleset.py# Tree-based raw dataset downloader
│   ├── load_data.py          # Match registry loader & file scanner
│   ├── clean_data.py         # Validation, translation, and quality audit
│   ├── build_rallies.py      # Pre-rally score tracking & rally record builder
│   ├── score_state.py        # Score state and game phase classification logic
│   ├── metrics.py            # Descriptive statistics & rate calculations
│   ├── statistics.py         # Mann-Whitney U, Kruskal-Wallis, Chi2, Wilcoxon tests
│   ├── visualizations.py     # Publication-grade chart generation
│   ├── predictor.py          # Pattern-based lookup engine with margin bins & fallbacks
│   ├── bootstrap.py          # Fast match-level cluster-bootstrap CIs (2,000 resamples)
│   ├── holdout_eval.py       # Leave-one-match-out cross-validation & calibration
│   ├── player_strength.py    # Regularized logistic model controlling player ability
│   └── run_all_hardening.py  # Master script reproducing all hardening metrics
├── app/
│   └── dashboard.py          # Interactive 6-tab Streamlit dashboard
├── reports/
│   ├── figures/              # 7 research charts (including LOMO calibration curve)
│   ├── tables/               # Exported CSV tables & JSON evaluation summaries
│   ├── research_report.md    # Full academic research report
│   └── hardened_execution_report.md # Formal statistical hardening & verification report
├── tests/
│   ├── test_score_state.py   # Unit tests for score state & game phase
│   ├── test_metrics.py       # Unit tests for rates, zero-denominators, summaries
│   ├── test_build_rallies.py # Unit tests for rally construction & stroke counting
│   └── test_lookahead_bias.py# 5 tests proving pre-rally scores prevent lookahead bias
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── main.py                   # End-to-end data pipeline execution
```

---

## 4. Installation & Execution

### 4.1 Prerequisites
- Python 3.9+ (tested on Python 3.14 on Windows)

### 4.2 Setup
```bash
# Clone the repository
git clone https://github.com/irfandarvan07-del/badminton-pressure-analysis.git
cd badminton-pressure-analysis

# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Run the Complete Pipeline
```bash
python main.py
```
This runs the full data pipeline:
1. Validates and cleans raw data
2. Builds pre-rally scored records
3. Calculates descriptive metrics
4. Executes statistical hypothesis tests
5. Exports publication figures to `reports/figures/`
6. Exports tables to `reports/tables/`

### 4.4 Run the Master Hardening Pipeline
```bash
python src/run_all_hardening.py
```
Executes:
- 2,000 cluster-bootstrap iterations across all cells
- Leave-one-match-out cross-validation and calibration plots
- Match-clustered claim significance testing
- Player strength and selection bias control regression

### 4.5 Run Tests
```bash
python -m pytest tests/ -v
```
All **18 unit tests** pass in ~1.3 seconds.

### 4.6 Launch Interactive Dashboard
```bash
python -m streamlit run app/dashboard.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 5. Summary of Empirical Findings

| Research Question | Metric / Finding | Statistical Test | Cluster-Bootstrap Result | Scientific Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Q1: Leading vs Trailing** | Stroke lengths are identical (Med=8.0); **Trailing players commit unforced errors at 33.6% vs 27.6% for Leading players**. | $\chi^2 = 27.33, p = 1.72 \times 10^{-7}$ | Diff = $+5.3\text{ pp}$ ($95\%\text{ CI: } [+1.1, +9.6]$, $p = 0.016$) Before 15; Diff = $+10.8\text{ pp}$ ($95\%\text{ CI: } [+6.1, +15.6]$, $p < 0.0001$) After 15 | **Statistically Significant Association** |
| **Q2: Game Phase (Late Game)** | Rallies shorten after 15 points (Median: 8.0 $\to$ 7.0 strokes; Mean: 10.04 $\to$ 9.61 strokes). | Mann-Whitney $U = 1,495,430.5, p = 0.0293$ | Leader error drop: Diff = $+4.2\text{ pp}$ ($95\%\text{ CI: } [+0.9, +7.9]$, $p = 0.020$) | **Statistically Significant Pacing Shift** |
| **Q3: Unforced Errors Late Game** | Error rate remains stable before 15 (65.6%) vs after 15 (62.4%). | $\chi^2 = 3.276, p = 0.0703$ | Cluster CI spans zero ($p = 0.07$) | **No Significant Error Surge** |
| **Q4: Men's vs Women's Singles** | Men's rallies (Med=8.0, Mean=10.3) vs Women's rallies (Med=7.0, Mean=8.55). Error rates do not differ significantly (64.1% vs 66.6%). | Mann-Whitney $U = 1,301,502.5, p = 1.06 \times 10^{-5}$ | Diff = $+1.0\text{ shots}$ ($95\%\text{ CI: } [0.0, +2.0]$, $p = 0.0675$) | **Inconclusive Under Cluster Resampling** |

---

## 6. Model-Validity & Hardened Statistical Tests

### 6.1 Cluster-Bootstrap Distributions (2,000 Match-Level Resamples)
Standard tests assume rallies are independent; cluster bootstrapping accounts for intra-match correlation by resampling whole matches:

| Score State | Game Phase | Category | Error Rate | 95% Cluster-Bootstrap CI | 95% Wilson CI | Median Rally (CI) | Rally IQR | Confidence |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Leading** | After 15 | Men's Singles | **25.5%** | **[22.5%, 28.5%]** | [22.5%, 28.8%] | 8.0 [7.0, 8.0] | 8.0 | 🟢 High |
| **Tied, after 15**| After 15 | Men's Singles | **33.3%** | **[27.8%, 38.9%]** | [25.5%, 42.2%] | 9.0 [6.0, 12.0]| 10.0 | 🟡 Medium |
| **Trailing** | After 15 | Men's Singles | **36.3%** | **[33.3%, 39.6%]** | [32.9%, 39.8%] | 8.0 [7.0, 8.0] | 8.0 | 🟢 High |
| **Leading** | Before 15 | Men's Singles | **29.8%** | **[27.5%, 32.1%]** | [27.5%, 32.1%] | 8.0 [8.0, 9.0] | 9.0 | 🟢 High |
| **Level** | Before 15 | Men's Singles | **32.5%** | **[29.9%, 34.9%]** | [28.9%, 36.3%] | 8.0 [7.0, 9.0] | 9.0 | 🟢 High |
| **Trailing** | Before 15 | Men's Singles | **35.1%** | **[32.6%, 37.5%]** | [32.8%, 37.5%] | 8.0 [8.0, 9.0] | 9.0 | 🟢 High |

### 6.2 Holdout Cross-Validation (Leave-One-Match-Out)
Evaluated across all 44 matches (7,016 classifiable player-rallies):

| Evaluation Model | Out-of-Sample Log Loss | Out-of-Sample Brier Score | Relative Advantage |
| :--- | :---: | :---: | :--- |
| **Global Baseline** | 0.6294 | 0.2188 | Unconditional match mean |
| **Category Baseline** | 0.6294 | 0.2188 | Stratified by Men's / Women's Singles |
| **Lookup (Category-Specific)** | 0.6281 | 0.2182 | State $\times$ Phase lookup table per category |
| **Lookup (All Categories)** | **0.6276** | **0.2180** | **Superior out-of-sample calibration & accuracy** |

*Calibration Curve:* Available at `reports/figures/7_holdout_calibration.png`.

### 6.3 Player Strength & Selection Bias Control
In ShuttleSet, Player A is uniformly the match victor. To disentangle player skill from score-state pressure, we fitted an L2-regularized logistic regression with player and match random-effect dummy intercepts:
$$\text{logit}(P(\text{unforced error})) \sim \text{score\_state} + \text{game\_phase} + \text{category} + (1|\text{player}) + (1|\text{match})$$

| Predictor | Coef | Odds Ratio (OR) | 95% Cluster-Bootstrap CI (OR) | $p < 0.05$ | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Score State: Trailing vs Leading** | +0.158 | **1.171** | **[0.992, 1.339]** | ❌ No | Attenuates when player ability and winner selection are controlled. |
| **Score State: Level vs Leading** | +0.120 | 1.127 | [0.942, 1.290] | ❌ No | Statistically indistinguishable from leading. |
| **Game Phase: After 15 vs Before 15**| -0.070 | 0.932 | [0.855, 1.007] | ❌ No | Directional reduction, non-significant. |
| **Category: Women's vs Men's** | -0.504 | 0.604 | [0.658, 0.928] | ✅ Yes | Statistically significant difference between categories. |

> **Key Discovery:** Unadjusted calculations show trailing players committing $33.6\%$ vs $27.6\%$ errors ($p < 0.001$). However, because stronger players disproportionately hold leads and commit fewer errors overall, controlling for player strength attenuates the odds ratio from $1.33 \to 1.17$ (95% CI: `[0.992, 1.339]`). The deficit pressure surge is an observational association rather than a pure causal impact.

---

## 7. Interactive Dashboard Features

The Streamlit dashboard (`app/dashboard.py`) includes 6 comprehensive tabs:

1. **🏠 Overview:** Hero KPIs (44 matches, 3,683 rallies, 36,483 shots), score-state distribution pie charts, and core findings.
2. **📊 Rally Analysis:** Interactive box plots and grouped bar charts with phase filters and IQR metrics.
3. **🎯 Error Analysis:** Cross-tab error rate heatmaps (`Score State × Game Phase`) and phase breakdowns.
4. **🔬 Statistical Tests:** Standard hypothesis tests, Leave-One-Match-Out holdout tables, calibration plots, player strength models, and bootstrap claims.
5. **🏆 Live Match Predictor:**
   - Pre-rally score inputs with automatic game-phase and score-state determination.
   - Confidence badges (🟢 High / 🟡 Medium / 🔴 Low confidence for $N < 100$).
   - Horizontal CI range bars visualizing 95% error rate intervals alongside global averages.
   - Granular margin bins ($\pm 1-2, \pm 3-5, \pm 6+$) with automatic fallbacks to coarser cells.
   - Separate **True Deuce** cell ($\ge 20-\ge 20$) with low-$N$ warning.
   - **Live Match Rally Tracker** with export buttons:
     - 📥 **Export Match Log (CSV)**
     - 🖼️ **Export Score Progression Chart (PNG)**
6. **⚠️ Methods & Limitations:** Detailed methodology, data quality audit, and limitation notes.

---

## 8. Limitations & Scope

> **This study is descriptive and observational. All findings are expressed as empirical associations, never causal claims.**

1. **Dataset Scope:** 44 matches between 27 elite international singles players (2018–2021). Results cannot be extrapolated to amateur, junior, or doubles competition.
2. **Player A Selection Bias:** In ShuttleSet, Player A is always the match victor. Leading rows disproportionately represent the eventual winner.
3. **Player Ability Confounding:** Controlling for player ability reduces the trailing error effect from OR=1.33 to OR=1.17 (CI spans 1.00).
4. **Coarse Margin Binning:** Score margins ($\pm 1-2, \pm 3-5, \pm 6+$) require $N \ge 100$ and fall back to broader categories when sparse.
5. **Unclassifiable Endings:** 175 rallies (4.8%) lacked camera-verified ending attributions and are excluded from error rates.
6. **Correlated Observations:** Rallies within matches are clustered. Wilson CIs are narrower than cluster-bootstrap CIs which resample whole matches.

---

## 9. Reproducibility

To reproduce all results, tables, and figures from scratch:

```bash
# Step 1: Download raw ShuttleSet data
python src/download_shuttleset.py

# Step 2: Build processed rally datasets
python main.py

# Step 3: Run full unit test suite (including lookahead bias tests)
python -m pytest tests/ -v

# Step 4: Run master hardening pipeline (bootstrap, holdout, player strength)
python src/run_all_hardening.py

# Step 5: Launch dashboard
python -m streamlit run app/dashboard.py
```

---

## 10. License & Citation

- **Code:** MIT License
- **Dataset:** Human-annotated stroke data licensed under the original ShuttleSet repository terms.
- **Citation:**
  ```bibtex
  @inproceedings{wang2023shuttleset,
    title={ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
    author={Wang, Wei-Yao and Huang, Yu-Ching and Ik, Teng-U and Peng, Wen-Chih},
    booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)},
    year={2023}
  }
  ```

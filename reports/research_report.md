# Pressure-Dependent Rally Behaviour by Score State in Elite Badminton

**Authors:** Group 43 (Rohith Raj Reddy, Dhyan M, Md Irfan Darvan)  
**Affiliation:** Department of Electronics & Telecommunication Engineering / Electronics & Communication Engineering, RV College of Engineering  
**Study Design:** Quantitative Shot-Level Observational Analytics with LLM-Assisted Software Verification  
**Dataset:** ShuttleSet (ACM SIGKDD 2023)  

---

## 1. Research Questions

This study investigates whether elite international badminton players alter their strategic behaviour under scoreboard pressure. Specifically, we address four empirical questions:

1. **Score-State Dependency:** Do players change rally length or error propensity depending on whether they are leading, level, or trailing?
2. **Game-Phase Dynamics:** Does rally length shorten or lengthen in the later part of a game (after 15 points)?
3. **Closing-Stage Error Frequency:** Do unforced errors become more frequent after 15 points under end-game psychological pressure?
4. **Gender Disparities:** Do these pressure-dependent patterns differ between Men's Singles (MS) and Women's Singles (WS)?

---

## 2. Dataset and Source Information

- **Dataset Name:** ShuttleSet (A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis)
- **Primary Reference:** W.-Y. Wang, Y.-C. Huang, T.-U. Ik, and W.-C. Peng, *ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis*, ACM SIGKDD, 2023.
- **Repository:** `https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet`
- **Competition Domain:** BWF World Tour, All England Open, Sudirman Cup, Indonesia Masters, 2018–2021.
- **Athletes:** 27 world-elite international singles players (e.g., Kento Momota, Viktor Axelsen, Chou Tien Chen, Anthony Sinisuka Ginting, Anders Antonsen, An Se Young, Carolina Marin, Pusarla V. Sindhu, Ratchanok Intanon).

---

## 3. Data Coverage

| Level | Count | Description |
| :--- | :--- | :--- |
| **Matches** | 44 | 33 Men's Singles (75.0%), 11 Women's Singles (25.0%) |
| **Sets / Games** | 104 | 56 two-set matches, 48 three-set matches |
| **Rallies** | 3,683 | Valid continuous rallies evaluated |
| **Shots (strokes)**| 36,483 | Granular, human-annotated stroke events |
| **Players Evaluated**| 27 | Top-ranking BWF international athletes |

---

## 4. Definitions and Methodology

### 4.1 Rally Length
Rally length is defined strictly as the **total number of strokes (shots) exchanged in a single rally**, starting with the service and terminating at the point-ending stroke.

### 4.2 Score State and Pre-Rally Timing
In raw ShuttleSet records, `roundscore_A` and `roundscore_B` capture the **post-rally score** (e.g., rally 1 is recorded as `1-0` or `0-1`). To eliminate **lookahead bias / target leakage**, our methodology computes the **pre-rally score** entering the point:
- For rally $k = 1$: $S_A = 0, S_B = 0$ (Level).
- For rally $k > 1$: Prior rally's post-score ($S_{A, k-1}, S_{B, k-1}$).

Classification rules:
- **Leading:** $\text{Player Score} > \text{Opponent Score}$
- **Level:** $\text{Player Score} == \text{Opponent Score}$
- **Trailing:** $\text{Player Score} < \text{Opponent Score}$

### 4.3 Game Phase
- **Before 15 points:** $\max(S_A, S_B) < 15$
- **After 15 points:** $\max(S_A, S_B) \ge 15$

### 4.4 Unforced Errors
An unforced error is defined as a classifiable rally-ending fault committed by the losing player without direct forced winner intervention:
- **Out of bounds (`出界`):** 1,047 rallies (28.4%)
- **Net fault / failed to cross (`掛網` / `未過網`):** 1,217 rallies (33.0%)
- **Service / technical fault (`犯規`):** 4 rallies (0.1%)
- *Total Unforced Errors:* **2,268 rallies** ($64.7\%$ of classifiable rallies).
- *Opponent Winners (Forced):* **1,189 rallies** ($33.9\%$).
- *Judgment Errors:* 51 rallies ($1.5\%$).
- *Unclassified / Missing:* 175 rallies ($4.8\%$).

---

## 5. Descriptive Statistics

### 5.1 Rally Length Distributions (Strokes)

| Group | Count ($N$) | Median | Mean | Std Dev | IQR | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Dataset** | 3,683 | **8.0** | 9.91 | 7.84 | 9.0 | 1 | 65 |
| **Before 15 Points** | 2,569 | **8.0** | 10.04 | 7.96 | 9.0 | 1 | 65 |
| **After 15 Points** | 1,114 | **7.0** | 9.61 | 7.54 | 9.0 | 1 | 58 |
| **Leading State** | 3,209 | **8.0** | 9.78 | 7.74 | 9.0 | 1 | 65 |
| **Level State** | 948 | **8.0** | 10.73 | 8.28 | 9.0 | 1 | 61 |
| **Trailing State** | 3,209 | **8.0** | 9.78 | 7.74 | 9.0 | 1 | 65 |
| **Men's Singles** | 2,854 | **8.0** | 10.30 | 7.93 | 9.0 | 1 | 65 |
| **Women's Singles** | 829 | **7.0** | 8.55 | 7.37 | 8.0 | 1 | 58 |

---

## 6. Statistical Test Results

### 6.1 Q1: Score State Comparison (Leading vs. Trailing)
- **Rally Length (Two-sided Mann-Whitney U):**
  - $U = 5,148,840.5$, $p = 1.0000$, rank-biserial $r = 0.000$.
  - *Finding:* Because every singles rally is co-constructed by a leading and trailing competitor, overall rally length distributions are mechanically symmetrical.
  - *Kruskal-Wallis across 3 states (Leading, Level, Trailing):* $H = 5.674, p = 0.0586$ (Level rallies are slightly longer at mean=10.73 strokes, reflecting tactical parity).
- **Unforced Error Committing Rate (Chi-Square Test):**
  - Leading player error rate: **27.58%** (885 errors / 3,209 opportunities).
  - Trailing player error rate: **33.62%** (1,079 errors / 3,209 opportunities).
  - $\chi^2 = 27.33$, $\text{dof} = 1$, **$p = 1.72 \times 10^{-7}$** (Odds Ratio = $1.331$).
  - *Finding:* Trailing players exhibit a highly statistically significant surge in unforced errors under deficit pressure.

### 6.2 Q2: Game Phase Dynamics (Before 15 vs. After 15)
- **Two-sided Mann-Whitney U Test:**
  - $U = 1,495,430.5$, **$p = 0.0293$**, rank-biserial $r = -0.045$.
  - Median shifts from **8.0 strokes** (Before 15) to **7.0 strokes** (After 15).
  - *Match-Level Clustered Robustness Check (Wilcoxon Signed-Rank Test across 44 matches):*
    - Mean within-match difference: **$-0.73$ strokes**, $W = 254.5, p = 0.0580$.
  - *Finding:* Late-game rallies exhibit a statistically significant reduction in duration, reflecting accelerated pace, heightened attacking urgency, and risk-taking.

### 6.3 Q3: Unforced Errors After 15 Points
- **Proportions Comparison:**
  - Before 15 points: $1,608 / 2,420 = \mathbf{66.45\%}$
  - After 15 points: $660 / 1,037 = \mathbf{63.65\%}$
  - $\chi^2 = 2.402$, $\text{dof} = 1$, **$p = 0.1212$** (Odds Ratio = $1.131$).
  - *Finding:* Unforced error rates do **not** significantly increase after 15 points. At elite international levels, errors remain stable while decisive winners account for a slightly higher proportion of late-game closures.

### 6.4 Q4: Men's vs. Women's Singles
- **Rally Length (Mann-Whitney U):**
  - Men's Singles (Med=8.0, Mean=10.30) vs. Women's Singles (Med=7.0, Mean=8.55).
  - $U = 1,301,502.5$, **$p = 1.059 \times 10^{-5}$**, rank-biserial $r = -0.100$.
  - *Finding:* Men's singles rallies are significantly longer than women's singles rallies in this elite sample.
- **Unforced Error Rate (Chi-Square):**
  - Men's Singles: $1,722 / 2,643 = 65.15\%$
  - Women's Singles: $546 / 814 = 67.08\%$
  - $\chi^2 = 0.936$, $p = 0.3332$. No significant difference between genders.

---

## 7. Limitations

1. **Unrecorded Rally Terminations:** 175 rallies (4.8%) lacked camera-verified outcome classifications and were excluded from error-rate denominators.
2. **Observational Confounding:** Deficit pressure correlates with fatigue in prolonged three-set matches.
3. **Singles Exclusivity:** Findings apply strictly to elite singles play and cannot be generalized to doubles pairings.

---

## 8. Conclusion

This study provides empirical, shot-level confirmation of scoreboard pressure dynamics in elite badminton:
1. **Deficit pressure drives unforced errors:** Trailing players commit significantly more unforced errors ($33.62\%$ vs $27.58\%$, $p < 0.0001$).
2. **Late-game pacing accelerates:** Rallies shorten after 15 points (median drops from 8.0 to 7.0 strokes, $p = 0.029$).
3. **Closing-stage error stability:** High error rates are sustained throughout games rather than peaking solely after 15 points.
4. **Gender differences:** Rally length is significantly greater in Men's Singles ($p < 0.0001$), but unforced error proportions remain virtually identical across genders ($p = 0.333$).

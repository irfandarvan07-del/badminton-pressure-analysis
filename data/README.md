# Dataset Documentation: ShuttleSet

## Source Information
- **Dataset Name:** ShuttleSet (A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis)
- **Publication:** ACM SIGKDD 2023 (`https://arxiv.org/abs/2306.04948`)
- **Repository:** [CoachAI-Projects (wywyWang)](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)
- **Domain:** Elite international singles badminton matches (BWF World Tour, All England, Indonesia Masters, Sudirman Cup, etc., 2018–2021).

## Coverage Summary
- **Matches:** 44 matches
  - Men's Singles (MS): 33 matches
  - Women's Singles (WS): 11 matches
- **Sets:** 104 played sets
- **Rallies:** 3,683 rallies
- **Shots (strokes):** 36,484 annotated shots
- **Unique Players:** 27 elite international singles players (e.g., Kento Momota, Viktor Axelsen, Chou Tien Chen, Anthony Sinisuka Ginting, An Se Young, Carolina Marin, Tai Tzu Ying opponents, Pusarla V. Sindhu, Ratchanok Intanon).

## File Layout
```
data/raw/shuttleset/
├── match.csv               # 44 rows of match metadata (players, tournament, date, duration, winner, loser)
├── homography.csv          # Camera homography calibration matrices
└── <Match_Folder_Name>/     # 44 folders, each corresponding to a match
    ├── set1.csv            # Stroke-by-stroke records for set 1
    ├── set2.csv            # Stroke-by-stroke records for set 2
    └── set3.csv            # Stroke-by-stroke records for set 3 (if played)
```

## Field Descriptions & Conventions
1. **Player Mapping:**
   - In each match, `Player A` designates the overall match winner (from `match.csv` `winner`).
   - `Player B` designates the overall match loser (from `match.csv` `loser`).
2. **Score Timing:**
   - Raw columns `roundscore_A` and `roundscore_B` represent the **post-rally score** (after the point is scored).
   - Pre-rally score (score at the start of the rally) is reconstructed for accurate psychological pressure analysis without lookahead bias.
3. **Rally Length:**
   - Number of strokes in a rally = count of stroke rows for that rally.
4. **Rally Endings and Unforced Errors:**
   - `lose_reason` and `win_reason` record rally-ending outcomes in Traditional Chinese:
     - `出界` (Out of bounds) -> **Unforced error**
     - `掛網` / `未過網` (Net fault / failed to cross net) -> **Unforced error**
     - `犯規` (Fault) -> **Fault error**
     - `對手落地致勝` / `落地致勝` (Opponent winner / clean landing winner) -> **Forced winner**
     - `落點判斷失誤` (Landing judgment mistake) -> Judged error
     - Missing / None -> Unclassifiable ending


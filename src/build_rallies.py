"""
Rally-level data construction module.
Transforms shot-level sequence into structured rally-level records with pre-rally
score states, game phases, stroke lengths, and unforced error attributions.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.score_state import classify_score_state, classify_game_phase
from src.clean_data import clean_and_validate_dataset

def build_rally_records(
    cleaned_shots_df: pd.DataFrame = None,
    phase_threshold: int = config.GAME_PHASE_THRESHOLD,
    save_to_disk: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs rally-level records from cleaned shot data.
    
    Returns:
        - canonical_rallies_df: 1 row per rally (3,683 rallies).
        - player_perspective_df: 2 rows per rally (evaluated from both player perspectives).
    """
    if cleaned_shots_df is None:
        shots_csv = os.path.join(config.PROCESSED_DATA_DIR, "cleaned_shots.csv")
        if os.path.exists(shots_csv):
            cleaned_shots_df = pd.read_csv(shots_csv)
        else:
            cleaned_shots_df, _ = clean_and_validate_dataset()

    canonical_rows = []
    perspective_rows = []

    # Group by game (match_id and set_num)
    for (mid, sid), game_shots in cleaned_shots_df.groupby(["match_id", "set_num"], sort=True):
        rallies_in_game = sorted(game_shots["rally"].unique())
        
        # Track running pre-scores
        prev_post_A = 0
        prev_post_B = 0

        for r_id in rallies_in_game:
            r_shots = game_shots[game_shots["rally"] == r_id].sort_values("ball_round")
            first_shot = r_shots.iloc[0]
            last_shot = r_shots.iloc[-1]

            # Pre-rally scores
            pre_A = prev_post_A
            pre_B = prev_post_B

            # Post-rally scores
            post_A = int(last_shot["roundscore_A"])
            post_B = int(last_shot["roundscore_B"])
            prev_post_A, prev_post_B = post_A, post_B

            # Classify score states
            state_A = classify_score_state(pre_A, pre_B)
            state_B = classify_score_state(pre_B, pre_A)
            leader_score = max(pre_A, pre_B)
            phase = classify_game_phase(leader_score, threshold=phase_threshold)

            # Rally stroke count
            rally_length = len(r_shots)

            # Winner / loser
            winner_tag = last_shot["getpoint_player"]
            if pd.isna(winner_tag) or winner_tag not in ["A", "B"]:
                winner_tag = None
                winner_name = "Unknown"
                loser_tag = None
                loser_name = "Unknown"
            elif winner_tag == "A":
                winner_name = first_shot["player_A_name"]
                loser_tag = "B"
                loser_name = first_shot["player_B_name"]
            else:
                winner_name = first_shot["player_B_name"]
                loser_tag = "A"
                loser_name = first_shot["player_A_name"]

            # Ending type and unforced error classification
            lose_reason_en = last_shot["lose_reason_en"]
            if lose_reason_en in ["none", "unclassified"]:
                ending_type = "unknown"
                unforced_error = None
                error_player_name = None
                error_score_state = None
            elif lose_reason_en in config.UNFORCED_ERROR_TYPES:
                ending_type = lose_reason_en
                unforced_error = True
                error_player_name = loser_name
                error_score_state = state_B if loser_tag == "B" else state_A
            elif lose_reason_en == "opponent_winner":
                ending_type = "winner"
                unforced_error = False
                error_player_name = None
                error_score_state = None
            else:
                ending_type = lose_reason_en
                # Other errors like judgment error are recorded as non-strict
                unforced_error = False
                error_player_name = None
                error_score_state = None

            # Canonical rally record
            c_row = {
                "match_id": int(mid),
                "game_id": f"{mid}_S{sid}",
                "set_num": int(sid),
                "rally_id": int(r_id),
                "tournament": first_shot["tournament"],
                "category": first_shot["category"],
                "round_stage": first_shot["round_stage"],
                "year": first_shot["year"],
                "player_A": first_shot["player_A_name"],
                "player_B": first_shot["player_B_name"],
                "pre_score_A": pre_A,
                "pre_score_B": pre_B,
                "post_score_A": post_A,
                "post_score_B": post_B,
                "score_diff_A": pre_A - pre_B,
                "score_diff_B": pre_B - pre_A,
                "score_state_A": state_A,
                "score_state_B": state_B,
                "leader_score": leader_score,
                "game_phase": phase,
                "rally_length": rally_length,
                "rally_winner": winner_name,
                "rally_winner_tag": winner_tag,
                "rally_loser": loser_name,
                "rally_loser_tag": loser_tag,
                "ending_type": ending_type,
                "unforced_error": unforced_error,
                "error_player": error_player_name,
                "error_score_state": error_score_state,
                "last_shot_type": last_shot["shot_type_en"]
            }
            canonical_rows.append(c_row)

            # Player perspective rows (one for A, one for B)
            # Row for Player A
            is_win_A = (winner_tag == "A") if winner_tag else None
            is_ue_A = (unforced_error is True and loser_tag == "A") if unforced_error is not None else None
            p_row_A = {
                "match_id": int(mid),
                "game_id": f"{mid}_S{sid}",
                "rally_id": int(r_id),
                "category": first_shot["category"],
                "player_id": first_shot["player_A_name"],
                "opponent_id": first_shot["player_B_name"],
                "player_tag": "A",
                "player_score": pre_A,
                "opponent_score": pre_B,
                "score_diff": pre_A - pre_B,
                "score_state": state_A,
                "game_phase": phase,
                "rally_length": rally_length,
                "rally_winner": winner_name,
                "is_winner": is_win_A,
                "ending_type": ending_type,
                "unforced_error": is_ue_A,
                "classifiable_ending": (ending_type != "unknown")
            }
            perspective_rows.append(p_row_A)

            # Row for Player B
            is_win_B = (winner_tag == "B") if winner_tag else None
            is_ue_B = (unforced_error is True and loser_tag == "B") if unforced_error is not None else None
            p_row_B = {
                "match_id": int(mid),
                "game_id": f"{mid}_S{sid}",
                "rally_id": int(r_id),
                "category": first_shot["category"],
                "player_id": first_shot["player_B_name"],
                "opponent_id": first_shot["player_A_name"],
                "player_tag": "B",
                "player_score": pre_B,
                "opponent_score": pre_A,
                "score_diff": pre_B - pre_A,
                "score_state": state_B,
                "game_phase": phase,
                "rally_length": rally_length,
                "rally_winner": winner_name,
                "is_winner": is_win_B,
                "ending_type": ending_type,
                "unforced_error": is_ue_B,
                "classifiable_ending": (ending_type != "unknown")
            }
            perspective_rows.append(p_row_B)

    canonical_df = pd.DataFrame(canonical_rows)
    perspective_df = pd.DataFrame(perspective_rows)

    # Save to data/processed if requested
    if save_to_disk:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        canonical_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies.csv"), index=False)
        perspective_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "rallies_player_perspective.csv"), index=False)

    return canonical_df, perspective_df

if __name__ == "__main__":
    c_df, p_df = build_rally_records()
    print(f"Built {len(c_df)} canonical rally records.")
    print(f"Built {len(p_df)} player-perspective rally records.")
    print("\nScore State Distribution (Canonical A):")
    print(c_df["score_state_A"].value_counts())
    print("\nGame Phase Distribution:")
    print(c_df["game_phase"].value_counts())
    print("\nUnforced Error Distribution:")
    print(c_df["unforced_error"].value_counts(dropna=False))

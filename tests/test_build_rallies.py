"""
Unit tests for rally construction and stroke counting.
"""

import pytest
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.build_rallies import build_rally_records

def test_build_rally_records_mock():
    # Construct small mock shots dataframe
    mock_shots = pd.DataFrame([
        # Game 1, Rally 1 (3 shots, A wins via out by B)
        {"match_id": 1, "set_num": 1, "rally": 1, "ball_round": 1, "player_A_name": "Player 1", "player_B_name": "Player 2",
         "roundscore_A": 1, "roundscore_B": 0, "getpoint_player": "A", "lose_reason_en": "out", "tournament": "Test", "category": "Men's Singles", "round_stage": "Final", "year": 2021, "shot_type_en": "service"},
        {"match_id": 1, "set_num": 1, "rally": 1, "ball_round": 2, "player_A_name": "Player 1", "player_B_name": "Player 2",
         "roundscore_A": 1, "roundscore_B": 0, "getpoint_player": "A", "lose_reason_en": "out", "tournament": "Test", "category": "Men's Singles", "round_stage": "Final", "year": 2021, "shot_type_en": "clear"},
        {"match_id": 1, "set_num": 1, "rally": 1, "ball_round": 3, "player_A_name": "Player 1", "player_B_name": "Player 2",
         "roundscore_A": 1, "roundscore_B": 0, "getpoint_player": "A", "lose_reason_en": "out", "tournament": "Test", "category": "Men's Singles", "round_stage": "Final", "year": 2021, "shot_type_en": "smash"},
        
        # Game 1, Rally 2 (2 shots, B wins via opponent_winner)
        {"match_id": 1, "set_num": 1, "rally": 2, "ball_round": 1, "player_A_name": "Player 1", "player_B_name": "Player 2",
         "roundscore_A": 1, "roundscore_B": 1, "getpoint_player": "B", "lose_reason_en": "opponent_winner", "tournament": "Test", "category": "Men's Singles", "round_stage": "Final", "year": 2021, "shot_type_en": "service"},
        {"match_id": 1, "set_num": 1, "rally": 2, "ball_round": 2, "player_A_name": "Player 1", "player_B_name": "Player 2",
         "roundscore_A": 1, "roundscore_B": 1, "getpoint_player": "B", "lose_reason_en": "opponent_winner", "tournament": "Test", "category": "Men's Singles", "round_stage": "Final", "year": 2021, "shot_type_en": "drop"},
    ])

    c_df, p_df = build_rally_records(mock_shots, phase_threshold=15, save_to_disk=False)
    
    # Check Canonical DF
    assert len(c_df) == 2
    r1 = c_df.iloc[0]
    assert r1["rally_length"] == 3
    assert r1["pre_score_A"] == 0
    assert r1["pre_score_B"] == 0
    assert r1["score_state_A"] == "Level"
    assert r1["game_phase"] == "Before 15"
    assert bool(r1["unforced_error"]) is True
    assert r1["error_player"] == "Player 2"
    assert r1["error_score_state"] == "Level"

    r2 = c_df.iloc[1]
    assert r2["rally_length"] == 2
    assert r2["pre_score_A"] == 1
    assert r2["pre_score_B"] == 0
    assert r2["score_state_A"] == "Leading"
    assert r2["score_state_B"] == "Trailing"
    assert bool(r2["unforced_error"]) is False
    assert r2["rally_winner"] == "Player 2"

    # Check Perspective DF
    assert len(p_df) == 4
    p_r2_B = p_df[(p_df["rally_id"] == 2) & (p_df["player_tag"] == "B")].iloc[0]
    assert p_r2_B["player_score"] == 0
    assert p_r2_B["opponent_score"] == 1
    assert p_r2_B["score_state"] == "Trailing"
    assert bool(p_r2_B["is_winner"]) is True

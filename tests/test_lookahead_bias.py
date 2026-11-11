"""
tests/test_lookahead_bias.py
P0.1 — Prove pre-rally scores are used (never post-rally roundscore_A/B).

Hand-built mini set with known score progression:
  Rally 1: (0,0) → A wins  → post (1,0)
  Rally 2: (1,0) → B wins  → post (1,1)
  Rally 3: (1,1) → A wins  → post (2,1)   phase: Before 15
  Rally 4: (2,1) → A wins  → post (3,1)
  ...
  Rally 15: (14,1) → A wins → post (15,1)  phase transition HERE
  Rally 16: (15,1) → A wins → post (16,1)  phase: After 15 (leader=15)
"""

import pytest
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.build_rallies import build_rally_records


def _make_shot(match_id, set_num, rally, ball_round,
               post_A, post_B, getpoint, lose_reason):
    """Helper: one shot row."""
    return {
        "match_id": match_id, "set_num": set_num, "rally": rally,
        "ball_round": ball_round,
        "player_A_name": "Alpha", "player_B_name": "Beta",
        "roundscore_A": post_A, "roundscore_B": post_B,
        "getpoint_player": getpoint, "lose_reason_en": lose_reason,
        "tournament": "Test", "category": "Men's Singles",
        "round_stage": "Final", "year": 2021, "shot_type_en": "smash",
    }


def _build_mini_set():
    """
    Build a 17-rally sequence where A wins almost every rally.
    Rally 1-15: leader stays below 15 at rally 15 start (pre=14,0).
    Rally 15 start: pre=(14,0) → leader=14 → Before 15
    Rally 15 end: post=(15,0)
    Rally 16 start: pre=(15,0) → leader=15 → After 15   ← transition
    Rally 17 start: pre=(16,0) → leader=16 → After 15
    """
    rows = []
    post_A, post_B = 0, 0
    for rally_num in range(1, 18):
        # A wins every rally (B commits an out)
        new_A = post_A + 1
        new_B = post_B
        rows.append(_make_shot(1, 1, rally_num, 1,
                               new_A, new_B, "A", "out"))
        post_A, post_B = new_A, new_B
    return pd.DataFrame(rows)


class TestNoLookaheadBias:
    """Every assertion uses PRE-rally scores, not roundscore_A/B."""

    def test_first_rally_always_starts_at_zero_zero(self):
        shots = _build_mini_set()
        c_df, _ = build_rally_records(shots, phase_threshold=15, save_to_disk=False)
        r1 = c_df[c_df["rally_id"] == 1].iloc[0]
        assert r1["pre_score_A"] == 0, "Rally 1 must start at 0-0 for A"
        assert r1["pre_score_B"] == 0, "Rally 1 must start at 0-0 for B"

    def test_pre_score_equals_previous_post_score(self):
        shots = _build_mini_set()
        c_df, _ = build_rally_records(shots, phase_threshold=15, save_to_disk=False)
        c_df = c_df.sort_values("rally_id").reset_index(drop=True)
        for i in range(1, len(c_df)):
            prev = c_df.iloc[i - 1]
            curr = c_df.iloc[i]
            assert curr["pre_score_A"] == prev["post_score_A"], (
                f"Rally {curr['rally_id']}: pre_A={curr['pre_score_A']} "
                f"≠ prev post_A={prev['post_score_A']}"
            )
            assert curr["pre_score_B"] == prev["post_score_B"], (
                f"Rally {curr['rally_id']}: pre_B={curr['pre_score_B']} "
                f"≠ prev post_B={prev['post_score_B']}"
            )

    def test_phase_transition_uses_pre_score_not_post_score(self):
        """
        Rally 15: pre=(14,0) → leader_pre=14 → Before 15 (threshold=15)
        Rally 16: pre=(15,0) → leader_pre=15 → After 15
        The phase must NOT be classified from the POST scores.
        """
        shots = _build_mini_set()
        c_df, _ = build_rally_records(shots, phase_threshold=15, save_to_disk=False)
        c_df = c_df.sort_values("rally_id").reset_index(drop=True)

        r15 = c_df[c_df["rally_id"] == 15].iloc[0]
        r16 = c_df[c_df["rally_id"] == 16].iloc[0]

        # Pre-rally check
        assert r15["pre_score_A"] == 14, f"Rally 15 pre_A should be 14, got {r15['pre_score_A']}"
        assert r16["pre_score_A"] == 15, f"Rally 16 pre_A should be 15, got {r16['pre_score_A']}"

        # Phase check — CRITICAL: must use PRE score, not POST score
        assert r15["game_phase"] == "Before 15", (
            f"Rally 15 (pre=14) must be 'Before 15', got '{r15['game_phase']}'. "
            "If this fails, POST scores were used (lookahead bias)."
        )
        assert r16["game_phase"] == "After 15", (
            f"Rally 16 (pre=15) must be 'After 15', got '{r16['game_phase']}'. "
            "If this fails, PRE scores were used incorrectly."
        )

    def test_score_state_uses_pre_score(self):
        """
        Rally 2: pre=(1,0) → A is Leading, B is Trailing.
        If we mistakenly used POST=(1,1), both would be Level. Catch that.
        """
        shots = _build_mini_set()
        c_df, _ = build_rally_records(shots, phase_threshold=15, save_to_disk=False)
        r2 = c_df[c_df["rally_id"] == 2].iloc[0]
        assert r2["pre_score_A"] == 1
        assert r2["pre_score_B"] == 0
        assert r2["score_state_A"] == "Leading", (
            "Rally 2 (pre 1-0): A must be 'Leading'. "
            "If 'Level', POST scores (1-1) were used — lookahead bias!"
        )
        assert r2["score_state_B"] == "Trailing"

    def test_player_perspective_pre_scores_match_canonical(self):
        """Player perspective rows must have the same pre scores as canonical."""
        shots = _build_mini_set()
        c_df, p_df = build_rally_records(shots, phase_threshold=15, save_to_disk=False)
        for rally_id in c_df["rally_id"].unique():
            canon = c_df[c_df["rally_id"] == rally_id].iloc[0]
            p_A = p_df[(p_df["rally_id"] == rally_id) & (p_df["player_tag"] == "A")].iloc[0]
            p_B = p_df[(p_df["rally_id"] == rally_id) & (p_df["player_tag"] == "B")].iloc[0]
            assert p_A["player_score"] == canon["pre_score_A"]
            assert p_B["player_score"] == canon["pre_score_B"]
            assert p_A["game_phase"] == canon["game_phase"]
            assert p_B["game_phase"] == canon["game_phase"]


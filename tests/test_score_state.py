"""
Unit tests for score state and game phase classification logic.
"""

import pytest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.score_state import classify_score_state, classify_game_phase, get_rally_score_context

def test_classify_score_state_leading():
    assert classify_score_state(10, 5) == "Leading"
    assert classify_score_state(1, 0) == "Leading"
    assert classify_score_state(21, 20) == "Leading"

def test_classify_score_state_level():
    assert classify_score_state(0, 0) == "Level"
    assert classify_score_state(14, 14) == "Level"
    assert classify_score_state(20, 20) == "Level"

def test_classify_score_state_trailing():
    assert classify_score_state(5, 10) == "Trailing"
    assert classify_score_state(0, 1) == "Trailing"
    assert classify_score_state(18, 19) == "Trailing"

def test_classify_score_state_invalid_inputs():
    with pytest.raises(ValueError):
        classify_score_state(-1, 5)
    with pytest.raises(ValueError):
        classify_score_state(5, -2)
    with pytest.raises(ValueError):
        classify_score_state(None, 5)
    with pytest.raises(ValueError):
        classify_score_state("invalid", 5)
    with pytest.raises(ValueError):
        classify_score_state(float('nan'), 5)

def test_classify_game_phase():
    assert classify_game_phase(0, 15) == "Before 15"
    assert classify_game_phase(14, 15) == "Before 15"
    assert classify_game_phase(15, 15) == "After 15"
    assert classify_game_phase(16, 15) == "After 15"
    assert classify_game_phase(20, 15) == "After 15"

def test_classify_game_phase_invalid():
    with pytest.raises(ValueError):
        classify_game_phase(-1)
    with pytest.raises(ValueError):
        classify_game_phase(None)
    with pytest.raises(ValueError):
        classify_game_phase("ten")

def test_rally_score_context():
    ctx = get_rally_score_context(16, 12, threshold=15)
    assert ctx["pre_score_A"] == 16
    assert ctx["pre_score_B"] == 12
    assert ctx["score_diff_A"] == 4
    assert ctx["score_diff_B"] == -4
    assert ctx["score_state_A"] == "Leading"
    assert ctx["score_state_B"] == "Trailing"
    assert ctx["leader_score"] == 16
    assert ctx["game_phase"] == "After 15"

    ctx_early = get_rally_score_context(0, 0, threshold=15)
    assert ctx_early["score_state_A"] == "Level"
    assert ctx_early["score_state_B"] == "Level"
    assert ctx_early["game_phase"] == "Before 15"


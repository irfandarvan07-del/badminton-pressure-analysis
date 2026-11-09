"""
Unit tests for metric calculation functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics import calculate_rate, summarize_numeric_series, compute_unforced_error_metrics

def test_calculate_rate_standard():
    assert calculate_rate(5, 10) == 0.5
    assert calculate_rate(1, 3, round_digits=2) == 0.33
    assert calculate_rate(0, 50) == 0.0

def test_calculate_rate_zero_and_negative_denominators():
    # Safeguard tests required by Section 9
    assert calculate_rate(5, 0) == 0.0
    assert calculate_rate(5, -1) == 0.0
    assert calculate_rate(5, None) == 0.0
    assert calculate_rate(-2, 10) == 0.0

def test_summarize_numeric_series():
    s = pd.Series([2, 4, 6, 8, 10])
    res = summarize_numeric_series(s)
    assert res["count"] == 5
    assert res["mean"] == 6.0
    assert res["median"] == 6.0
    assert res["min"] == 2
    assert res["max"] == 10
    assert res["p25"] == 4.0
    assert res["p75"] == 8.0
    assert res["iqr"] == 4.0

def test_summarize_empty_series():
    empty_s = pd.Series([], dtype=float)
    res = summarize_numeric_series(empty_s)
    assert res["count"] == 0
    assert res["median"] == 0.0

def test_unforced_error_metrics_calculation():
    mock_df = pd.DataFrame([
        {"ending_type": "out", "unforced_error": True, "game_phase": "Before 15", "category": "Men's Singles"},
        {"ending_type": "net", "unforced_error": True, "game_phase": "Before 15", "category": "Men's Singles"},
        {"ending_type": "winner", "unforced_error": False, "game_phase": "After 15", "category": "Men's Singles"},
        {"ending_type": "unknown", "unforced_error": None, "game_phase": "After 15", "category": "Men's Singles"},
    ])

    res = compute_unforced_error_metrics(mock_df)
    assert res["total_rallies"] == 4
    assert res["classifiable_rallies"] == 3
    assert res["unforced_errors"] == 2
    assert res["clean_winners"] == 1
    assert res["unclassified_endings"] == 1
    # 2 UEs out of 3 classifiable = 0.6667
    assert res["overall_unforced_error_rate_classifiable"] == 0.6667

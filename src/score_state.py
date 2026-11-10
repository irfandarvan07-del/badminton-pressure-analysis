"""
Score State and Game Phase classification module.
Provides consistent, deterministic classifications for player pressure analysis.
"""

from typing import Union, Tuple

def classify_score_state(player_score: Union[int, float], opponent_score: Union[int, float]) -> str:
    """
    Classifies player's score state relative to opponent.
    
    Returns:
        - "Leading" if player_score > opponent_score
        - "Level" if player_score == opponent_score
        - "Trailing" if player_score < opponent_score
        
    Raises:
        ValueError if either score is None, NaN, or negative.
    """
    if player_score is None or opponent_score is None:
        raise ValueError("Scores cannot be None")
    
    try:
        p_val = float(player_score)
        o_val = float(opponent_score)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid score values: player={player_score}, opponent={opponent_score}")

    import math
    if math.isnan(p_val) or math.isnan(o_val):
        raise ValueError("Scores cannot be NaN")

    if p_val < 0 or o_val < 0:
        raise ValueError(f"Scores cannot be negative: player={p_val}, opponent={o_val}")

    if p_val > o_val:
        return "Leading"
    elif p_val < o_val:
        return "Trailing"
    else:
        return "Level"

def classify_game_phase(reference_score: Union[int, float], threshold: int = 15) -> str:
    """
    Divides game into "Before 15" (< threshold) and "After 15" (>= threshold).
    The threshold is configurable (default 15).
    """
    if reference_score is None:
        raise ValueError("Score cannot be None")
    
    try:
        val = float(reference_score)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid reference score: {reference_score}")

    import math
    if math.isnan(val) or val < 0:
        raise ValueError(f"Score must be non-negative and non-NaN: {val}")

    if val < threshold:
        return f"Before {threshold}"
    else:
        return f"After {threshold}"

def get_rally_score_context(
    pre_score_A: int,
    pre_score_B: int,
    threshold: int = 15
) -> dict:
    """
    Generates complete score context from pre-rally scores for both players.
    Uses the leading score (max of both players) to determine game phase.
    """
    state_A = classify_score_state(pre_score_A, pre_score_B)
    state_B = classify_score_state(pre_score_B, pre_score_A)
    leader_score = max(pre_score_A, pre_score_B)
    phase = classify_game_phase(leader_score, threshold=threshold)

    return {
        "pre_score_A": int(pre_score_A),
        "pre_score_B": int(pre_score_B),
        "score_diff_A": int(pre_score_A - pre_score_B),
        "score_diff_B": int(pre_score_B - pre_score_A),
        "score_state_A": state_A,
        "score_state_B": state_B,
        "leader_score": int(leader_score),
        "game_phase": phase,
    }


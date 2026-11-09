"""
Configuration parameters for Badminton Pressure Analysis.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "shuttleset")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

# Analysis parameters
GAME_PHASE_THRESHOLD = 15  # Score >= 15 is "After 15 points"

# Score state timing convention:
# "start_of_rally": player's score entering the rally (psychologically sound, avoids lookahead bias)
# "end_of_rally": player's score after point is awarded
DEFAULT_SCORE_TIMING = "start_of_rally"

# Unforced error definitions
# Mappings from ShuttleSet Chinese labels to standardized English terms
REASON_TRANSLATIONS = {
    "出界": "out",
    "掛網": "net",
    "未過網": "net",
    "犯規": "fault",
    "對手落地致勝": "opponent_winner",
    "落地致勝": "winner",
    "落點判斷失誤": "judgment_error",
    "對手落地判斷失誤": "opponent_judgment_error",
    "對手出界": "opponent_out",
    "對手掛網": "opponent_net",
    "對手未過網": "opponent_net",
    "對手犯規": "opponent_fault",
}

# Strict unforced error types: net, out, service fault
UNFORCED_ERROR_TYPES = {"out", "net", "fault"}

# Shot type translation mapping (ShuttleSet 18 types)
SHOT_TYPE_TRANSLATIONS = {
    "發短球": "short service",
    "發長球": "long service",
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過渡切球": "passive drop",
    "過度切球": "passive drop",  # alternate character variant
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "未知球種": "unknown",
}

# Elite Women's Singles players in ShuttleSet
WOMEN_SINGLES_PLAYERS = {
    "An Se Young",
    "Busanan ONGBAMRUNGPHAN",
    "Carolina MARIN",
    "Evgeniya KOSETSKAYA",
    "Mia BLICHFELDT",
    "Michelle LI",
    "Neslihan YIGIT",
    "PUSARLA V. Sindhu",
    "Pornpawee CHOCHUWONG",
    "Ratchanok INTANON",
    "Supanida KATETHONG",
}


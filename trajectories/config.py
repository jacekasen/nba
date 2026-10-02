"""Configuration constants, file paths, and season definitions for trajectories."""

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trajectories"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
PARQUET_OUTPUT_PATH = NORMALIZED_DIR / "player_game_logs.parquet"
CSV_OUTPUT_PATH = NORMALIZED_DIR / "player_game_logs.csv"
VALIDATION_REPORT_PATH = NORMALIZED_DIR / "validation_report.json"
TEAM_SCHEDULES_PATH = NORMALIZED_DIR / "team_schedules.json"

# All seasons in the official play-by-play era (1996-97 through 2025-26)
ALL_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2026)]

# Shortened/lockout seasons and their expected max team games
SHORTENED_SEASONS = {
    "1998-99": {"expected_max": 50, "min_required": 50},
    "2011-12": {"expected_max": 66, "min_required": 66},
    "2019-20": {"expected_max": 75, "min_required": 71},  # Bubble teams played 71-75; non-bubble teams 64-67
    "2020-21": {"expected_max": 72, "min_required": 72},
}

# Standard regular season expected games
STANDARD_SEASON_GAMES = 82
STANDARD_SEASON_MIN_REQUIRED = 81  # 2012-13 Boston/Indiana played 81 due to marathon cancellation

# Query all calendar months (1-12) to catch summer restarts (e.g. 2020 bubble in July/August: months 10 and 11)
REGULAR_SEASON_MONTHS = list(range(1, 13))

# Database table
SUPABASE_TABLE = "player_game_logs"

# HTTP collection settings
STATS_HEADERS = {
    "Host": "stats.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
CURL_IMPERSONATE = "chrome120"
REQUEST_DELAY_SECONDS = 1.2
REQUEST_TIMEOUT_SECONDS = 25
MAX_RETRIES = 4

"""Paths and constants for the salary analysis pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Cached Basketball-Reference HTML (regenerable).
RAW_SALARIES_DIR = DATA_DIR / "raw-salaries"
PLAYER_HTML_DIR = RAW_SALARIES_DIR / "players"
CAP_HTML_PATH = RAW_SALARIES_DIR / "salary_cap_history.html"

# Parsed salary intermediates (regenerable).
INTERMEDIATE_SALARIES_DIR = DATA_DIR / "salaries"
RAW_PLAYER_SALARIES_PATH = INTERMEDIATE_SALARIES_DIR / "raw_player_salaries.csv"
SALARY_CAPS_PATH = INTERMEDIATE_SALARIES_DIR / "salary_caps.csv"
SCRAPE_PROGRESS_PATH = INTERMEDIATE_SALARIES_DIR / "scrape_progress.json"
SCRAPE_FAILURES_PATH = INTERMEDIATE_SALARIES_DIR / "scrape_failures.csv"

MODELING_DIR = DATA_DIR / "modeling"
PLAYER_SALARIES_PATH = MODELING_DIR / "player_salaries.csv"
TEAM_SEASON_SALARIES_PATH = MODELING_DIR / "team_season_salaries.csv"
PLAYER_SALARY_HISTORY_PATH = MODELING_DIR / "player_salary_history.csv"
VALIDATION_REPORT_PATH = MODELING_DIR / "salary_validation_report.json"

PLAYER_URLS_PATH = DATA_DIR / "player_pages.csv"
RAW_STATS_PATH = DATA_DIR / "nba_player_stats.csv"

BASE_URL = "https://www.basketball-reference.com"
SALARY_CAP_URL = f"{BASE_URL}/contracts/salary-cap-history.html"

USER_AGENT = (
    "BoardManGetsPaidResearchBot/0.1 "
    "(+https://github.com/jacekasen/nba; educational NBA research; "
    "occasional offline ingestion only; never used for live website requests)"
)

# Basketball-Reference publicly documents ~20 requests/minute; stay well under that.
DEFAULT_REQUEST_DELAY_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 8.0

# Cap-era coverage target.
DEFAULT_START_SEASON = "1984-85"

# Seasons historically known to have sparse or missing salary coverage on BBRef.
KNOWN_PROBLEMATIC_SEASONS = ("1986-87", "1989-90")

# Minimum known player-salary rows on a team-season before computing payroll share.
MIN_TEAM_ROWS_FOR_PAYROLL_SHARE = 8

SALARY_SOURCE = "basketball_reference_player_page"

# Representative validation sample: modern active, long career, 1990s star, 1980s career.
SAMPLE_PLAYERS: tuple[dict[str, str], ...] = (
    {
        "player_id": "curryst01",
        "player_name": "Stephen Curry",
        "player_url": f"{BASE_URL}/players/c/curryst01.html",
    },
    {
        "player_id": "jamesle01",
        "player_name": "LeBron James",
        "player_url": f"{BASE_URL}/players/j/jamesle01.html",
    },
    {
        "player_id": "jordami01",
        "player_name": "Michael Jordan",
        "player_url": f"{BASE_URL}/players/j/jordami01.html",
    },
    {
        "player_id": "birdla01",
        "player_name": "Larry Bird",
        "player_url": f"{BASE_URL}/players/b/birdla01.html",
    },
)

RAW_SALARY_COLUMNS = [
    "player_id",
    "player_name",
    "player_url",
    "season",
    "season_start",
    "season_end",
    "team",
    "team_name",
    "league",
    "salary",
    "salary_source",
    "salary_quality",
    "html_cache_path",
]

PLAYER_SALARY_COLUMNS = [
    "player_id",
    "player_name",
    "player_url",
    "season",
    "season_start",
    "season_end",
    "team",
    "salary",
    "salary_cap",
    "cap_share",
    "salary_source",
    "salary_quality",
    "team_payroll_share",
    "team_known_salary_total",
    "team_known_player_count",
]

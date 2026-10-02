"""Configuration constants and paths for the EPM sub-project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EPM_DATA_DIR = REPO_ROOT / "data" / "EPM"
SUPABASE_TABLE = "nba_player_epm"
CONFLICT_COLUMNS = ("nba_player_id", "season", "season_type")

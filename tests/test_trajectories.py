"""Unit tests for the trajectories data processing and rolling metric calculations."""

import pandas as pd
import pytest

from trajectories.normalize import (
    assign_team_game_numbers,
    compute_player_trajectories,
    parse_matchup,
    parse_minutes,
)


def test_parse_minutes():
    assert parse_minutes("31:30") == 31.5
    assert parse_minutes("0:45") == 0.75
    assert parse_minutes(32.456) == 32.46
    assert parse_minutes(None) == 0.0
    assert parse_minutes("") == 0.0


def test_parse_matchup():
    is_home, opp = parse_matchup("TOR vs. BKN")
    assert is_home is True
    assert opp == "BKN"

    is_home, opp = parse_matchup("NOP @ MIN")
    assert is_home is False
    assert opp == "MIN"

    is_home, opp = parse_matchup("")
    assert is_home is False
    assert opp == ""


def test_assign_team_game_numbers():
    df = pd.DataFrame(
        [
            {"SEASON_YEAR": "2025-26", "TEAM_ABBREVIATION": "DAL", "GAME_ID": "G1", "GAME_DATE": "2025-10-24"},
            {"SEASON_YEAR": "2025-26", "TEAM_ABBREVIATION": "DAL", "GAME_ID": "G2", "GAME_DATE": "2025-10-26"},
            {"SEASON_YEAR": "2025-26", "TEAM_ABBREVIATION": "DAL", "GAME_ID": "G3", "GAME_DATE": "2025-10-28"},
            {"SEASON_YEAR": "2025-26", "TEAM_ABBREVIATION": "HOU", "GAME_ID": "G10", "GAME_DATE": "2025-10-25"},
        ]
    )

    res_df, schedules = assign_team_game_numbers(df)
    dal_rows = res_df[res_df["TEAM_ABBREVIATION"] == "DAL"].sort_values("team_game_number")
    assert list(dal_rows["team_game_number"]) == [1, 2, 3]

    hou_rows = res_df[res_df["TEAM_ABBREVIATION"] == "HOU"]
    assert list(hou_rows["team_game_number"]) == [1]


def test_compute_player_trajectories():
    df = pd.DataFrame(
        [
            {
                "PLAYER_ID": 101,
                "SEASON_YEAR": "2025-26",
                "game_date": "2025-10-24",
                "GAME_ID": "G1",
                "TEAM_ABBREVIATION": "DAL",
                "plus_minus": 10,
                "minutes": 30.0,
            },
            {
                "PLAYER_ID": 101,
                "SEASON_YEAR": "2025-26",
                "game_date": "2025-10-26",
                "GAME_ID": "G2",
                "TEAM_ABBREVIATION": "DAL",
                "plus_minus": -4,
                "minutes": 20.0,
            },
            {
                "PLAYER_ID": 101,
                "SEASON_YEAR": "2025-26",
                "game_date": "2025-10-28",
                "GAME_ID": "G3",
                "TEAM_ABBREVIATION": "MIA",  # Traded to MIA
                "plus_minus": 6,
                "minutes": 10.0,
            },
        ]
    )

    res = compute_player_trajectories(df)
    assert list(res["player_season_game_number"]) == [1, 2, 3]
    # rolling_pm_5
    assert res.loc[0, "rolling_pm_5"] == 10.0
    assert res.loc[1, "rolling_pm_5"] == 3.0  # (10 + -4) / 2
    assert res.loc[2, "rolling_pm_5"] == 4.0  # (10 + -4 + 6) / 3

    # trade transition flag
    assert res.loc[0, "is_trade_transition"] == False
    assert res.loc[1, "is_trade_transition"] == False
    assert res.loc[2, "is_trade_transition"] == True  # Changed from DAL to MIA

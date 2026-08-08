from __future__ import annotations

import pandas as pd

from ml.data import collapse_player_seasons, derive_player_id, parse_year_id


def test_parse_year_id() -> None:
    season, start, end = parse_year_id("2024-25")
    assert season == "2024-25"
    assert start == 2024
    assert end == 2025


def test_collapse_prefers_ntm_row() -> None:
    df = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "player_url": "https://www.basketball-reference.com/players/a/alpha01.html",
                "season": "2020-21",
                "team_name_abbr": "BOS",
                "games": 20,
                "mp": 500,
                "bpm": 1.0,
                "per": 15.0,
                "ws": 2.0,
                "ws_per_48": 0.12,
            },
            {
                "player_name": "Player A",
                "player_url": "https://www.basketball-reference.com/players/a/alpha01.html",
                "season": "2020-21",
                "team_name_abbr": "2TM",
                "games": 40,
                "mp": 1000,
                "bpm": 1.4,
                "per": 16.0,
                "ws": 3.0,
                "ws_per_48": 0.14,
            },
        ]
    )

    out, qa = collapse_player_seasons(df)
    assert len(out) == 1
    assert out.iloc[0]["team_name_abbr"] == "2TM"
    assert qa["groups_using_ntm_row"] == 1


def test_stable_player_id_from_url() -> None:
    url = "https://www.basketball-reference.com/players/j/jamesle01.html"
    assert derive_player_id(url) == "jamesle01"

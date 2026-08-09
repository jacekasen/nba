"""Unit tests for salary HTML parsing and normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from salary.normalize import normalize_salaries
from salary.parsing import (
    classify_salary_quality,
    extract_player_name,
    parse_all_salaries_table,
    parse_money,
    parse_salary_cap_table,
)
from salary.scrape import parse_player_html
from salary.validate import validate_salary_dataset
from bs4 import BeautifulSoup

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "salaries"


def test_parse_money() -> None:
    assert parse_money("$2,710,560") == 2_710_560
    assert parse_money("") is None
    assert parse_money("-") is None


def test_parse_sample_player_salaries() -> None:
    expectations = {
        "curry.html": ("Stephen Curry", "2009-10", "GSW", 2_710_560),
        "james.html": ("LeBron James", "2003-04", "CLE", 4_018_920),
        "jordan.html": ("Michael Jordan", "1984-85", "CHI", 550_000),
        "bird.html": ("Larry Bird", "1979-80", "BOS", 650_000),
    }
    for filename, (name, first_season, team, salary) in expectations.items():
        html = (FIXTURES / filename).read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        assert extract_player_name(soup) == name
        rows = parse_all_salaries_table(html)
        assert rows, filename
        assert rows[0]["season"] == first_season
        assert rows[0]["team"] == team
        assert rows[0]["salary"] == salary
        assert all(not str(r["season"]).lower().startswith("career") for r in rows)


def test_parse_player_html_filters_pre_cap_when_requested() -> None:
    html = (FIXTURES / "bird.html").read_text(encoding="utf-8")
    rows = parse_player_html(
        html,
        player_id="birdla01",
        player_url="https://www.basketball-reference.com/players/b/birdla01.html",
        player_name="Larry Bird",
        html_cache_path=FIXTURES / "bird.html",
        start_season="1984-85",
        end_season=None,
    )
    assert rows
    assert min(r["season_start"] for r in rows) >= 1984
    assert all(r["salary_quality"] == "historical_unknown_quality" for r in rows)


def test_parse_salary_caps() -> None:
    html = (FIXTURES / "salary_cap.html").read_text(encoding="utf-8")
    rows = parse_salary_cap_table(html)
    assert rows[0]["season"] == "1984-85"
    assert rows[0]["salary_cap"] == 3_600_000
    assert rows[-1]["season_start"] >= 2025


def test_normalize_and_validate_sample() -> None:
    raw_rows = []
    for player_id, filename, url, name in [
        ("curryst01", "curry.html", "https://www.basketball-reference.com/players/c/curryst01.html", "Stephen Curry"),
        ("jordami01", "jordan.html", "https://www.basketball-reference.com/players/j/jordami01.html", "Michael Jordan"),
    ]:
        html = (FIXTURES / filename).read_text(encoding="utf-8")
        raw_rows.extend(
            parse_player_html(
                html,
                player_id=player_id,
                player_url=url,
                player_name=name,
                html_cache_path=FIXTURES / filename,
                start_season="1984-85",
                end_season=None,
            )
        )
    raw = pd.DataFrame(raw_rows)
    caps = pd.DataFrame(parse_salary_cap_table((FIXTURES / "salary_cap.html").read_text(encoding="utf-8")))
    player_salaries, team_season, history = normalize_salaries(raw, caps, min_team_rows=1)
    assert not player_salaries.empty
    assert player_salaries["cap_share"].notna().all()
    curry_2015 = player_salaries[
        (player_salaries["player_id"] == "curryst01") & (player_salaries["season"] == "2015-16")
    ]
    assert len(curry_2015) == 1
    assert float(curry_2015.iloc[0]["cap_share"]) > 0
    assert len(history) == len(player_salaries)
    assert not team_season.empty

    report = validate_salary_dataset(player_salaries, raw=raw, caps=caps)
    assert report["number_of_players"] == 2
    assert report["duplicate_count"] == 0
    assert report["structural_failure"] is False


def test_classify_salary_quality() -> None:
    assert classify_salary_quality(1) == "historical_unknown_quality"
    assert classify_salary_quality(None) == "missing"

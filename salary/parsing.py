"""Shared parsing helpers for salary pipeline data."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment, Tag

from ml.data import derive_player_id, normalize_player_url, parse_year_id

TEAM_HREF_PATTERN = re.compile(r"^/teams/(?P<team>[A-Z]{3})/\d{4}\.html$")
MONEY_PATTERN = re.compile(r"[^0-9\-]")


def parse_money(value: Any) -> int | None:
    """Parse a BBRef salary string like '$2,710,560' into integer dollars."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"", "-", "—", "–"}:
        return None
    negative = text.startswith("-") or text.startswith("(")
    digits = MONEY_PATTERN.sub("", text)
    if not digits:
        return None
    amount = int(digits)
    return -amount if negative else amount


def season_sort_key(season: str) -> tuple[int, int]:
    """Sort key for YYYY-YY season labels."""
    parsed = parse_year_id(season)
    start = parsed[1] if parsed[1] is not None else -1
    end = parsed[2] if parsed[2] is not None else -1
    return start, end


def player_id_from_url(player_url: str) -> str:
    """Derive Basketball Reference player_id from a normalized player_url."""
    normalized = normalize_player_url(player_url)
    if normalized is None:
        raise ValueError(f"Invalid player_url: {player_url}")
    return derive_player_id(normalized)


def normalize_player_url_or_raise(player_url: str) -> str:
    normalized = normalize_player_url(player_url)
    if normalized is None:
        raise ValueError(f"Invalid player_url: {player_url}")
    return normalized


def extract_player_name(soup: BeautifulSoup) -> str | None:
    """Extract display name from a BBRef player page."""
    meta_h1 = soup.select_one("div#meta h1")
    if meta_h1 is not None:
        text = meta_h1.get_text(" ", strip=True)
        if text:
            return text
    title = soup.find("title")
    if title and title.get_text(strip=True):
        # "Stephen Curry Stats | Basketball-Reference.com"
        return title.get_text(strip=True).split(" Stats")[0].strip() or None
    return None


def _table_from_comments(soup: BeautifulSoup, table_id: str) -> Tag | None:
    direct = soup.find("table", id=table_id)
    if isinstance(direct, Tag):
        return direct

    needle = f'id="{table_id}"'
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        if needle not in comment:
            continue
        commented = BeautifulSoup(comment, "html.parser")
        table = commented.find("table", id=table_id)
        if isinstance(table, Tag):
            return table
    return None


def extract_team_abbr(team_cell: Tag | None) -> str | None:
    """Extract team abbreviation from a BBRef team cell link when available."""
    if team_cell is None:
        return None
    link = team_cell.find("a")
    if not isinstance(link, Tag):
        return None
    href = str(link.get("href") or "")
    path = urlparse(href).path if "://" in href else href
    match = TEAM_HREF_PATTERN.match(path)
    if match:
        return match.group("team")
    return None


def parse_all_salaries_table(html: str) -> list[dict[str, Any]]:
    """Parse the player-page `all_salaries` table into raw row dicts."""
    soup = BeautifulSoup(html, "html.parser")
    table = _table_from_comments(soup, "all_salaries")
    if table is None:
        return []

    rows: list[dict[str, Any]] = []
    body_rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
    for row in body_rows:
        if not isinstance(row, Tag):
            continue
        if "thead" in (row.get("class") or []):
            continue
        season_cell = row.find(["th", "td"], attrs={"data-stat": "season"})
        if season_cell is None:
            continue
        season_text = season_cell.get_text(strip=True)
        if not season_text or season_text.lower().startswith("career"):
            continue

        season, season_start, season_end = parse_year_id(season_text)
        team_cell = row.find(["th", "td"], attrs={"data-stat": "team_name"})
        team_name = team_cell.get_text(strip=True) if team_cell else ""
        team = extract_team_abbr(team_cell)
        league_cell = row.find(["th", "td"], attrs={"data-stat": "lg_id"})
        league = league_cell.get_text(strip=True) if league_cell else ""
        salary_cell = row.find(["th", "td"], attrs={"data-stat": "salary"})
        salary = parse_money(salary_cell.get_text(strip=True) if salary_cell else None)

        rows.append(
            {
                "season": season,
                "season_start": season_start,
                "season_end": season_end,
                "team": team,
                "team_name": team_name or None,
                "league": league or None,
                "salary": salary,
            }
        )
    return rows


def parse_salary_cap_table(html: str) -> list[dict[str, Any]]:
    """Parse Basketball Reference salary-cap history table."""
    soup = BeautifulSoup(html, "html.parser")
    table = _table_from_comments(soup, "salary_cap_history")
    if table is None:
        raise ValueError("Could not find salary_cap_history table in HTML")

    rows: list[dict[str, Any]] = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        if "thead" in (row.get("class") or []):
            continue
        year_cell = row.find(["th", "td"], attrs={"data-stat": "year_id"})
        cap_cell = row.find(["th", "td"], attrs={"data-stat": "cap"})
        if year_cell is None or cap_cell is None:
            continue
        season_text = year_cell.get_text(strip=True)
        if not season_text:
            continue
        season, season_start, season_end = parse_year_id(season_text)
        salary_cap = parse_money(cap_cell.get_text(strip=True))
        if season is None or season_start is None or season_end is None or salary_cap is None:
            continue
        rows.append(
            {
                "season": season,
                "season_start": season_start,
                "season_end": season_end,
                "salary_cap": salary_cap,
            }
        )
    return rows


def classify_salary_quality(salary: int | None) -> str:
    """
    Assign a conservative salary_quality label.

    Basketball Reference does not expose reliable row-level provenance distinguishing
    reported contracts from historically imputed/estimated values, so known amounts are
    labeled historical_unknown_quality rather than guessed as reported/imputed.
    """
    if salary is None:
        return "missing"
    return "historical_unknown_quality"

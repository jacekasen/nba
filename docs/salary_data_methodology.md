# Salary Data Methodology

## Objective

This pipeline measures NBA player salaries relative to the league salary cap:

```text
cap_share = salary / salary_cap * 100
```

`cap_share` is stored in percentage points (for example `25.0`), not as a fraction.

A secondary frontend metric, `team_payroll_share`, is derived only when a team-season has enough known salary rows:

```text
team_payroll_share = salary / team_known_salary_total * 100
```

These metrics are intentionally distinct. Cap share compares a contract to the league cap; payroll share compares a contract to the known salaries on that team in that season.

## Primary Source

Historical salaries and salary-cap values are ingested from [Basketball Reference](https://www.basketball-reference.com/).

Ingestion path:

1. Player inventory (`data/01-pages/all_nba_players.csv`), optionally filtered by existing season coverage in `data/nba_player_stats.csv`
2. Individual Basketball Reference player pages
3. Hidden/`<!-- -->` commented HTML table `all_salaries`
4. Raw player-season-team salary rows
5. Salary-cap history page table `salary_cap_history`
6. Normalization + `cap_share` calculation
7. Validation report

Current team contract pages are **not** treated as a sufficient historical source. Player-page salary histories are the systematic source used here.

## Historical Coverage

Target coverage is the NBA salary-cap era: approximately **1984-85 through the present**.

Basketball Reference publishes salary-cap history beginning in 1984-85. Player salary tables may also contain earlier seasons; those pre-cap rows are retained when scraped, but salary-cap joins will fail for seasons before the cap table begins.

## Known Limitations And Caveats

- **1986-87** and **1989-90** are treated as known problematic seasons. Coverage may be sparse or incomplete even when some individual players have values.
- Older historical salaries on Basketball Reference may include estimated, extrapolated, or league-minimum substitutions in the upstream dataset.
- Exact historical reliability is not uniform across eras.
- Career salary footers on Basketball Reference are often annotated "(may be incomplete)". Those career aggregates are not ingested as observations.
- This pipeline **never invents** missing salary values.

## Salary Quality Labels

`salary_quality` values:

| Value | Meaning |
| --- | --- |
| `historical_unknown_quality` | A numeric salary was present on Basketball Reference, but row-level provenance does not reliably distinguish reported contracts from historically imputed/estimated values. |
| `missing` | No usable numeric salary was present for the observation. |
| `reported` | Reserved for future use when provenance clearly indicates a reported contract. |
| `imputed_or_estimated` | Reserved for future use when provenance clearly indicates estimation/imputation. |

Because Basketball Reference does not expose dependable per-row provenance flags in the player salary table, this pipeline conservatively classifies extracted numeric salaries as `historical_unknown_quality` rather than guessing `reported` or `imputed_or_estimated`.

## Scraper Behavior

- Every downloaded HTML page is cached under `data/01-raw/salaries/html/`.
- Cached pages are reused unless `--refresh` is supplied.
- Requests use a descriptive research User-Agent.
- Requests are rate-limited substantially (default 5 seconds between network calls).
- Transient failures use exponential backoff.
- Progress is persisted continuously; runs are resume-safe.
- CAPTCHAs / Cloudflare challenges / access blocks are **not** bypassed. If blocked, the scraper stops cleanly and preserves collected data.
- The website/frontend must never scrape Basketball Reference on user requests. Scraping is an occasional offline ingestion process.

## Pipeline Assumptions

- `player_url` is the stable player identity key, matching the ML pipeline convention.
- `player_id` is the Basketball Reference identifier embedded in `player_url` (for example `curryst01`).
- One output row represents one player-season-team salary observation.
- Team abbreviations are taken from Basketball Reference team links (`/teams/GSW/2010.html` → `GSW`) when available.
- Salaries are stored as integer dollars.
- `cap_share > 100` is not treated as automatically impossible; it is flagged for review during validation.
- `team_payroll_share` is only populated when a team-season has at least `MIN_TEAM_ROWS_FOR_PAYROLL_SHARE` known salary rows (default 8). Otherwise the field is left null because the team payroll denominator may be too incomplete.

## Outputs

- `data/02-intermediate/salaries/raw_player_salaries.csv`
- `data/02-intermediate/salaries/salary_caps.csv`
- `data/05-modeling/player_salaries.csv` (canonical)
- `data/05-modeling/team_season_salaries.csv`
- `data/05-modeling/player_salary_history.csv`
- `data/05-modeling/salary_validation_report.json`

## Reproduction

Representative sample:

```bash
conda activate nba
python -m salary.run_all --sample
```

Full historical collection (slow; respects cache/resume):

```bash
conda activate nba
python -m salary.run_all
```

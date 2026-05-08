# NBA Player Peak Analysis — Prompt

## Context
- Project: Analyze when NBA players peak using PER, BPM, and Win Shares.
- Data: Complete season-by-season stats (~25k+ seasons, 5,313 players) from Basketball Reference.
- Cleaned datasets produced in prior step:
  - `data/03-cleaned/nba_modern_era_clean.csv` (1993–2024; PER, BPM, VORP, WS available)
  - `data/03-cleaned/nba_per_era_clean.csv` (1978–2024; PER, WS widely available)
  - `data/03-cleaned/nba_complete_clean.csv` (1946–2024; basic stats across eras)
  - `data/03-cleaned/nba_career_stats.csv` (career aggregates post-filtering)
- Notebook to implement analysis: `notebooks/04_peak_analysis.ipynb`.

## Objective
Determine peak ages and peak seasons across players using PER, BPM, and WS metrics, segmented by era, position, and talent tier. Deliver robust, reproducible outputs and visualizations suitable for a portfolio and future interactive dashboard.

## Inputs
- Primary: `data/03-cleaned/nba_modern_era_clean.csv`, `nba_per_era_clean.csv`, `nba_complete_clean.csv`.
- Columns (core): `player_name, year_id, age, games, mp, per, bpm, vorp, ws, ws_per_48, era, team_name_abbr`.

## Definitions
- Season year: `year_id` is the season end year (int), e.g., 1998 for 1997–98.
- Minutes per game: `mpg = mp / games`.
- Peak Season (per metric): the season with maximum metric value after minimum playing-time filters. Ties resolved by:
  1) higher games, 2) higher mpg, 3) later year.
- Peak Age (per metric): `age` in the peak season.
- Optional smoothed peak: Use centered rolling mean (window=3 seasons) to reduce noise and define peak by the max of the smoothed series; keep raw peak as baseline.

## Filters (apply before peak detection)
- Per-season: `games >= 20` and `mpg >= 10` (same as cleaning step; recompute to be safe).
- Career: at least `3` seasons and `>= 82` total games for inclusion in peak analysis.
- Era handling:
  - Modern analysis (full advanced): limit to `era == 'Full Advanced'` or `year_id >= 1993`.
  - PER-era analysis: `year_id >= 1978` for PER/WS.
  - Historical baseline: basic participation metrics only from `nba_complete_clean.csv` (if needed for context).

## Segments
- Position: If available, use position column; if missing, skip or infer later. Otherwise segment only by era and talent.
- Talent tiers (define from modern/per-era data):
  - Elite: season `bpm >= 5` or `ws >= 10` or `per >= 22`.
  - All-Star: `bpm in [2,5)`, `ws in [6,10)`, or `per in [19,22)`.
  - Starter/Rotation: `bpm in [0,2)` or `ws in [3,6)` or `per in [15,19)`.
  - Role/Bench: below thresholds above.
Use a composite rule (first metric with data wins); prioritize BPM, then WS, then PER.

## Tasks
1) Load datasets; recompute `mpg`; assert dtypes and era labels.
2) Apply per-season and career filters; keep only seasons meeting criteria per segment.
3) Compute per-player peak season and peak age for each metric (PER, BPM, WS, WS/48):
   - Raw peak and smoothed-peak (rolling=3), when metric coverage allows.
   - Save both peak season `year_id` and `age`.
4) Summaries by segment:
   - Distribution of peak ages (median, IQR, mean, std) per metric.
   - Peaks by talent tier and by era (Modern vs PER-era).
   - Optional: peaks by years-of-experience instead of age.
5) Visualizations:
   - Histograms/density of peak ages per metric (Modern and PER-era).
   - Box/violin plots comparing peak ages by talent tier.
   - Line plots of average metric vs age (smoothed), with shaded 95% CI.
   - Heatmap: peak age vs career length bins or vs entry age bins.
6) Robustness checks:
   - Change filters (games >= 30, mpg >= 15) and compare peak-age shifts.
   - Smoothed vs raw peaks correlation; quantify differences.
7) Save outputs for downstream use.

## Outputs
- Tables (CSV/Parquet in `data/04-peak/`):
  - `player_peaks_modern.csv`: per player peaks for PER/BPM/WS/WS48 (raw + smoothed), with `player_name, peak_age_per, peak_year_per, peak_age_bpm, ... , tier, era`.
  - `peak_age_summary_by_tier.csv`: summary stats per metric x talent tier.
  - `peak_age_summary_by_era.csv`: summary stats per metric x era.
  - `metric_vs_age_curves.csv`: age (or experience) with mean±CI per metric.
- Figures (PNG/SVG in `figures/peak/`):
  - `peak_age_hist_{metric}_{segment}.png`
  - `peak_age_box_by_tier_{metric}.png`
  - `metric_vs_age_{metric}_{era}.png`
- Notebook cells in `notebooks/04_peak_analysis.ipynb` to reproduce all steps.

## Methodology Details
- Rolling smoothing: centered window=3, `min_periods=1` to keep edges.
- Experience proxy: `experience = season_index` per player (first valid season = 0), for alternative peak definitions.
- Confidence intervals: bootstrap (1,000 resamples) or analytic SEM for means by age.
- Missing data: restrict metric-specific analyses to seasons with that metric non-null.
- Multiple teams in a season: already handled in cleaning; assume one row per player-season.

## Validation & QA
- Sanity checks:
  - Elite players (e.g., LeBron, MJ, Jokic) should peak roughly ages 24–28 on BPM/WS.
  - WS/48 peaks may be slightly earlier than raw WS peaks.
  - Distribution tails: very early (<20) or very late (>35) peaks should be rare.
- Cross-metric consistency: Report correlations between peak ages across metrics.
- Sensitivity: Report how peak-age medians change under stricter filters.

## Deliverable Format
- Clean, reproducible code cells in `notebooks/04_peak_analysis.ipynb` with section headers:
  1) Setup & Load
  2) Filtering & Segmentation
  3) Peak Extraction (raw & smoothed)
  4) Summaries & Visualizations
  5) Robustness Checks
  6) Save Outputs
- Save artifacts to `data/04-peak/` and `figures/peak/` (create if missing).

## Starter Pseudocode (Python/pandas)
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/03-cleaned/nba_modern_era_clean.csv')
df['mpg'] = df['mp'] / df['games']
df = df[(df['games'] >= 20) & (df['mpg'] >= 10)]

def add_smoothed(group, cols, w=3):
    group = group.sort_values('year_id').copy()
    for c in cols:
        if c in group:
            group[c + '_smooth'] = group[c].rolling(w, center=True, min_periods=1).mean()
    return group

df = df.groupby('player_name', group_keys=False).apply(add_smoothed, cols=['per','bpm','ws','ws_per_48'])

def pick_peak(group, metric):
    g = group.dropna(subset=[metric]).copy()
    if g.empty:
        return pd.Series({'peak_year': np.nan, 'peak_age': np.nan, metric: np.nan})
    # tie-breaks: metric desc, games desc, mpg desc, year desc
    g = g.sort_values([metric, 'games', 'mpg', 'year_id'], ascending=[False, False, False, False])
    top = g.iloc[0]
    return pd.Series({'peak_year': int(top['year_id']), 'peak_age': int(top['age']), metric: float(top[metric])})

def compute_peaks(df, metric):
    peaks = df.groupby('player_name').apply(pick_peak, metric=metric).reset_index()
    peaks.columns = ['player_name', f'{metric}_peak_year', f'{metric}_peak_age', f'{metric}_peak_value']
    return peaks

peaks_raw = [compute_peaks(df, m) for m in ['per','bpm','ws','ws_per_48']]
from functools import reduce
peaks = reduce(lambda l, r: pd.merge(l, r, on='player_name', how='outer'), peaks_raw)
```

## Success Criteria
- Clear, reproducible peak ages per metric; sensible distributions (24–28 dominant for elite on BPM/WS).
- Segment comparisons documented (era, talent tiers) with at least three visualizations.
- Outputs saved and referenced for future dashboarding.

## Notes
- Prefer BPM for talent tiering; backfill with WS then PER when missing.
- Keep both raw and smoothed results; report differences succinctly.

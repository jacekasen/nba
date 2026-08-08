# ML Methodology

## Objective
Predict each player's short-term trajectory (`improving`, `stable`, `regressing`), next-season BPM change, and the probability that current performance is near eventual career peak.

## Data Source
- Raw source: `data/nba_player_stats.csv`
- Stable identity key: `player_url` (never `player_name`)

## Canonical Modeling Dataset
Built by `python -m ml.data` into `data/05-modeling/player_seasons.csv`.

Processing steps:
1. Normalize and validate `player_url`.
2. Remove `Did not play` rows.
3. Parse `year_id` (e.g., `2024-25`) into `season_start` and `season_end`.
4. Collapse to one row per player-season:
   - keep `nTM` aggregate row when present,
   - otherwise keep deterministic best row by games/minutes/metrics.
5. Apply configurable qualification thresholds (default: `games >= 15`, `mp >= 250`, required non-null BPM/PER/WS/WS48).

## Leakage Prevention
- Features at season `t` only use data available through `t`.
- Targets are built only when `t+1` is consecutive.
- Rolling windows are gap-aware and reset after non-consecutive seasons.
- No random row split: all evaluation is chronological by season.

## Feature Set (Core)
- Current-season fundamentals: age, experience, games, minutes, MPG, PER, BPM, VORP, WS, WS/48.
- Deltas: one-season BPM/PER/WS48/G/MP/MPG, two-season BPM delta.
- Rolling means: 2- and 3-season windows.
- Trend: 3-season BPM slope.
- Career context: prior and current career-high BPM, distance from career high.
- Context flags: changed team, season gap, seasons to date, season_start.

## Targets
- `target_bpm_change = bpm_{t+1} - bpm_t` for consecutive `t -> t+1` only.
- Trajectory threshold (default ±0.5 BPM):
  - `> +0.5`: improving
  - `< -0.5`: regressing
  - otherwise: stable

## Models
Trained by `python -m ml.train`.

Trajectory classification:
- Baseline: `DummyClassifier`
- Interpretable: multinomial logistic regression
- Main: `HistGradientBoostingClassifier`
- Optional validation-period calibration when supported by installed scikit-learn

BPM delta regression:
- Baselines: dummy zero and dummy mean
- Interpretable: Ridge
- Main: `HistGradientBoostingRegressor`

Near-peak probability (binary):
- Uses completed careers only (`last_season_end <= max_end - completion_gap_years`, default 5)
- Label: smoothed BPM within tolerance of eventual max (default 0.5 BPM)
- Uses only season-`t` features as inputs

## Evaluation
Classifier:
- Accuracy, macro F1, per-class precision/recall/F1, confusion matrix, log loss, multiclass Brier-style score.

Regressor:
- MAE, RMSE, R², grouped error by age and current BPM tier.

Near-peak:
- Accuracy/F1/Brier for binary classification; skipped if target is not valid/binary under chosen configuration.

## Limitations
- Career completion and near-peak labels are censoring-sensitive.
- Era effects and role changes are only partially captured by available box-score features.
- Results are probabilistic estimates, not deterministic forecasts.

# Board Man Gets Paid

**Author:** Jace Kasen  

### Project Overview
Educational NBA data analysis project covering:
- leakage-safe player trajectory / BPM machine learning (`ml/`)
- historical salary analysis via Cap Share (`salary/`)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/jacekasen/nba.git
   cd nba
   ```

2. **Create a Conda environment**
   ```bash
   conda create --name nba python=3.11
   conda activate nba
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ML Objective
Build a leakage-safe, script-based machine learning system that predicts:
- Player trajectory next season: `improving`, `stable`, or `regressing` (exit-aware: failing to log a qualified next season counts as regressing)
- Next-season BPM change (expected value over exit risk; a conditional if-plays estimate is also emitted)
- Probability the player logs a qualified season next year (continuation model)
- Probability the current season is near eventual career peak

## ML Architecture
Pipeline modules live in `ml/`:
- `ml/data.py`: canonical modeling dataset builder from raw scraped data
- `ml/features.py`: feature engineering + target construction
- `ml/train.py`: chronological training/evaluation/model selection
- `ml/predict.py`: latest-season player predictions for portfolio cards
- `ml/upload_supabase.py`: dry-run-safe Supabase payload validation/upload

Key output locations:
- `data/05-modeling/player_seasons.csv`
- `data/05-modeling/model_features.csv`
- `data/05-modeling/current_player_predictions.csv`
- `models/` (joblib artifacts + metrics/metadata)

## Dataset Construction
The modeling dataset is built from `data/nba_player_stats.csv` without mutating source data.

Core rules:
1. Use `player_url` as the stable identity key.
2. Remove `Did not play` rows.
3. Parse `year_id` (`YYYY-YY`) into `season_start` and `season_end`.
4. Collapse each player-season to one row, preferring aggregate `nTM` rows.
5. Apply configurable qualification thresholds (default: `games >= 15`, `mp >= 250`, non-null BPM/PER/WS/WS48).

## Leakage Prevention
- No random row splits: train/validation/test are chronological.
- Numeric BPM-change targets only created when season `t+1` is consecutive.
- Rolling features are gap-aware and reset after non-consecutive seasons.
- Feature inputs at season `t` only use information available through `t`.
- Final shipped models are refit on train+validation; reported test metrics come from held-out later seasons only.

## Survivorship-Bias Handling
Roughly 25% of qualified player-seasons (and >40% for declining veterans) are not followed
by another qualified season. Training only on survivors teaches the model that aging players
with bad seasons "bounce back" (the ones who don't simply vanish from the data). To counter this:
- The trajectory classifier labels a missing qualified next season as `regressing` when the league
  played on without the player (rows in the newest data season stay unlabeled/censored).
- A dedicated continuation classifier estimates P(qualified season next year).
- `predicted_bpm_delta` marginalizes over exit risk: `p * delta_if_plays + (1 - p) * (replacement_bpm - current_bpm)`
  with replacement level at -2.0 BPM (clamped so exiting never counts as improvement).

## Model Targets
- `target_trajectory`: multiclass label from next-season BPM change with asymmetric bands (improving > +0.75, regressing < -1.25, otherwise stable — minor declines sit inside the metric's noise floor), with exits labeled `regressing`
- `target_bpm_change`: numeric next-season BPM delta (survivors only)
- `target_played_next`: binary continuation label (qualified consecutive season logged or not)
- `near_peak`: binary label for completed careers only, based on smoothed BPM proximity to eventual peak

## Commands
Run in the conda env (`nba`) or equivalent Python 3.11 environment.

0. Run the end-to-end pipeline in one command:
   ```bash
   python -m ml.run_all
   ```

1. Build canonical modeling dataset:
   ```bash
   python -m ml.data
   ```

2. Build model features:
   ```bash
   python -m ml.features
   ```

3. Train and evaluate models:
   ```bash
   python -m ml.train
   ```

4. Generate current player predictions:
   ```bash
   python -m ml.predict
   ```
   Notes:
   - Output `season` is the forecast season (next season), based on the player's latest available source season.
   - Only players whose latest qualified season is within `--recency-years` (default 1) of the newest data season are included.
   - `predicted_bpm_delta` is the exit-risk-adjusted expectation; `predicted_bpm_delta_if_plays` is the conditional estimate; `continuation_probability` is P(qualified season next year).

5. Validate Supabase payload (no writes):
   ```bash
   python -m ml.upload_supabase --dry-run
   ```

6. Apply Supabase upsert (explicit):
   ```bash
   python -m ml.upload_supabase --apply
   ```

## Supabase Notes
- SQL migration: `supabase/migrations/20260807190000_create_player_predictions.sql`
- Browser/frontend should only use public anon credentials.
- Service role key must remain server-side only.

## Portfolio Integration
Integration details and a TypeScript example are documented in:
- `docs/portfolio_integration.md`

## Reproducibility
- Fixed random seed for model training.
- Saved metadata includes model version, training date, split boundaries, and selected models.
- Metrics and confusion matrix outputs are generated by `ml.train`.

## Known Limitations
- Near-peak labeling depends on a career-completion heuristic and is sensitive to right-censoring.
- Era effects and role/context shifts are not fully captured by box-score features alone.
- Predictions are probabilistic estimates, not guarantees.

## EPM Data

Dunks & Threes actual regular-season EPM exports are stored privately in the
`nba_player_epm` Supabase table. Raw licensed CSV files in `data/EPM/` are
gitignored. The uploader validates filenames, season values, required columns,
and duplicate player-season keys before any network write.

```bash
python -m epm.upload_supabase --dry-run

set -a
source .env.local
set +a
python -m epm.upload_supabase --apply
```

SQL migration: `supabase/migrations/20260809110000_create_nba_player_epm.sql`.

## Salary Analysis
Historical salary analysis is intentionally separate from the ML pipeline (`salary/`).

### Objective
Measure NBA player salaries relative to the salary cap, not only as nominal dollars:

```text
cap_share = salary / salary_cap * 100
```

Target historical range: approximately **1984-85 through the present** (NBA salary-cap era).

### Source
Basketball Reference player salary histories and salary-cap history. Scraping is an offline ingestion process with local HTML caching, rate limiting, resume support, and clean stops on access blocks. The eventual website must never scrape Basketball Reference on user requests.

### Commands
```bash
# Representative sample (recommended first run)
python -m salary.run_all --sample

# Or step-by-step
python -m salary.scrape --sample
python -m salary.caps
python -m salary.normalize
python -m salary.validate

# Full historical collection (slow; resumes and reuses cache)
python -m salary.run_all

# Supabase (dry-run default; writes require --apply)
python -m salary.upload_supabase --dry-run
python -m salary.upload_supabase --apply
```

Useful scrape flags: `--player`, `--limit`, `--start-season`, `--end-season`, `--refresh`, `--resume`.

### Outputs
- `data/02-intermediate/salaries/raw_player_salaries.csv`
- `data/02-intermediate/salaries/salary_caps.csv`
- `data/05-modeling/player_salaries.csv` (canonical)
- `data/05-modeling/team_season_salaries.csv`
- `data/05-modeling/player_salary_history.csv`
- `data/05-modeling/salary_validation_report.json`
- SQL migration: `supabase/migrations/20260809080000_create_player_salaries.sql`

### Historical-data Limitations
- Coverage and reliability are uneven before the modern contract era.
- 1986-87 and 1989-90 may be sparse or incomplete.
- Some older values in the upstream source may be estimated or incomplete; this pipeline does not invent missing salaries and conservatively labels extracted amounts as `historical_unknown_quality` unless stronger provenance is available.

Methodology details: `docs/salary_data_methodology.md`.

# NBA Season Trajectories Methodology

**Author:** Jace Kasen  
**Sub-project:** `trajectories/`

---

## 1. Overview and Objective

The **NBA Season Trajectories** sub-project processes and models game-level player results across the entire NBA play-by-play era (**1996–97 through 2025–26**, 30 seasons).

Its core objective is to move past post-hoc sports narratives (*"he hit the rookie wall,"* *"he surged after the All-Star break"*) by providing an empirical, game-by-game on-court performance timeline for every qualified player, anchored to a synchronized team game axis.

---

## 2. Metric Definition & Epistemic Boundaries

### On-Court Box-Score Plus-Minus ($\pm$)
- $\pm$ measures the team scoring margin while an individual player is on the court:
  $$\pm = \text{Team Points Scored While on Court} - \text{Opponent Points Scored While on Court}$$
- **What it captures**: Realized net scoring margin during a player's rotations.
- **What it does not prove**: It is not an isolated measure of individual skill. A player's single-game $\pm$ is heavily confounded by teammate quality, opponent lineups, and garbage time.
- **Analytical Treatment**:
  - We treat raw single-game $\pm$ as an empirical event, smoothed with **rolling window averages (5, 10, and 15 appearances)** to observe sustained momentum over multi-week stretches.
  - We also compute **Minutes-Weighted Rolling $\pm$**:
    $$\text{Rolling Weighted } \pm_{10} = \frac{\sum_{i=1}^{10} (\text{MIN}_i \times \pm_i)}{\sum_{i=1}^{10} \text{MIN}_i}$$
    preventing low-minute blowout appearances from dominating player trends.

---

## 3. The Fixed 1–82 Team Game Axis

Most public sports sites plot game logs along a player's appearance index ($1 \dots N$). That approach destroys calendar synchronization:
- If Player A missed 30 games with an ankle sprain and returned in March, plotting appearances $1 \dots 52$ obscures the injury and distorts comparisons with uninjured players.
- In our pipeline, every appearance is mapped to **`team_game_number` (1 through 82)**:
  - Missed games appear as explicit horizontal gaps on the timeline.
  - A player who returns at Team Game 55 is rendered at Game 55, allowing synchronized multi-player comparison.

### Handling Shortened & Lockout Seasons
Between 1996–97 and 2025–26, 26 seasons followed the standard 82-game schedule, while 4 seasons were shortened:
- **1998–99 Lockout**: 50 games
- **2011–12 Lockout**: 66 games
- **2019–20 COVID Bubble**: 71–75 games
- **2020–21 Compressed Season**: 72 games

For these seasons, the timeline max dynamically aligns with the season's team schedule ceiling, with explicit season-context badges in the UI.

---

## 4. Pipeline Architecture

1. **`trajectories/collect.py`**:
   - Fetches regular season game logs from NBA Stats via `curl_cffi` using Chrome TLS fingerprint impersonation.
   - Queries by regular season months (Months 1–8), caching raw JSON payloads to `data/trajectories/raw/{season}/month_{m}.json`.
2. **`trajectories/normalize.py`**:
   - Parses raw logs, resolves chronological team schedules to assign `team_game_number` (1..82).
   - Computes rolling metrics (`rolling_pm_5`, `rolling_pm_10`, `rolling_pm_15`, `minutes_weighted_pm_10`).
   - Flags trade transitions when a player switches teams midway through appearances.
   - Exports clean datasets to Parquet and CSV.
3. **`trajectories/validate.py`**:
   - Validates non-null constraints, game number boundaries, and rolling average sanity.
4. **`trajectories/upload_supabase.py`**:
   - Upserts records into `public.player_game_logs`.

# Portfolio Integration Guide

This repository publishes model predictions to Supabase. A portfolio frontend can consume that table for searchable player cards.

## Data Contract
Table: `public.player_predictions`

Key fields:
- `player_id`
- `player_name`
- `season` (forecast season, e.g. `2026-27`)
- `age`
- `current_bpm`
- `trajectory`
- `improving_probability`
- `stable_probability`
- `regressing_probability`
- `peak_probability`
- `predicted_bpm_delta`
- `prediction_factors` (json array)
- `model_version`
- `updated_at`

Semantics notes:
- `regressing_probability` is exit-aware: it includes the chance the player logs no qualified season next year.
- `predicted_bpm_delta` is the expected change marginalized over that exit risk (a player unlikely to play again is pulled toward replacement level, -2.0 BPM), not a conditional "if he plays" estimate.

## Frontend Safety Rules
1. Use `SUPABASE_URL` + public anon key only in browser code.
2. Never expose service role key in client code.
3. Treat outputs as probabilistic estimates, not guarantees.

## TypeScript Example (Framework-Neutral)
```ts
import { createClient } from "@supabase/supabase-js";

type PlayerPrediction = {
  player_id: string;
  player_name: string;
  season: string;
  age: number | null;
  current_bpm: number | null;
  trajectory: "improving" | "stable" | "regressing";
  improving_probability: number;
  stable_probability: number;
  regressing_probability: number;
  peak_probability: number | null;
  predicted_bpm_delta: number | null;
  prediction_factors: string[] | null;
  model_version: string;
  updated_at: string;
};

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);

export async function searchPlayers(query: string): Promise<PlayerPrediction[]> {
  if (!query.trim()) return [];

  const { data, error } = await supabase
    .from("player_predictions")
    .select("*")
    .ilike("player_name", `%${query}%`)
    .order("player_name", { ascending: true })
    .limit(10);

  if (error) throw error;
  return (data ?? []) as PlayerPrediction[];
}

export async function getPlayer(playerId: string): Promise<PlayerPrediction | null> {
  const { data, error } = await supabase
    .from("player_predictions")
    .select("*")
    .eq("player_id", playerId)
    .maybeSingle();

  if (error) throw error;
  return (data as PlayerPrediction | null) ?? null;
}
```

## UI Behavior
Autocomplete:
- Loading: show a small spinner/skeleton in the result list.
- Empty: show “No players found”.
- Error: show “Could not load players. Try again.”

Player card:
```text
Anthony Edwards · Age 24 · 2025–26

Trajectory: Improving

Improving — 61%
Stable — 28%
Regressing — 11%

Probability near career peak — 34%
Predicted next-season BPM change — +0.6

Why the model thinks this:
• BPM improved in three consecutive qualified seasons
• Minutes remained stable
• Current BPM is near the player's career high

Model version: 1.0
```

## Formatting Notes
- Multiply probabilities by 100 and round to whole percentages for UI.
- Show BPM delta with sign (`+0.6`, `-0.4`).
- Gracefully handle `peak_probability = null` (show “Not available”).
- Keep a “Model version” badge so users understand prediction provenance.

## Recommended API Pattern
- Client-side search for portfolio scale datasets.
- Optional server-side proxy if you need stricter request control or analytics.

## Salary Tables Integration

The salary pipeline (`salary/`) publishes three publicly readable tables that back the portfolio's
NBA Salary Cap Explorer at `/projects/nba/salaries`:

| Table | Role in the UI |
| --- | --- |
| `player_salaries` | One row per player-season-team. Drives the roster bars, donut, career history, and leaderboard. |
| `salary_caps` | Season cap values. Used for the cap headline and to detect seasons with no cap. |
| `team_season_salaries` | Pre-aggregated team-seasons. Used to build the team/season selectors without scanning the salary table. |

Query shapes the frontend relies on (all indexed by the migration):

```text
team + season      -> roster view
player_id          -> career salary history
cap_share desc     -> leaderboard, optionally filtered by season
```

Semantics the UI depends on:

- `cap_share` is percentage points against the league cap and can exceed 100 (Jordan, 1996-97).
  It is never treated as a slice of a 100% whole.
- `team_payroll_share` is salary divided by the team's known payroll, and is null when a team-season
  has fewer than `MIN_TEAM_ROWS_FOR_PAYROLL_SHARE` known salaries. The donut view disables itself in
  that case rather than implying complete coverage.
- `salary = null` means the amount is unknown. It is displayed as "not recorded" and excluded from
  payroll totals — never rendered as `$0`.

Known issues to fix upstream rather than in the frontend:

- `team_season_salaries.team_known_player_count` counts every record in the group, including rows
  with a null salary, so it is not comparable to the identically named column in `player_salaries`
  (which counts only known salaries). The frontend recomputes payroll coverage from the records it
  fetches.
- Some `player_name` values are stored as UTF-8 bytes decoded as Latin-1 (`Anderson VarejÃ£o`,
  `Bogdan BogdanoviÄ‡`); roughly 148 distinct names are affected. The frontend repairs these for
  display only.

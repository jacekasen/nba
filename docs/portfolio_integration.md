# Portfolio Integration Guide

This repository publishes model predictions to Supabase. A portfolio frontend can consume that table for searchable player cards.

## Data Contract
Table: `public.player_predictions`

Key fields:
- `player_id`
- `player_name`
- `season`
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

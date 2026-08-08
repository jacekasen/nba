create table if not exists public.player_predictions (
  player_id text primary key,
  player_name text not null,
  season text not null,
  age integer,
  current_bpm double precision,
  trajectory text not null check (trajectory in ('improving', 'stable', 'regressing')),
  improving_probability double precision not null check (improving_probability >= 0 and improving_probability <= 1),
  stable_probability double precision not null check (stable_probability >= 0 and stable_probability <= 1),
  regressing_probability double precision not null check (regressing_probability >= 0 and regressing_probability <= 1),
  peak_probability double precision check (peak_probability is null or (peak_probability >= 0 and peak_probability <= 1)),
  predicted_bpm_delta double precision,
  model_version text not null,
  prediction_factors jsonb,
  updated_at timestamptz not null,
  check (abs((improving_probability + stable_probability + regressing_probability) - 1.0) <= 0.01)
);

alter table public.player_predictions enable row level security;

drop policy if exists player_predictions_public_read on public.player_predictions;
create policy player_predictions_public_read
on public.player_predictions
for select
to anon
using (true);

create index if not exists idx_player_predictions_player_name_lower
on public.player_predictions (lower(player_name));

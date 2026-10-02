create table if not exists public.player_game_logs (
  id bigint generated always as identity primary key,
  player_id integer not null,
  player_name text not null,
  season text not null,
  game_id text not null,
  game_date date not null,
  team_id integer not null,
  team_abbreviation text not null,
  opponent_abbreviation text,
  is_home boolean not null,
  team_game_number integer not null,
  player_season_game_number integer not null,
  win_loss text,
  minutes double precision not null default 0,
  plus_minus integer not null default 0,
  points integer not null default 0,
  rebounds integer not null default 0,
  assists integer not null default 0,
  rolling_pm_5 double precision,
  rolling_pm_10 double precision,
  rolling_pm_15 double precision,
  minutes_weighted_pm_10 double precision,
  is_trade_transition boolean not null default false,
  created_at timestamptz not null default timezone('utc'::text, now()),
  constraint uq_player_game_team unique (player_id, game_id, team_id)
);

alter table public.player_game_logs enable row level security;

drop policy if exists player_game_logs_public_read on public.player_game_logs;
create policy player_game_logs_public_read
on public.player_game_logs
for select
to anon
using (true);

create index if not exists idx_pgl_player_season
on public.player_game_logs (player_id, season);

create index if not exists idx_pgl_season_team
on public.player_game_logs (season, team_abbreviation);

create index if not exists idx_pgl_season_team_game
on public.player_game_logs (season, team_game_number);

create index if not exists idx_pgl_player_name_lower
on public.player_game_logs (lower(player_name));

-- Private storage for licensed Dunks & Threes actual regular-season EPM exports.
-- Uploads use the service role through epm/upload_supabase.py.

create table if not exists public.nba_player_epm (
  nba_player_id bigint not null,
  season text not null,
  season_start integer not null,
  season_end integer not null,
  season_type integer not null,
  player_name text not null,
  age integer,
  team_id bigint,
  team_alias text,
  team_alias_all text,
  position text,
  games integer,
  roster_games integer,
  starts integer,
  minutes double precision,
  rookie_year integer,
  height_inches integer,
  weight_lbs integer,
  minutes_per_game double precision,
  offensive_epm double precision,
  defensive_epm double precision,
  epm double precision,
  estimated_wins double precision,
  usage_rate double precision,
  true_shooting_pct double precision,
  effective_fg_pct double precision,
  rim_fg_pct double precision,
  midrange_fg_pct double precision,
  two_point_fg_pct double precision,
  three_point_fg_pct double precision,
  free_throw_pct double precision,
  offensive_rebound_pct double precision,
  defensive_rebound_pct double precision,
  assist_pct double precision,
  turnover_pct double precision,
  steal_pct double precision,
  block_pct double precision,
  rim_fga_per_75 double precision,
  midrange_fga_per_75 double precision,
  three_point_fga_per_75 double precision,
  fta_per_75 double precision,
  fga_per_75 double precision,
  source_file text not null,
  updated_at timestamptz not null default now(),
  primary key (nba_player_id, season, season_type)
);

alter table public.nba_player_epm enable row level security;

drop policy if exists nba_player_epm_public_read on public.nba_player_epm;
drop policy if exists nba_player_epm_authenticated_read on public.nba_player_epm;
revoke all on table public.nba_player_epm from anon, authenticated;

create index if not exists idx_nba_player_epm_season
on public.nba_player_epm (season);

create index if not exists idx_nba_player_epm_player_name_lower
on public.nba_player_epm (lower(player_name));

create index if not exists idx_nba_player_epm_value
on public.nba_player_epm (epm desc nulls last);

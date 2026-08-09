-- Historical NBA salary analysis tables for Board Man Gets Paid.
-- Dry-run-safe application code lives in salary/upload_supabase.py.
-- Service-role credentials must remain server-side only.

create table if not exists public.salary_caps (
  season text primary key,
  season_start integer not null,
  season_end integer not null,
  salary_cap bigint not null check (salary_cap > 0),
  updated_at timestamptz not null default now()
);

create table if not exists public.player_salaries (
  player_id text not null,
  player_name text not null,
  player_url text not null,
  season text not null,
  season_start integer not null,
  season_end integer not null,
  team text not null,
  salary bigint check (salary is null or salary >= 0),
  salary_cap bigint check (salary_cap is null or salary_cap > 0),
  cap_share double precision,
  salary_source text not null,
  salary_quality text not null check (
    salary_quality in (
      'reported',
      'historical_unknown_quality',
      'imputed_or_estimated',
      'missing'
    )
  ),
  team_payroll_share double precision,
  team_known_salary_total bigint,
  team_known_player_count integer,
  updated_at timestamptz not null default now(),
  primary key (player_id, season, team)
);

create table if not exists public.team_season_salaries (
  team text not null,
  season text not null,
  season_start integer not null,
  season_end integer not null,
  salary_cap bigint,
  team_known_salary_total bigint,
  team_known_player_count integer,
  total_cap_share double precision,
  max_cap_share double precision,
  payroll_share_available boolean not null default false,
  team_payroll_vs_cap double precision,
  updated_at timestamptz not null default now(),
  primary key (team, season)
);

alter table public.salary_caps enable row level security;
alter table public.player_salaries enable row level security;
alter table public.team_season_salaries enable row level security;

drop policy if exists salary_caps_public_read on public.salary_caps;
create policy salary_caps_public_read
on public.salary_caps
for select
to anon
using (true);

drop policy if exists player_salaries_public_read on public.player_salaries;
create policy player_salaries_public_read
on public.player_salaries
for select
to anon
using (true);

drop policy if exists team_season_salaries_public_read on public.team_season_salaries;
create policy team_season_salaries_public_read
on public.team_season_salaries
for select
to anon
using (true);

create index if not exists idx_player_salaries_player_id
on public.player_salaries (player_id);

create index if not exists idx_player_salaries_season
on public.player_salaries (season);

create index if not exists idx_player_salaries_team
on public.player_salaries (team);

create index if not exists idx_player_salaries_team_season
on public.player_salaries (team, season);

create index if not exists idx_player_salaries_player_season
on public.player_salaries (player_id, season);

create index if not exists idx_player_salaries_cap_share
on public.player_salaries (cap_share desc nulls last);

create index if not exists idx_player_salaries_player_name_lower
on public.player_salaries (lower(player_name));

create index if not exists idx_team_season_salaries_season
on public.team_season_salaries (season);

CREATE INDEX idx_tbl
ON footballbetspredictor.table (season_id, date);

CREATE INDEX idx_team_tbl
ON footballbetspredictor.tableteam (team_id, table_id);

CREATE INDEX idx_match
ON footballbetspredictor.match (date, home_team_id, away_team_id);

CREATE INDEX idx_season
ON footballbetspredictor.season (end_date);

CREATE INDEX idx_season_date_league
ON footballbetspredictor.season (end_date, league_id);
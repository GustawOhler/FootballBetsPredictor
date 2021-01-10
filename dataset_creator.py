from datetime import datetime

from models import Match, Team, Table, TableTeam


def create_dataset():
    dataset = {"home_position": [], "away_position": [], "result": []}
    root_matches = Match.select()
    for root_match in root_matches:
        table_before_match = Table.select().where(Table.season == root_match.season & Table.date == root_match.date.date())
        home_team_table_stats = TableTeam.get(TableTeam.team == root_match.home_team)
        away_team_table_stats = TableTeam.get(TableTeam.team == root_match.away_team)
        dataset.get("home_position").append(home_team_table_stats.position)
        dataset.get("away_position").append(away_team_table_stats.position)
        dataset.get("result").append(root_match.full_time_result)

    print(dataset)
from datetime import datetime

from models import Match, Team, Table, TableTeam

results_dict = {'H': 0, 'D': 1, 'A': 2}


def create_dataset():
    dataset = {"home_position": [], "home_played_matches": [], "home_wins": [], "home_draws": [], "home_loses": [],
               "home_goal_difference": [],
               "away_position": [], "away_played_matches": [], "away_wins": [], "away_draws": [], "away_loses": [],
               "away_goal_difference": [],
               "result": []}
    root_matches = Match.select()
    for root_match in root_matches:
        table_before_match = Table.get(
            (Table.season == root_match.season) & (Table.date == root_match.date.date()))
        home_team_table_stats = TableTeam.get((TableTeam.team == root_match.home_team) & (TableTeam.table == table_before_match))
        away_team_table_stats = TableTeam.get((TableTeam.team == root_match.away_team) & (TableTeam.table == table_before_match))
        home_last_5_matches = Match.select().where(
                                                  (Match.date < root_match.date) &
                                                  ((Match.home_team == root_match.home_team)
                                                   | (Match.away_team == root_match.home_team))).order_by(Match.date.desc()).limit(5)
        away_last_5_matches = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_match.away_team)
             | (Match.away_team == root_match.away_team))).order_by(Match.date.desc()).limit(5)
        dataset.get("home_position").append(home_team_table_stats.position)
        dataset.get("home_played_matches").append(home_team_table_stats.matches_played)
        dataset.get("home_wins").append(home_team_table_stats.wins)
        dataset.get("home_draws").append(home_team_table_stats.draws)
        dataset.get("home_loses").append(home_team_table_stats.loses)
        dataset.get("home_goal_difference").append(home_team_table_stats.goal_difference)
        dataset.get("away_position").append(away_team_table_stats.position)
        dataset.get("away_played_matches").append(away_team_table_stats.matches_played)
        dataset.get("away_wins").append(away_team_table_stats.wins)
        dataset.get("away_draws").append(away_team_table_stats.draws)
        dataset.get("away_loses").append(away_team_table_stats.loses)
        dataset.get("away_goal_difference").append(away_team_table_stats.goal_difference)
        dataset.get("result").append(results_dict[root_match.full_time_result.value])

    # print(dataset)
    return dataset
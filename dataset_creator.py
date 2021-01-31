from datetime import datetime
import pandas as pd
from models import Match, Team, Table, TableTeam, MatchResult

results_dict = {'H': 0, 'D': 1, 'A': 2}


def get_scored_goals(matches: [Match], team: Team):
    return sum(match.full_time_home_goals for match in matches
               if match.home_team == team) + sum(match.full_time_away_goals for match in matches
                                                 if match.away_team == team)


def get_conceded_goals(matches: [Match], team: Team):
    return sum(match.full_time_away_goals for match in matches
               if match.home_team == team) + sum(match.full_time_home_goals for match in matches
                                                 if match.away_team == team)


def create_dataset():
    dataset = {"home_position": [], "home_played_matches": [], "home_wins": [], "home_draws": [], "home_loses": [],
               "home_goals_scored": [], "home_goals_conceded": [], "home_goal_difference": [],
               "home_team_wins_in_last_5_matches": [],
               "home_team_draws_in_last_5_matches": [],
               "home_team_loses_in_last_5_matches": [],
               "home_team_scored_goals_in_last_5_matches": [],
               "home_team_conceded_goals_in_last_5_matches": [],
               "away_position": [], "away_played_matches": [], "away_wins": [], "away_draws": [], "away_loses": [],
               "away_goals_scored": [], "away_goals_conceded": [], "away_goal_difference": [],
               "away_team_wins_in_last_5_matches": [],
               "away_team_draws_in_last_5_matches": [],
               "away_team_loses_in_last_5_matches": [],
               "away_team_scored_goals_in_last_5_matches": [],
               "away_team_conceded_goals_in_last_5_matches": [],
               "result": []}
    root_matches = Match.select()
    for root_match in root_matches:
        table_before_match = Table.get(
            (Table.season == root_match.season) & (Table.date == root_match.date.date()))
        home_team_table_stats = TableTeam.get(
            (TableTeam.team == root_match.home_team) & (TableTeam.table == table_before_match))
        away_team_table_stats = TableTeam.get(
            (TableTeam.team == root_match.away_team) & (TableTeam.table == table_before_match))
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
        dataset.get("home_goals_scored").append(home_team_table_stats.goals_scored)
        dataset.get("home_goals_conceded").append(home_team_table_stats.goals_conceded)
        dataset.get("home_goal_difference").append(home_team_table_stats.goal_difference)
        dataset.get("home_team_wins_in_last_5_matches").append(sum(1 for match in home_last_5_matches
                                                                   if (match.full_time_result == MatchResult.HOME_WIN
                                                                       and match.home_team == root_match.home_team)
                                                                   or (match.full_time_result == MatchResult.AWAY_WIN
                                                                       and match.away_team == root_match.home_team)))
        dataset.get("home_team_draws_in_last_5_matches").append(sum(1 for match in home_last_5_matches
                                                                    if match.full_time_result == MatchResult.DRAW))
        dataset.get("home_team_loses_in_last_5_matches").append(sum(1 for match in home_last_5_matches
                                                                    if (match.full_time_result == MatchResult.AWAY_WIN
                                                                        and match.home_team == root_match.home_team)
                                                                    or (match.full_time_result == MatchResult.HOME_WIN
                                                                        and match.away_team == root_match.home_team)))
        dataset.get("home_team_scored_goals_in_last_5_matches").append(get_scored_goals(home_last_5_matches,
                                                                                        root_match.home_team))
        dataset.get("home_team_conceded_goals_in_last_5_matches").append(get_conceded_goals(home_last_5_matches,
                                                                                            root_match.home_team))
        dataset.get("away_position").append(away_team_table_stats.position)
        dataset.get("away_played_matches").append(away_team_table_stats.matches_played)
        dataset.get("away_wins").append(away_team_table_stats.wins)
        dataset.get("away_draws").append(away_team_table_stats.draws)
        dataset.get("away_loses").append(away_team_table_stats.loses)
        dataset.get("away_goals_scored").append(away_team_table_stats.goals_scored)
        dataset.get("away_goals_conceded").append(away_team_table_stats.goals_conceded)
        dataset.get("away_goal_difference").append(away_team_table_stats.goal_difference)
        dataset.get("away_team_wins_in_last_5_matches").append(sum(1 for match in away_last_5_matches
                                                                   if (match.full_time_result == MatchResult.HOME_WIN
                                                                       and match.home_team == root_match.away_team)
                                                                   or (match.full_time_result == MatchResult.AWAY_WIN
                                                                       and match.away_team == root_match.away_team)))
        dataset.get("away_team_draws_in_last_5_matches").append(sum(1 for match in away_last_5_matches
                                                                    if match.full_time_result == MatchResult.DRAW))
        dataset.get("away_team_loses_in_last_5_matches").append(sum(1 for match in away_last_5_matches
                                                                    if (match.full_time_result == MatchResult.AWAY_WIN
                                                                        and match.home_team == root_match.away_team)
                                                                    or (match.full_time_result == MatchResult.HOME_WIN
                                                                        and match.away_team == root_match.away_team)))
        dataset.get("away_team_scored_goals_in_last_5_matches").append(get_scored_goals(away_last_5_matches,
                                                                                        root_match.away_team))
        dataset.get("away_team_conceded_goals_in_last_5_matches").append(get_conceded_goals(away_last_5_matches,
                                                                                            root_match.away_team))
        dataset.get("result").append(results_dict[root_match.full_time_result.value])

    pd.DataFrame.from_dict(dataset).to_csv('dataset.csv', index=False)
    # print(dataset)
    return dataset


def load_dataset():
    return pd.read_csv("dataset.csv")

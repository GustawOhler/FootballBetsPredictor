from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from flatten_dict import flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

from models import Match, Team, Table, TableTeam, MatchResult

results_dict = {'H': 0, 'D': 1, 'A': 2}


@dataclass
class AggregatedMatchData:
    wins: float = 0.0
    draws: float = 0.0
    loses: float = 0.0
    scored_goals: float = 0.0
    conceded_goals: float = 0.0
    shots_fired: float = 0.0
    shots_fired_on_target: float = 0.0
    shots_conceded: float = 0.0
    shots_conceded_on_target: float = 0.0


@dataclass
class NNDatasetRow:
    home_position: int
    home_played_matches: int
    home_wins: float
    home_draws: float
    home_loses: float
    home_goals_scored: float
    home_goals_conceded: float
    home_goal_difference: int
    home_last_5_matches: AggregatedMatchData
    home_last_3_matches_at_home: AggregatedMatchData
    away_position: int
    away_played_matches: int
    away_wins: float
    away_draws: float
    away_loses: float
    away_goals_scored: float
    away_goals_conceded: float
    away_goal_difference: int
    away_last_5_matches: AggregatedMatchData
    away_last_3_matches_at_away: AggregatedMatchData
    result: int
    home_odds: float
    draw_odds: float
    away_odds: float
    last_3_matches_between_teams: AggregatedMatchData


def get_scored_goals(matches: [Match], team: Team):
    return sum(match.full_time_home_goals for match in matches
               if match.home_team == team) + sum(match.full_time_away_goals for match in matches
                                                 if match.away_team == team)


def get_conceded_goals(matches: [Match], team: Team):
    return sum(match.full_time_away_goals for match in matches
               if match.home_team == team) + sum(match.full_time_home_goals for match in matches
                                                 if match.away_team == team)


def get_shots_fired(matches: [Match], team: Team):
    return sum(match.home_team_shots for match in matches
               if match.home_team == team) + sum(match.away_team_shots for match in matches
                                                 if match.away_team == team)


def get_shots_fired_on_target(matches: [Match], team: Team):
    return sum(match.home_team_shots_on_target for match in matches
               if match.home_team == team) + sum(match.away_team_shots_on_target for match in matches
                                                 if match.away_team == team)


def get_shots_conceded(matches: [Match], team: Team):
    return sum(match.away_team_shots for match in matches
               if match.home_team == team) + sum(match.home_team_shots for match in matches
                                                 if match.away_team == team)


def get_shots_conceded_on_target(matches: [Match], team: Team):
    return sum(match.away_team_shots_on_target for match in matches
               if match.home_team == team) + sum(match.home_team_shots_on_target for match in matches
                                                 if match.away_team == team)


def fill_last_matches_stats(matches: [Match], team: Team):
    matches_count = matches.count()
    if matches_count == 0:
        return AggregatedMatchData()
    return AggregatedMatchData(wins=(sum(1 for match in matches
                                         if (match.full_time_result == MatchResult.HOME_WIN
                                             and match.home_team == team)
                                         or (match.full_time_result == MatchResult.AWAY_WIN
                                             and match.away_team == team)) / matches_count),
                               draws=(sum(1 for match in matches if match.full_time_result ==
                                          MatchResult.DRAW) / matches_count),
                               loses=(sum(1 for match in matches
                                          if (match.full_time_result == MatchResult.AWAY_WIN
                                              and match.home_team == team)
                                          or (match.full_time_result == MatchResult.HOME_WIN
                                              and match.away_team == team)) / matches_count),
                               scored_goals=get_scored_goals(matches, team) / matches_count,
                               conceded_goals=get_conceded_goals(matches, team) / matches_count,
                               shots_fired=get_shots_fired(matches, team) / matches_count,
                               shots_fired_on_target=get_shots_fired_on_target(matches, team) / matches_count,
                               shots_conceded=get_shots_conceded(matches, team) / matches_count,
                               shots_conceded_on_target=get_shots_conceded_on_target(matches, team) / matches_count)




def create_dataset():
    dataset = []
    root_matches = Match.select()
    root_matches_count = root_matches.count()
    sum_of_time_elapsed = 0
    for index, root_match in enumerate(root_matches.iterator()):
        row_create_start = timer()
        root_home_team = root_match.home_team
        root_away_team = root_match.away_team
        table_before_match = Table.get(
            (Table.season == root_match.season) & (Table.date == root_match.date.date()))
        home_team_table_stats = TableTeam.get(
            (TableTeam.team == root_home_team) & (TableTeam.table == table_before_match))
        away_team_table_stats = TableTeam.get(
            (TableTeam.team == root_away_team) & (TableTeam.table == table_before_match))
        home_last_5_matches = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_home_team)
             | (Match.away_team == root_home_team))).order_by(Match.date.desc()).limit(5)
        home_last_3_matches_as_home = Match.select().where(
            (Match.date < root_match.date) &
            (Match.home_team == root_home_team)).order_by(Match.date.desc()).limit(5)
        away_last_5_matches = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_away_team)
             | (Match.away_team == root_away_team))).order_by(Match.date.desc()).limit(5)
        away_last_3_matches_as_away = Match.select().where(
            (Match.date < root_match.date) &
            (Match.away_team == root_away_team)).order_by(Match.date.desc()).limit(5)
        last_3_matches_between_teams = Match.select().where(
            (Match.date < root_match.date) &
            ((Match.home_team == root_home_team & Match.away_team == root_away_team) |
             (Match.home_team == root_away_team & Match.away_team == root_home_team))).order_by(Match.date.desc()).limit(3)
        if home_last_5_matches.count() < 2 or away_last_5_matches.count() < 2 or home_team_table_stats.matches_played < 2 or \
                away_team_table_stats.matches_played < 2:
            continue
        dataset_row = NNDatasetRow(home_position=home_team_table_stats.position, home_played_matches=home_team_table_stats.matches_played,
                                   home_wins=home_team_table_stats.wins / home_team_table_stats.matches_played,
                                   home_draws=home_team_table_stats.draws / home_team_table_stats.matches_played,
                                   home_loses=home_team_table_stats.loses / home_team_table_stats.matches_played,
                                   home_goals_scored=home_team_table_stats.goals_scored / home_team_table_stats.matches_played,
                                   home_goals_conceded=home_team_table_stats.goals_conceded / home_team_table_stats.matches_played,
                                   home_goal_difference=home_team_table_stats.goal_difference,
                                   home_last_5_matches=fill_last_matches_stats(home_last_5_matches, root_home_team),
                                   home_last_3_matches_at_home=fill_last_matches_stats(home_last_3_matches_as_home, root_home_team),
                                   away_position=away_team_table_stats.position, away_played_matches=away_team_table_stats.matches_played,
                                   away_wins=away_team_table_stats.wins / away_team_table_stats.matches_played,
                                   away_draws=away_team_table_stats.draws / away_team_table_stats.matches_played,
                                   away_loses=away_team_table_stats.loses / away_team_table_stats.matches_played,
                                   away_goals_scored=away_team_table_stats.goals_scored / away_team_table_stats.matches_played,
                                   away_goals_conceded=away_team_table_stats.goals_conceded / away_team_table_stats.matches_played,
                                   away_goal_difference=away_team_table_stats.goal_difference,
                                   away_last_5_matches=fill_last_matches_stats(away_last_5_matches, root_away_team),
                                   away_last_3_matches_at_away=fill_last_matches_stats(away_last_3_matches_as_away, root_away_team),
                                   result=results_dict[root_match.full_time_result.value], home_odds=root_match.average_home_odds,
                                   draw_odds=root_match.average_draw_odds,
                                   away_odds=root_match.average_away_odds,
                                   last_3_matches_between_teams=fill_last_matches_stats(last_3_matches_between_teams, root_home_team))
        dataset.append(dataset_row)
        sum_of_time_elapsed = sum_of_time_elapsed + timer() - row_create_start
        index_from_1 = index + 1
        print("Przetwarzany rekord " + str(index_from_1) + " z " + str(root_matches_count) + " czyli "
              + str("{:.2f}".format(index_from_1 * 100 / root_matches_count)) + "%. Sredni czas przetwarzania dla 100 rekordow: " + str(
            "{:.2f} s".format(sum_of_time_elapsed * 100/index_from_1)), end=("\r" if index_from_1 != root_matches_count else "\n"))

    csv_proccesing_start = timer()
    pd_dataset = pd.DataFrame(flatten(asdict(row), reducer='underscore') for row in dataset)
    pd_dataset.to_csv('dataset.csv', index=False, float_format='%.3f')
    csv_proccesing_end = timer()
    print("Czas przetwarzania rekordow do csvki: " + str("{:.2f} s".format(csv_proccesing_end - csv_proccesing_start)))
    return pd_dataset


def load_dataset():
    return pd.read_csv("dataset.csv")


def split_dataset(dataset: pd.DataFrame, validation_split=0.2):
    x = dataset.drop('result', axis='columns').to_numpy()
    y = dataset['result'].to_numpy()
    one_hot_y = to_categorical(y, num_classes=3)
    odds = dataset[['home_odds', 'draw_odds', 'away_odds']].to_numpy()
    zero_vector = np.zeros((one_hot_y.shape[0], 1))
    y_final = np.concatenate((one_hot_y, zero_vector, odds), axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y_final, test_size=validation_split)
    return (x_train, y_train), (x_val, y_val)

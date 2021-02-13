from typing import List, Tuple
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta
from models import Season, Team, Match, League, TeamSeason, Table, TableTeam, MatchResult


def add_points_to_tuple(home_team, away_team, home_points, away_points, home_goals, away_goals, small_table):
    if home_team in small_table:
        small_table[home_team] = (small_table[home_team][0] + home_points,
                                  small_table[home_team][1] + home_goals - away_goals)
    else:
        small_table[home_team] = (home_points, home_goals - away_goals)
    if away_team in small_table:
        small_table[away_team] = (small_table[away_team][0] + away_points,
                                  small_table[away_team][1] + away_goals - home_goals)
    else:
        small_table[away_team] = (away_points, away_goals - home_goals)


def process_match_to_head_to_head_table(match: Match, head_to_head_table):
    if match.full_time_result == MatchResult.HOME_WIN:
        add_points_to_tuple(match.home_team, match.away_team, 3, 0, match.full_time_home_goals,
                            match.full_time_away_goals, head_to_head_table)
    elif match.full_time_result == MatchResult.DRAW:
        add_points_to_tuple(match.home_team, match.away_team, 1, 1, match.full_time_home_goals,
                            match.full_time_away_goals, head_to_head_table)
    elif match.full_time_result == MatchResult.AWAY_WIN:
        add_points_to_tuple(match.home_team, match.away_team, 0, 3, match.full_time_home_goals,
                            match.full_time_away_goals, head_to_head_table)


def accurate_sort_by_league(teams_with_same_points: List[TableTeam], season: Season, date, league_of_table: League):
    if league_of_table.league_name == 'Premier League' or league_of_table.league_name == 'Ligue 1':
        return sorted(teams_with_same_points, key=lambda x: (x.goal_difference, x.goals_scored), reverse=True)
    else:
        only_teams = [team.team for team in teams_with_same_points]
        matches = Match.select().where((Match.season == season) & (Match.date < date) &
                                       (Match.home_team << only_teams) & (Match.away_team << only_teams))
        if len(matches) == len(only_teams) * (len(only_teams) - 1):
            head_to_head_table = {}
            for match in matches:
                process_match_to_head_to_head_table(match, head_to_head_table)
            tuple_to_sort = ((team, head_to_head_table[team.team]) for team in teams_with_same_points)
            if league_of_table.league_name == 'La Liga' or league_of_table.league_name == 'Serie A':
                sorted_tuples = sorted(tuple_to_sort, key=lambda x: (x[1][0], x[1][1], x[0].goal_difference,
                                                                     x[0].goals_scored), reverse=True)
            elif league_of_table.league_name == 'Bundesliga':
                sorted_tuples = sorted(tuple_to_sort, key=lambda x: (x[0].goal_difference, x[0].goals_scored, x[1][0]),
                                       reverse=True)
            return [single_tuple[0] for single_tuple in sorted_tuples]
        else:
            return sorted(teams_with_same_points, key=lambda x: (x.goal_difference, x.goals_scored), reverse=True)


def find_first_and_last_index(teams_in_table: List[TableTeam], searched_points_value):
    first_index = -1
    last_index = -1
    for index, item in enumerate(teams_in_table):
        if item.points == searched_points_value:
            if first_index == -1:
                first_index = index
            elif last_index < index:
                last_index = index
    return first_index, last_index


def sort_teams_in_table(teams_in_table: List[TableTeam], season: Season, date, league_of_table: League):
    sorted_teams = teams_in_table
    if any(team.matches_played > 0 for team in teams_in_table):
        sorted_teams = sorted(teams_in_table, key=lambda x: x.points, reverse=True)
        same_points_count = Counter(getattr(item, 'points') for item in sorted_teams)
        for item in same_points_count:
            if same_points_count[item] > 1:
                teams_to_accurate_sorting = [team for team in sorted_teams if team.points == item]
                teams_after_acc_sort = accurate_sort_by_league(teams_to_accurate_sorting, season, date, league_of_table)
                indexes = find_first_and_last_index(sorted_teams, item)
                sorted_teams[indexes[0]: indexes[1] + 1] = teams_after_acc_sort
    return sorted_teams


def table_creation(season, date, league):
    db_table = Table.create(season=season, date=date)
    teams_in_season = Team.select().join(TeamSeason).where(TeamSeason.season == season)
    table_teams = []
    for team_in_season in teams_in_season:
        team_in_table = TableTeam(team=team_in_season, table=db_table, points=0, loses=0, draws=0, wins=0,
                                  goals_scored=0, goals_conceded=0, matches_played=0, goal_difference=0)
        table_teams.append(team_in_table)
        team_matches = Match.select().where((Match.season == season) & (Match.date < date) &
                                            ((Match.home_team == team_in_season) | (Match.away_team == team_in_season)))
        for team_match in team_matches:
            if team_match.home_team == team_in_season:
                team_in_table.goals_scored += team_match.full_time_home_goals
                team_in_table.goals_conceded += team_match.full_time_away_goals
                team_in_table.goal_difference += team_match.full_time_home_goals - team_match.full_time_away_goals
                team_in_table.matches_played += 1
                if team_match.full_time_result == MatchResult.HOME_WIN:
                    team_in_table.points += 3
                    team_in_table.wins += 1
                elif team_match.full_time_result == MatchResult.DRAW:
                    team_in_table.points += 1
                    team_in_table.draws += 1
                elif team_match.full_time_result == MatchResult.AWAY_WIN:
                    team_in_table.loses += 1
            elif team_match.away_team == team_in_season:
                team_in_table.goals_scored += team_match.full_time_away_goals
                team_in_table.goals_conceded += team_match.full_time_home_goals
                team_in_table.goal_difference += team_match.full_time_away_goals - team_match.full_time_home_goals
                team_in_table.matches_played += 1
                if team_match.full_time_result == MatchResult.HOME_WIN:
                    team_in_table.loses += 1
                elif team_match.full_time_result == MatchResult.DRAW:
                    team_in_table.points += 1
                    team_in_table.draws += 1
                elif team_match.full_time_result == MatchResult.AWAY_WIN:
                    team_in_table.points += 3
                    team_in_table.wins += 1
    # teams_in_table_sorted = sorted(table_teams, key=lambda x: (x.points, x.goal_difference, x.goals_scored),
    #                                reverse=True)
    teams_in_table_sorted = sort_teams_in_table(table_teams, season, date, league)
    bulk_dictionary = []
    for index, sorted_team in enumerate(teams_in_table_sorted):
        sorted_team.position = index + 1
        bulk_dictionary.append(sorted_team.__data__)
    TableTeam.insert_many(bulk_dictionary).execute()


def process_csv_and_save_to_db(csv_file_path):
    matches_data = pd.read_csv(csv_file_path)
    db_league = None
    if matches_data["Div"].iloc[0] == "E0":
        db_league, is_league_created = League.get_or_create(league_name='Premier League',
                                                            defaults={'country': 'EN', 'division': 1})
    elif matches_data["Div"].iloc[0] == "D1":
        db_league, is_league_created = League.get_or_create(league_name='Bundesliga',
                                                            defaults={'country': 'DE', 'division': 1})
    elif matches_data["Div"].iloc[0] == "F1":
        db_league, is_league_created = League.get_or_create(league_name='Ligue 1',
                                                            defaults={'country': 'FR', 'division': 1})
    elif matches_data["Div"].iloc[0] == "SP1":
        db_league, is_league_created = League.get_or_create(league_name='La Liga',
                                                            defaults={'country': 'ES', 'division': 1})
    elif matches_data["Div"].iloc[0] == "I1":
        db_league, is_league_created = League.get_or_create(league_name='Serie A',
                                                            defaults={'country': 'IT', 'division': 1})

    dates = matches_data["Date"]
    if 'Time' not in matches_data:
        matches_data['Time'] = "00:00"
    times = matches_data["Time"]
    try:
        league_start_date = datetime.strptime(dates.iloc[0] + ' ' + times.iloc[0], "%d/%m/%Y %H:%M")
    except ValueError:
        league_start_date = datetime.strptime(dates.iloc[0] + ' ' + times.iloc[0], "%d/%m/%y %H:%M")
    try:
        league_end_date = datetime.strptime(dates.iloc[-1] + ' ' + times.iloc[-1], "%d/%m/%Y %H:%M")
    except ValueError:
        league_end_date = datetime.strptime(dates.iloc[-1] + ' ' + times.iloc[-1], "%d/%m/%y %H:%M")
    db_season, is_season_created = Season.get_or_create(league=db_league,
                                                        years=league_start_date.strftime(
                                                            "%y") + "/" + league_end_date.strftime("%y"),
                                                        defaults={'start_date': league_start_date,
                                                                  'end_date': league_end_date})

    if is_season_created:
        for team_name in matches_data["HomeTeam"].unique():
            team_tuple = Team.get_or_create(name=team_name)
            TeamSeason.create(team=team_tuple[0], season=db_season)

        matches_to_save = []
        for index, single_match_row in matches_data.iterrows():
            try:
                match_date = datetime.strptime(single_match_row["Date"] + ' ' + single_match_row["Time"], "%d/%m/%Y %H:%M")
            except:
                match_date = datetime.strptime(single_match_row["Date"] + ' ' + single_match_row["Time"],
                                               "%d/%m/%y %H:%M")
            matches_to_save.append({
                'date': match_date,
                'home_team': Team.get(Team.name == single_match_row["HomeTeam"]),
                'away_team': Team.get(Team.name == single_match_row["AwayTeam"]),
                'season': db_season,
                'full_time_home_goals': single_match_row["FTHG"],
                'full_time_away_goals': single_match_row["FTAG"],
                'full_time_result': MatchResult(single_match_row["FTR"]),
                'half_time_home_goals': single_match_row["HTHG"],
                'half_time_away_goals': single_match_row["HTAG"],
                'half_time_result': MatchResult(single_match_row["HTR"]),
                'home_team_shots': single_match_row["HS"],
                'home_team_shots_on_target': single_match_row["HST"],
                'home_team_woodwork_hits': single_match_row["HHW"] if 'HHW' in matches_data.columns else None,
                'home_team_corners': single_match_row["HC"],
                'home_team_fouls_committed': single_match_row["HF"],
                'home_team_free_kicks_conceded': single_match_row["HFKC"] if 'HFKC' in matches_data.columns else None,
                'home_team_offsides': single_match_row["HO"] if 'HO' in matches_data.columns else None,
                'home_team_yellow_cards': single_match_row["HY"],
                'home_team_red_cards': single_match_row["HR"],
                'away_team_shots': single_match_row["AS"],
                'away_team_shots_on_target': single_match_row["AST"],
                'away_team_woodwork_hits': single_match_row["AHW"] if 'AHW' in matches_data.columns else None,
                'away_team_corners': single_match_row["AC"],
                'away_team_fouls_committed': single_match_row["AF"],
                'away_team_free_kicks_conceded': single_match_row["AFKC"] if 'AFKC' in matches_data.columns else None,
                'away_team_offsides': single_match_row["AO"] if 'AO' in matches_data.columns else None,
                'away_team_yellow_cards': single_match_row["AY"],
                'away_team_red_cards': single_match_row["AR"],
                'average_home_odds': (
                    single_match_row["AvgH"] if 'AvgH' in matches_data.columns else single_match_row["BbAvH"]),
                'average_draw_odds': (
                    single_match_row["AvgD"] if 'AvgD' in matches_data.columns else single_match_row["BbAvD"]),
                'average_away_odds': (
                    single_match_row["AvgA"] if 'AvgA' in matches_data.columns else single_match_row["BbAvA"])})
        Match.insert_many(matches_to_save).execute()
        for matchDate in matches_data["Date"].unique():
            try:
                table_creation(db_season, datetime.strptime(matchDate, "%d/%m/%Y"), db_league)
            except ValueError:
                table_creation(db_season, datetime.strptime(matchDate, "%d/%m/%y"), db_league)
        # Table for the end of the season
        table_creation(db_season, league_end_date + timedelta(days=1), db_league)

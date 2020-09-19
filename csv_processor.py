import pandas as pd
from datetime import datetime, timedelta
from models import Season, Team, Match, League, TeamSeason, Table, TableTeam


def table_creation(season, date):
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
                if team_match.full_time_result == 'H':
                    team_in_table.points += 3
                    team_in_table.wins += 1
                elif team_match.full_time_result == 'D':
                    team_in_table.points += 1
                    team_in_table.draws += 1
                elif team_match.full_time_result == 'A':
                    team_in_table.loses += 1
            elif team_match.away_team == team_in_season:
                team_in_table.goals_scored += team_match.full_time_away_goals
                team_in_table.goals_conceded += team_match.full_time_home_goals
                team_in_table.goal_difference += team_match.full_time_away_goals - team_match.full_time_home_goals
                team_in_table.matches_played += 1
                if team_match.full_time_result == 'H':
                    team_in_table.loses += 1
                elif team_match.full_time_result == 'D':
                    team_in_table.points += 1
                    team_in_table.draws += 1
                elif team_match.full_time_result == 'A':
                    team_in_table.points += 3
                    team_in_table.wins += 1
    teams_in_table_sorted = sorted(table_teams, key=lambda x: (x.points, x.goal_difference, x.goals_scored), reverse=True)
    bulk_dictionary = []
    for index, sorted_team in enumerate(teams_in_table_sorted):
        sorted_team.position = index + 1
        bulk_dictionary.append(sorted_team.__data__)
    TableTeam.insert_many(bulk_dictionary).execute()


def process_csv_and_save_to_db(csv_file_path):
    matches_data = pd.read_csv(csv_file_path)
    if matches_data["Div"].iloc[0] == "E0":
        db_league, is_league_created = League.get_or_create(league_name='Premier League',
                                                            defaults={'country': 'EN', 'division': 1})

    dates = matches_data["Date"]
    if 'Time' not in matches_data:
        matches_data['Time'] = "00:00"
    times = matches_data["Time"]
    league_start_date = datetime.strptime(dates.iloc[0] + ' ' + times.iloc[0], "%d/%m/%Y %H:%M")
    league_end_date = datetime.strptime(dates.iloc[-1] + ' ' + times.iloc[-1], "%d/%m/%Y %H:%M")
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
            matches_to_save.append({
                'date': datetime.strptime(single_match_row["Date"] + ' ' + single_match_row["Time"], "%d/%m/%Y %H:%M"),
                'home_team': Team.get(Team.name == single_match_row["HomeTeam"]),
                'away_team': Team.get(Team.name == single_match_row["AwayTeam"]),
                'season': db_season,
                'full_time_home_goals': single_match_row["FTHG"],
                'full_time_away_goals': single_match_row["FTAG"],
                'full_time_result': single_match_row["FTR"],
                'half_time_home_goals': single_match_row["HTHG"],
                'half_time_away_goals': single_match_row["HTAG"],
                'half_time_result': single_match_row["HTR"],
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
                'average_home_odds': (single_match_row["AvgH"] if 'AvgH' in matches_data.columns else single_match_row["BbAvH"]),
                'average_draw_odds': (single_match_row["AvgD"] if 'AvgD' in matches_data.columns else single_match_row["BbAvD"]),
                'average_away_odds': (single_match_row["AvgA"] if 'AvgA' in matches_data.columns else single_match_row["BbAvA"])})
        Match.insert_many(matches_to_save).execute()
        for matchDate in matches_data["Date"].unique():
            table_creation(db_season, datetime.strptime(matchDate, "%d/%m/%Y"))
        # Table for the end of the season
        table_creation(db_season, league_end_date + timedelta(days=1))

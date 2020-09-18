import pandas as pd
from datetime import datetime, timedelta
from Models import Season, Team, Match, League, TeamSeason, Table, TableTeam


def tableCreation(season, date):
    table = Table.create(season=season, date=date)
    teamsInSeason = Team.select().join(TeamSeason).where(TeamSeason.season == season)
    tableTeams = []
    for team in teamsInSeason:
        tableTeam = TableTeam(team=team, table=table, points=0, loses=0, draws=0, wins=0, goalsScored=0,
                              goalsConceded=0, matchesPlayed=0)
        tableTeams.append(tableTeam)
        teamMatches = Match.select().where((Match.season == season) & (Match.date < date) &
                                           ((Match.homeTeam == team) | (Match.awayTeam == team)))
        for teamMatch in teamMatches:
            if teamMatch.homeTeam == team:
                tableTeam.goalsScored += teamMatch.fullTimeHomeGoals
                tableTeam.goalsConceded += teamMatch.fullTimeAwayGoals
                tableTeam.matchesPlayed += 1
                if teamMatch.fullTimeResult == 'H':
                    tableTeam.points += 3
                    tableTeam.wins += 1
                elif teamMatch.fullTimeResult == 'D':
                    tableTeam.points += 1
                    tableTeam.draws += 1
                elif teamMatch.fullTimeResult == 'A':
                    tableTeam.loses += 1
            elif teamMatch.awayTeam == team:
                tableTeam.goalsScored += teamMatch.fullTimeAwayGoals
                tableTeam.goalsConceded += teamMatch.fullTimeHomeGoals
                tableTeam.matchesPlayed += 1
                if teamMatch.fullTimeResult == 'H':
                    tableTeam.loses += 1
                elif teamMatch.fullTimeResult == 'D':
                    tableTeam.points += 1
                    tableTeam.draws += 1
                elif teamMatch.fullTimeResult == 'A':
                    tableTeam.points += 3
                    tableTeam.wins += 1
    sortedTableTeams = sorted(tableTeams, key=lambda x: x.points, reverse=True)
    bulkDictionary = []
    for index, sortedTeam in enumerate(sortedTableTeams):
        sortedTeam.position = index + 1
        bulkDictionary.append(sortedTeam.__data__)
    TableTeam.insert_many(bulkDictionary).execute()


def processCsvAndSaveToDb(csvFilePath):
    matchesData = pd.read_csv(csvFilePath)
    if matchesData["Div"].iloc[0] == "E0":
        premierLeague, isLeagueCreated = League.get_or_create(leagueName='Premier League',
                                                              defaults={'country': 'EN', 'division': 1})

    dates = matchesData["Date"]
    if 'Time' not in matchesData:
        matchesData['Time'] = "00:00"
    times = matchesData["Time"]
    leagueStartDate = datetime.strptime(dates.iloc[0] + ' ' + times.iloc[0], "%d/%m/%Y %H:%M")
    leagueEndDate = datetime.strptime(dates.iloc[-1] + ' ' + times.iloc[-1], "%d/%m/%Y %H:%M")
    season, isSeasonCreated = Season.get_or_create(league=premierLeague,
                                                   years=leagueStartDate.strftime("%y") + "/" + leagueEndDate.strftime(
                                                       "%y"),
                                                   defaults={'startDate': leagueStartDate,
                                                             'endDate': leagueEndDate})

    if isSeasonCreated:
        for team in matchesData["HomeTeam"].unique():
            teamTuple = Team.get_or_create(name=team)
            TeamSeason.create(team=teamTuple[0], season=season)

        matchesToSave = []
        for index, singleMatch in matchesData.iterrows():
            matchesToSave.append({
                'date': datetime.strptime(singleMatch["Date"] + ' ' + singleMatch["Time"], "%d/%m/%Y %H:%M"),
                'homeTeam': Team.get(Team.name == singleMatch["HomeTeam"]),
                'awayTeam': Team.get(Team.name == singleMatch["AwayTeam"]),
                'season': season,
                'fullTimeHomeGoals': singleMatch["FTHG"],
                'fullTimeAwayGoals': singleMatch["FTAG"],
                'fullTimeResult': singleMatch["FTR"],
                'halfTimeHomeGoals': singleMatch["HTHG"],
                'halfTimeAwayGoals': singleMatch["HTAG"],
                'halfTimeResult': singleMatch["HTR"],
                'homeTeamShots': singleMatch["HS"],
                'homeTeamShotsOnTarget': singleMatch["HST"],
                'homeTeamWoodworkHits': singleMatch["HHW"] if 'HHW' in matchesData.columns else None,
                'homeTeamCorners': singleMatch["HC"],
                'homeTeamFoulsCommitted': singleMatch["HF"],
                'homeTeamFreeKicksConceded': singleMatch["HFKC"] if 'HFKC' in matchesData.columns else None,
                'homeTeamOffsides': singleMatch["HO"] if 'HO' in matchesData.columns else None,
                'homeTeamYellowCards': singleMatch["HY"],
                'homeTeamRedCards': singleMatch["HR"],
                'awayTeamShots': singleMatch["AS"],
                'awayTeamShotsOnTarget': singleMatch["AST"],
                'awayTeamWoodworkHits': singleMatch["AHW"] if 'AHW' in matchesData.columns else None,
                'awayTeamCorners': singleMatch["AC"],
                'awayTeamFoulsCommitted': singleMatch["AF"],
                'awayTeamFreeKicksConceded': singleMatch["AFKC"] if 'AFKC' in matchesData.columns else None,
                'awayTeamOffsides': singleMatch["AO"] if 'AO' in matchesData.columns else None,
                'awayTeamYellowCards': singleMatch["AY"],
                'awayTeamRedCards': singleMatch["AR"],
                'averageHomeOdds': (singleMatch["AvgH"] if 'AvgH' in matchesData.columns else singleMatch["BbAvH"]),
                'averageDrawOdds': (singleMatch["AvgD"] if 'AvgD' in matchesData.columns else singleMatch["BbAvD"]),
                'averageAwayOdds': (singleMatch["AvgA"] if 'AvgA' in matchesData.columns else singleMatch["BbAvA"])})
            # actualMatch.save()
        Match.insert_many(matchesToSave).execute()
        for matchDate in matchesData["Date"].unique():
            tableCreation(season, datetime.strptime(matchDate, "%d/%m/%Y"))
        # Table for the end of the season
        tableCreation(season, leagueEndDate + timedelta(days=1))

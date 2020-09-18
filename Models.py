from peewee import *
from DatabaseHelper import db


class BaseModel(Model):
    class Meta:
        database = db


class Team(BaseModel):
    name = CharField(max_length=40, unique=True)


class League(BaseModel):
    leagueName = CharField(max_length=40, unique=True)  # models.CharField(max_length=40)
    country = CharField(max_length=20)  # models.CharField(max_length=20)
    division = IntegerField()  # models.IntegerField()


class Season(BaseModel):
    years = CharField(max_length=10)
    startDate = DateTimeField()
    endDate = DateTimeField()
    league = ForeignKeyField(League, on_delete='CASCADE')


class TeamSeason(BaseModel):
    team = ForeignKeyField(Team, on_delete='CASCADE')
    season = ForeignKeyField(Season, on_delete='CASCADE')


class Match(BaseModel):
    date = DateTimeField()
    homeTeam = ForeignKeyField(Team, on_delete='CASCADE')
    awayTeam = ForeignKeyField(Team, on_delete='CASCADE')
    season = ForeignKeyField(Season, on_delete='CASCADE')
    fullTimeHomeGoals = IntegerField()
    fullTimeAwayGoals = IntegerField()
    fullTimeResult = CharField()
    halfTimeHomeGoals = IntegerField()
    halfTimeAwayGoals = IntegerField()
    halfTimeResult = CharField()
    homeTeamShots = IntegerField()
    homeTeamShotsOnTarget = IntegerField()
    homeTeamWoodworkHits = IntegerField(null=True)
    homeTeamCorners = IntegerField()
    homeTeamFoulsCommitted = IntegerField()
    homeTeamFreeKicksConceded = IntegerField(null=True)
    homeTeamOffsides = IntegerField(null=True)
    homeTeamYellowCards = IntegerField()
    homeTeamRedCards = IntegerField()
    awayTeamShots = IntegerField()
    awayTeamShotsOnTarget = IntegerField()
    awayTeamWoodworkHits = IntegerField(null=True)
    awayTeamCorners = IntegerField()
    awayTeamFoulsCommitted = IntegerField()
    awayTeamFreeKicksConceded = IntegerField(null=True)
    awayTeamOffsides = IntegerField(null=True)
    awayTeamYellowCards = IntegerField()
    awayTeamRedCards = IntegerField()
    averageHomeOdds = FloatField()
    averageDrawOdds = FloatField()
    averageAwayOdds = FloatField()


class Table(BaseModel):
    season = ForeignKeyField(Season)
    date = DateField()


class TableTeam(BaseModel):
    team = ForeignKeyField(Team, on_delete='CASCADE')
    table = ForeignKeyField(Table, on_delete='CASCADE')
    matchesPlayed = IntegerField()
    goalsScored = IntegerField()
    goalsConceded = IntegerField()
    wins = IntegerField()
    draws = IntegerField()
    loses = IntegerField()
    points = IntegerField()
    position = IntegerField()

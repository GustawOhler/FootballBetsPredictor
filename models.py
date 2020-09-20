from enum import Enum
from typing import Any, Callable

from peewee import *
from database_helper import db


class EnumField(CharField):
    """
    This class enable an Enum like field for Peewee
    """

    def __init__(self, choices: Callable, *args: Any, **kwargs: Any) -> None:
        super(CharField, self).__init__(*args, **kwargs)
        self.choices = choices
        self.max_length = 255

    def db_value(self, value: Any) -> Any:
        return value.value

    def python_value(self, value: Any) -> Any:
        return self.choices(type(list(self.choices)[0].value)(value))


class BaseModel(Model):
    class Meta:
        database = db


class Team(BaseModel):
    name = CharField(max_length=40, unique=True)


class League(BaseModel):
    league_name = CharField(max_length=40, unique=True)
    country = CharField(max_length=20)
    division = IntegerField()


class Season(BaseModel):
    years = CharField(max_length=10)
    start_date = DateTimeField()
    end_date = DateTimeField()
    league = ForeignKeyField(League, on_delete='CASCADE')


class TeamSeason(BaseModel):
    team = ForeignKeyField(Team, on_delete='CASCADE')
    season = ForeignKeyField(Season, on_delete='CASCADE')


class MatchResult(Enum):
    HOME_WIN = 'H'
    DRAW = 'D'
    AWAY_WIN = 'A'


class Match(BaseModel):
    date = DateTimeField()
    home_team = ForeignKeyField(Team, on_delete='CASCADE')
    away_team = ForeignKeyField(Team, on_delete='CASCADE')
    season = ForeignKeyField(Season, on_delete='CASCADE')
    full_time_home_goals = IntegerField()
    full_time_away_goals = IntegerField()
    full_time_result = EnumField(choices=MatchResult)
    half_time_home_goals = IntegerField()
    half_time_away_goals = IntegerField()
    half_time_result = EnumField(choices=MatchResult)
    home_team_shots = IntegerField()
    home_team_shots_on_target = IntegerField()
    home_team_woodwork_hits = IntegerField(null=True)
    home_team_corners = IntegerField()
    home_team_fouls_committed = IntegerField()
    home_team_free_kicks_conceded = IntegerField(null=True)
    home_team_offsides = IntegerField(null=True)
    home_team_yellow_cards = IntegerField()
    home_team_red_cards = IntegerField()
    away_team_shots = IntegerField()
    away_team_shots_on_target = IntegerField()
    away_team_woodwork_hits = IntegerField(null=True)
    away_team_corners = IntegerField()
    away_team_fouls_committed = IntegerField()
    away_team_free_kicks_conceded = IntegerField(null=True)
    away_team_offsides = IntegerField(null=True)
    away_team_yellow_cards = IntegerField()
    away_team_red_cards = IntegerField()
    average_home_odds = FloatField()
    average_draw_odds = FloatField()
    average_away_odds = FloatField()


class Table(BaseModel):
    season = ForeignKeyField(Season)
    date = DateField()


class TableTeam(BaseModel):
    team = ForeignKeyField(Team, on_delete='CASCADE')
    table = ForeignKeyField(Table, on_delete='CASCADE')
    matches_played = IntegerField()
    goals_scored = IntegerField()
    goals_conceded = IntegerField()
    goal_difference = IntegerField()
    wins = IntegerField()
    draws = IntegerField()
    loses = IntegerField()
    points = IntegerField()
    position = IntegerField()

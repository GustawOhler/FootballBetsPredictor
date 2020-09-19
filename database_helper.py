import logging
from peewee import *

db = MySQLDatabase("FootballBetsPredictor", host="localhost", port=3306, user="root", passwd="root")


def setup_db(should_log, should_drop_tables):
    db.connect()
    if should_log:
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
    if should_drop_tables:
        from models import Season, Team, Match, League, TeamSeason, Table, TableTeam
        db.drop_tables([Season, Team, Match, League, TeamSeason, Table, TableTeam])
        db.create_tables([Season, Team, Match, League, TeamSeason, Table, TableTeam])
    return db

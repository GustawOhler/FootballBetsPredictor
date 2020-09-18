import logging

from peewee import *

NEED_TO_DROP_TABLES = True
SHOULD_LOG = False

db = MySQLDatabase("FootballBetsPredictor", host="localhost", port=3306, user="root", passwd="root")

from Models import *

def setupDB():
    db.connect()
    if SHOULD_LOG:
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
    if NEED_TO_DROP_TABLES:
        db.drop_tables([Season, Team, Match, League, TeamSeason, Table, TableTeam])
        db.create_tables([Season, Team, Match, League, TeamSeason, Table, TableTeam])
    return db

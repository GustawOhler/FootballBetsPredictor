from os import listdir
from os.path import isfile, join
from DatabaseHelper import db
from CsvProcessor import processCsvAndSaveToDb
from DatabaseHelper import setupDB


setupDB()
csvFolderPath = '.\\MatchesData'
onlyfiles = [join(csvFolderPath, f) for f in listdir(csvFolderPath) if isfile(join(csvFolderPath, f))]
for file in onlyfiles:
    print(file)
    processCsvAndSaveToDb(file)

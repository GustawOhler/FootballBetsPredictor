from os import listdir
from os.path import isfile, join
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_creator import create_dataset

NEED_TO_DROP_TABLES = False
SHOULD_LOG = False

setup_db(SHOULD_LOG, NEED_TO_DROP_TABLES)
csv_folder_path = '.\\MatchesData\\UsedData'
only_files_in_dir = [join(csv_folder_path, f) for f in listdir(csv_folder_path) if isfile(join(csv_folder_path, f))]
for file in only_files_in_dir:
    print(file)
    process_csv_and_save_to_db(file)

dataset = create_dataset()


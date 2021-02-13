from os import listdir
from os.path import isfile, join
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_creator import create_dataset, load_dataset
from neural_network_manager import perform_nn_learning

NEED_TO_DROP_TABLES = False
SHOULD_LOG = False
NEED_TO_CREATE_DATASET = False

setup_db(SHOULD_LOG, NEED_TO_DROP_TABLES)
csv_folder_path = '.\\MatchesData\\UsedData'
only_files_in_dir = [join(csv_folder_path, f) for f in listdir(csv_folder_path) if isfile(join(csv_folder_path, f))]
for file in only_files_in_dir:
    print(file)
    process_csv_and_save_to_db(file)

if NEED_TO_CREATE_DATASET:
    dataset = create_dataset()
else:
    dataset = load_dataset()

perform_nn_learning(dataset)

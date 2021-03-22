from os import listdir
from os.path import isfile, join
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_creator import create_dataset, load_dataset
from neural_network_manager import perform_nn_learning
import web_data_scraper
from timeit import default_timer as timer

NEED_TO_DROP_TABLES = False
SHOULD_LOG = False
NEED_TO_CREATE_DATASET = False
SHOULD_DOWNLOAD_DATA = False
CSV_FOLDER_PATH = '.\\MatchesData\\AutomatedDownloads'

setup_db(SHOULD_LOG, NEED_TO_DROP_TABLES)
if SHOULD_DOWNLOAD_DATA:
    web_data_scraper.download_data_from_web(CSV_FOLDER_PATH)
only_files_in_dir = [join(CSV_FOLDER_PATH, f) for f in listdir(CSV_FOLDER_PATH) if isfile(join(CSV_FOLDER_PATH, f))]
for file in only_files_in_dir:
    print(file)
    # start = timer()
    process_csv_and_save_to_db(file)
    # end = timer()
    # print("Execution time: " + str(end-start))

if NEED_TO_CREATE_DATASET:
    dataset = create_dataset()
else:
    dataset = load_dataset()

perform_nn_learning(dataset)

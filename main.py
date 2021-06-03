from os import listdir
from os.path import isfile, join
from constants import NEED_TO_DROP_TABLES, SHOULD_LOG, NEED_TO_CREATE_DATASET, SHOULD_DOWNLOAD_DATA, SHOULD_LOAD_MODEL_FROM_FILE, NEED_TO_PROCESS_CSV, \
    SHOULD_RUN_NN, SHOULD_CREATE_NEW_SPLIT, CSV_FOLDER_PATH, VALIDATION_TO_TRAIN_SPLIT_RATIO, curr_nn_manager_name
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_manager.dataset_manager import get_splitted_dataset
from nn_manager.nn_choose_bets_menager import NNChoosingBetsManager
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
from nn_manager.common import load_model
import web_data_scraper
from timeit import default_timer as timer
from nn_manager.nn_pred_matches_then_choose_bets_manager import NNChoosingBetsThenDevelopingStrategyManager
from nn_manager.recurrent_nn_choose_bets_manager import RecurrentNNChoosingBetsManager
from nn_manager.gru_nn_choose_bets_manager import GruNNChoosingBetsManager

setup_db(SHOULD_LOG, NEED_TO_DROP_TABLES)
if SHOULD_DOWNLOAD_DATA:
    web_data_scraper.download_data_from_web(CSV_FOLDER_PATH)
if NEED_TO_PROCESS_CSV:
    match_csv_filepaths = [join(CSV_FOLDER_PATH, f) for f in listdir(CSV_FOLDER_PATH) if isfile(join(CSV_FOLDER_PATH, f))]
    for file in match_csv_filepaths:
        print(file)
        start = timer()
        process_csv_and_save_to_db(file)
        end = timer()
        print("Execution time: " + str(end-start))

(x_train, y_train), (x_val, y_val) = get_splitted_dataset(NEED_TO_CREATE_DATASET, SHOULD_CREATE_NEW_SPLIT, VALIDATION_TO_TRAIN_SPLIT_RATIO)

if SHOULD_RUN_NN:
    curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val))
    if SHOULD_LOAD_MODEL_FROM_FILE:
        curr_nn_manager.model = load_model(curr_nn_manager.get_path_for_saving_model())
    else:
        curr_nn_manager.perform_model_learning()
        curr_nn_manager.evaluate_model()

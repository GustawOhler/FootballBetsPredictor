import datetime
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import KFold
from constants import NEED_TO_DROP_TABLES, SHOULD_LOG, NEED_TO_CREATE_DATASET, SHOULD_DOWNLOAD_DATA, NEED_TO_PROCESS_CSV, \
    SHOULD_CREATE_NEW_SPLIT, CSV_FOLDER_PATH, VALIDATION_TO_TRAIN_SPLIT_RATIO, curr_nn_manager_name, is_model_rnn, \
    TEST_TO_VALIDATION_SPLIT_RATIO, curr_dataset,CURRENT_NN_RUN_TYPE, NNRunType
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.dataset_manager import get_splitted_dataset, get_whole_dataset, get_already_splitted_raw_dataset
from nn_manager.best_model_researcher import BestModelResearcher
from nn_manager.k_fold_validator import perform_k_fold_with_different_parameters, perform_standard_k_fold, perform_k_fold_with_different_datasets, \
    perform_k_fold_with_different_models, print_results_to_csv, perform_k_fold_on_expotential, perform_k_fold_for_different_strategies, \
    perform_k_fold_on_last_2, perform_test_check_on_last_2, search_for_best_configuration
from nn_manager.nn_choose_bets_menager import NNChoosingBetsManager
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
from nn_manager.common import load_model, plot_profit_for_thesis
import web_data_scraper
from timeit import default_timer as timer
from nn_manager.recurrent_nn_choose_bets_manager import RecurrentNNChoosingBetsManager
from nn_manager.recurrent_nn_pred_matches_manager import RecurrentNNPredictingMatchesManager



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

datasets = get_splitted_dataset(NEED_TO_CREATE_DATASET, SHOULD_CREATE_NEW_SPLIT,
                                                              VALIDATION_TO_TRAIN_SPLIT_RATIO, TEST_TO_VALIDATION_SPLIT_RATIO)
(x_train, y_train) = datasets[0]
(x_val, y_val) = datasets[1]
test_set = datasets[2] if TEST_TO_VALIDATION_SPLIT_RATIO > 0 else None

if CURRENT_NN_RUN_TYPE == NNRunType.KFold:
    search_for_best_configuration(datasets)
elif CURRENT_NN_RUN_TYPE == NNRunType.Research:
    curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), False, test_set, True)
    researcher = BestModelResearcher(curr_nn_manager.model, get_already_splitted_raw_dataset(DatasetSplit.TEST), test_set)
    researcher.perform_full_research()
elif CURRENT_NN_RUN_TYPE == NNRunType.BestModelEvaluation:
    curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), False, test_set, True)
    curr_nn_manager.evaluate_model(False)
elif CURRENT_NN_RUN_TYPE == NNRunType.HyperTuning:
    curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), True, test_set, False)
    tuned = curr_nn_manager.hyper_tune_model()
elif CURRENT_NN_RUN_TYPE == NNRunType.Learning:
    curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), False, test_set, False)
    curr_nn_manager.perform_model_learning(verbose=True)
    curr_nn_manager.evaluate_model()
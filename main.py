from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import KFold
from constants import NEED_TO_DROP_TABLES, SHOULD_LOG, NEED_TO_CREATE_DATASET, SHOULD_DOWNLOAD_DATA, SHOULD_LOAD_MODEL_FROM_FILE, NEED_TO_PROCESS_CSV, \
    SHOULD_RUN_NN, SHOULD_CREATE_NEW_SPLIT, CSV_FOLDER_PATH, VALIDATION_TO_TRAIN_SPLIT_RATIO, curr_nn_manager_name, PERFORM_K_FOLD, is_model_rnn, \
    SHOULD_HYPERTUNE, TEST_TO_VALIDATION_SPLIT_RATIO, curr_dataset
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_manager.dataset_manager import get_splitted_dataset, get_whole_dataset
from nn_manager.k_fold_validator import perform_k_fold_with_different_parameters, perform_standard_k_fold, perform_k_fold_with_different_datasets, \
    perform_k_fold_with_different_models, print_results_to_csv, perform_k_fold_on_expotential, perform_k_fold_for_different_strategies
from nn_manager.nn_choose_bets_menager import NNChoosingBetsManager
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
from nn_manager.common import load_model
import web_data_scraper
from timeit import default_timer as timer
from nn_manager.recurrent_nn_choose_bets_manager import RecurrentNNChoosingBetsManager
from nn_manager.gru_nn_choose_bets_manager import GruNNChoosingBetsManager
from nn_manager.lstm_nn_choose_bets_manager import LstmNNChoosingBetsManager
from nn_manager.gru_nn_pred_matches_manager import GruNNPredictingMatchesManager
from nn_manager.lstm_nn_pred_matches_manager import LstmNNPredictingMatchesManager
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

if PERFORM_K_FOLD:
    # X, y = get_whole_dataset(NEED_TO_CREATE_DATASET)
    # perform_standard_k_fold(X, y, globals()[curr_nn_manager_name])
    # tracked_metrics = perform_k_fold_with_different_datasets(globals()[curr_nn_manager_name])
    # tracked_metrics = perform_k_fold_with_different_models()
    # tracked_metrics = perform_k_fold_on_expotential(globals()[curr_nn_manager_name], get_whole_dataset(False))
    tracked_metrics = perform_k_fold_for_different_strategies(globals()[curr_nn_manager_name], get_whole_dataset(False), is_model_rnn, False)
    print_results_to_csv(tracked_metrics, 'final_results/strategy_comparison_predicting_matches.csv')
else:
    datasets = get_splitted_dataset(NEED_TO_CREATE_DATASET, SHOULD_CREATE_NEW_SPLIT,
                                                              VALIDATION_TO_TRAIN_SPLIT_RATIO, TEST_TO_VALIDATION_SPLIT_RATIO)
    (x_train, y_train) = datasets[0]
    (x_val, y_val) = datasets[1]
    test_set = datasets[2] if TEST_TO_VALIDATION_SPLIT_RATIO > 0 else None
    if SHOULD_RUN_NN:
        curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), SHOULD_HYPERTUNE, test_set)
        if SHOULD_LOAD_MODEL_FROM_FILE:
            curr_nn_manager.model = load_model(curr_nn_manager.get_path_for_saving_model())
        elif SHOULD_HYPERTUNE:
            tuned = curr_nn_manager.hyper_tune_model()
        else:
            curr_nn_manager.perform_model_learning(verbose=True)
            curr_nn_manager.evaluate_model()
            # curr_nn_manager.get_best_strategies_value()
            # curr_nn_manager.model.evaluate(test_set[0], test_set[1], batch_size=test_set[1].shape[0])

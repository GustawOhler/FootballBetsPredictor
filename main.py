from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.model_selection import KFold
from constants import NEED_TO_DROP_TABLES, SHOULD_LOG, NEED_TO_CREATE_DATASET, SHOULD_DOWNLOAD_DATA, SHOULD_LOAD_MODEL_FROM_FILE, NEED_TO_PROCESS_CSV, \
    SHOULD_RUN_NN, SHOULD_CREATE_NEW_SPLIT, CSV_FOLDER_PATH, VALIDATION_TO_TRAIN_SPLIT_RATIO, curr_nn_manager_name, PERFORM_K_FOLD, is_model_rnn, \
    SHOULD_HYPERTUNE
from csv_processor import process_csv_and_save_to_db
from database_helper import setup_db
from dataset_manager.dataset_manager import get_splitted_dataset, get_whole_dataset
from nn_manager.nn_choose_bets_menager import NNChoosingBetsManager
from nn_manager.nn_pred_matches_manager import NNPredictingMatchesManager
from nn_manager.common import load_model
import web_data_scraper
from timeit import default_timer as timer
from nn_manager.nn_pred_matches_then_choose_bets_manager import NNChoosingBetsThenDevelopingStrategyManager
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
    X, y = get_whole_dataset(NEED_TO_CREATE_DATASET)
    k_folder = KFold(n_splits=10, shuffle=True)
    metrics = []
    metrics_names = []
    loop_index = 1
    for train_index, val_index in k_folder.split(y):
        print("Rozpoczynam uczenie modelu nr " + str(loop_index), end="\r")
        loop_index += 1
        if is_model_rnn:
            X_train, X_val = [X[i][train_index] for i in range(len(X))], [X[i][val_index] for i in range(len(X))]
        else:
            X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        curr_nn_manager = (globals()[curr_nn_manager_name])((X_train, y_train), (X_val, y_val), False)
        curr_nn_manager.perform_model_learning(verbose=False)
        metrics.append(curr_nn_manager.model.evaluate(X_val, y_val, verbose=1))
        if len(metrics_names) == 0:
            metrics_names = curr_nn_manager.model.metrics_names
    mean_metrics = np.asarray(metrics).mean(axis=0)
    print("Srednie wyniki dla modelu: ")
    for i, name in enumerate(metrics_names):
        print(name + ": " + str(mean_metrics[i]))
else:
    (x_train, y_train), (x_val, y_val) = get_splitted_dataset(NEED_TO_CREATE_DATASET, SHOULD_CREATE_NEW_SPLIT, VALIDATION_TO_TRAIN_SPLIT_RATIO)
    if SHOULD_RUN_NN:
        curr_nn_manager = (globals()[curr_nn_manager_name])((x_train, y_train), (x_val, y_val), SHOULD_HYPERTUNE)
        if SHOULD_LOAD_MODEL_FROM_FILE:
            curr_nn_manager.model = load_model(curr_nn_manager.get_path_for_saving_model())
        elif SHOULD_HYPERTUNE:
            tuned = curr_nn_manager.hyper_tune_model()
        else:
            curr_nn_manager.perform_model_learning()
            curr_nn_manager.evaluate_model()

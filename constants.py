# region dataset_manager
from enum import Enum

class DatasetType(Enum):
    BASIC = 'BasicDatasetCreator'
    AGGREGATED_MATCHES = 'DatasetWithAggregatedMatchesCreator'
    SEPARATED_MATCHES = 'DatasetWithSeparatedMatchesCreator'

class ModelType(Enum):
    PREDICTING_MATCHES = 'NNPredictingMatchesManager'
    CHOOSING_BETS = 'NNChoosingBetsManager'
    PRED_MATCH_THEN_CHOOSE_BETS = 'NNChoosingBetsThenDevelopingStrategyManager'
    RNN = 'RecurrentNNChoosingBetsManager'
    GRU = 'GruNNChoosingBetsManager'
    LSTM = 'LstmNNChoosingBetsManager'

ids_path = 'dataset_manager/datasets/match_ids'
base_dataset_path = 'dataset_manager/datasets/'
curr_dataset_name = DatasetType.SEPARATED_MATCHES.value
dataset_path = 'dataset_manager/datasets/' + curr_dataset_name
dataset_ext = '.csv'
dataset_with_ext = dataset_path + dataset_ext
# endregion
# region neural_network_manager
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
saved_model_based_path = "./NN_full_model/"
saved_model_weights_base_path = "./NN_model_weights/"
confidence_threshold = 0.03
results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
curr_nn_manager_name = ModelType.GRU.value
is_model_rnn = curr_nn_manager_name in [ModelType.RNN.value, ModelType.GRU.value, ModelType.LSTM.value]
# endregion
# region main
NEED_TO_DROP_TABLES = False
SHOULD_LOG = False
NEED_TO_CREATE_DATASET = False
SHOULD_DOWNLOAD_DATA = False
SHOULD_LOAD_MODEL_FROM_FILE = False
NEED_TO_PROCESS_CSV = False
SHOULD_RUN_NN = True
SHOULD_CREATE_NEW_SPLIT = False
SPLIT_MATCHES_BY_QUERY = False
PERFORM_K_FOLD = True
TAKE_MATCHES_FROM_QUERY = False
CSV_FOLDER_PATH = '.\\MatchesData\\AutomatedDownloads'
VALIDATION_TO_TRAIN_SPLIT_RATIO = 0.1
# endregion
SHOULD_DROP_ODDS_FROM_DATASET = False

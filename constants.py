# region dataset_manager
from enum import Enum, IntEnum


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
    GRU_pred_matches = 'GruNNPredictingMatchesManager'
    RNN_pred_matches = 'RecurrentNNPredictingMatchesManager'
    LSTM_pred_matches = 'LstmNNPredictingMatchesManager'


class ChoosingBetsStrategy(Enum):
    AllOnBestResult = 'AllOnBestResult'
    BetOnBestResultWithRetProb = 'BetOnBestResultWithRetProb'
    MalafosseUnlessNoBet = 'MalafosseUnlessNoBet'
    OriginalMalafosse = 'OriginalMalafosse'


class PredMatchesStrategy(Enum):
    AllOnBestOverThreshold = 'AllOnBestOverThreshold'
    AllOnBiggestDifferenceOverThreshold = 'AllOnBiggestDifferenceOverThreshold'
    RelativeOnBestOverThreshold = 'RelativeOnBestOverThreshold'
    RelativeOnBiggestDifferenceOverThreshold = 'RelativeOnBiggestDifferenceOverThreshold'
    RelativeOnResultsOverThreshold = 'RelativeOnResultsOverThreshold'
    KellyCriterion = 'KellyCriterion'


ids_path = 'dataset_manager/datasets/match_ids'
base_dataset_path = 'dataset_manager/datasets/'
curr_dataset = DatasetType.SEPARATED_MATCHES
curr_dataset_name = curr_dataset.value
dataset_path = 'dataset_manager/datasets/' + curr_dataset_name
dataset_ext = '.csv'
dataset_with_ext = dataset_path + dataset_ext
# endregion
# region neural_network_manager
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# if not "KERASTUNER_TUNER_ID" in os.environ:
#     os.environ["KERASTUNER_TUNER_ID"] = "chief"
# os.environ["KERASTUNER_ORACLE_IP"] = "127.0.0.1"
# os.environ["KERASTUNER_ORACLE_PORT"] = "8000"
saved_model_based_path = "./NN_full_model/"
saved_model_weights_base_path = "./NN_model_weights/"
# confidence_threshold = 0.03
results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
curr_nn_manager_name = ModelType.RNN_pred_matches.value
is_model_rnn = curr_nn_manager_name in [ModelType.RNN.value, ModelType.GRU.value, ModelType.LSTM.value, ModelType.GRU_pred_matches.value,
                                        ModelType.RNN_pred_matches.value, ModelType.LSTM_pred_matches.value]
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
PERFORM_K_FOLD = False
TAKE_MATCHES_FROM_QUERY = True
SHOULD_HYPERTUNE = False
CSV_FOLDER_PATH = '.\\MatchesData\\AutomatedDownloads'
VALIDATION_TO_TRAIN_SPLIT_RATIO = 0.2
TEST_TO_VALIDATION_SPLIT_RATIO = 0.5
# endregion
SHOULD_DROP_ODDS_FROM_DATASET = False

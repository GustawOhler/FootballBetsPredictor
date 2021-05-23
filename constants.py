# region dataset_manager
ids_path = 'dataset_manager/datasets/match_ids'
dataset_path = 'dataset_manager/datasets/dataset_ver_3'
dataset_with_ext = dataset_path + '.csv'
# endregion
# region neural_network_manager
saved_model_location = "./NN_full_model/"
saved_weights_location = "./NN_model_weights/checkpoint_weights"
confidence_threshold = 0.03
results_to_description_dict = {0: 'Wygrana gospodarzy', 1: 'Remis', 2: 'Wygrana gości', 3: 'Brak zakładu'}
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
CSV_FOLDER_PATH = '.\\MatchesData\\AutomatedDownloads'
VALIDATION_TO_TRAIN_SPLIT_RATIO = 0.125
# endregion
SHOULD_DROP_ODDS_FROM_DATASET = False
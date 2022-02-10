import os

# Data constants.
DATA_PATH = os.path.abspath("../data/train-test")
DATA_TRAIN = "train_data"
DATA_TEST = "test_data"
DATA_EXT = "csv"
LABEL = "calls"
WB_SIZE = 1

# Model constants.
NN_MODEL_SAVE_PATH = os.path.abspath("../models/nn")
NN_MODEL_SAVE_NAME = "nn_lstm_trained"

SK_MODEL_SAVE_PATH = os.path.abspath("../models/ml") 
SK_MODEL_SAVE_NAME = "ml_model_trained"
SK_MODEL_SAVE_EXT = "pkl"

# Plot constants.
FIG_EXT = "png"
TRAIN_FIG_SAVE_PATH = os.path.abspath("../plots/training")
TEST_FIG_SAVE_PATH = os.path.abspath("../plots/testing")
TRAIN_FIG_SAVE_NAME = "learning_curve"

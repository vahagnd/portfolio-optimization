import os

import pandas as pd
from dotenv import load_dotenv

from data.preprocessing import Data

load_dotenv()


SEED = int(os.getenv("SEED", 42))
MAG7_DATA_CSV_PATH = os.getenv("MAGF7_DATA_CSV_PATH")
LATEST_MODEL_PATH = os.getenv("LATEST_MODEL_PATH")

PRICES = "close"  # Options: 'open', 'high', 'low', 'close'
YEARS = [2018, 2019, 2020]  # Years to split data
NORMALIZE = True  # Normalize data
SCALER = "StandardScaler"  # Options: 'MinMaxScaler', 'StandardScaler'

DF_MAG7_RAW = pd.read_csv(MAG7_DATA_CSV_PATH)

PREPROCESS_KWARGS = {
    "years": YEARS,
    "prices": PRICES,
    "normalize": NORMALIZE,
    "scaler": SCALER,
}

BATCH_SIZE = 8
SHARPE_WINDOW = 25
TIME_WINDOW = 32
STOCK_COUNT = Data(DF_MAG7_RAW).get_test_dataframes(years=YEARS)[0].shape[1]
FEATURE_COUNT = 32
HIDDEN_SIZE = 64
NUM_EPOCHS = 200

MODEL_NAME_LSTM = "single_layer_lstm_on_mag7"
MODEL_NAME_FC = "single_layer_fc_on_mag7"
NOTES_FC = "fp-relu-mp-fc-relu-fc-relu-fc-relu-fc-sm"
NOTES_LSTM = "fp-relu-lstm-fc-sm"

LR = 1e-4
MOMENTUM = 0.99
WEIGHT_DECAY = 1e-5
OPTIMIZE_TYPE = "SGD"  # Options: "Adam", "SGD"
LOSS_FUNCTION = "SharpeRatioLoss"  # Options: "SharpeRatioLoss"

import torch
import logging
from data.preprocessing import Data
import pandas as pd
import random
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d - %(message)s"
)
logging.getLogger('matplotlib.font_manager').disabled = True # This fucker floods logs

logger = logging.getLogger(__name__)

SEED = 42
def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PRICES = 'close'  # Options: 'open', 'high', 'low', 'close'
YEARS = [2018, 2019, 2020]  # Years to split data
NORMALIZE = True  # Normalize data
SCALER = 'StandardScaler'  # Options: 'MinMaxScaler', 'StandardScaler'

DF_MAG7_RAW = pd.read_csv("data-csv/ohlcv_adjusted_mag7.csv")

PREPROCESS_KWARGS = {
    'years': YEARS,
    'prices': PRICES,
    'normalize': NORMALIZE,
    'scaler': SCALER,
}

def inspect_dataloader(dataloader, name="Train"):
    for x, y in dataloader:
        logger.debug(f"{name} batch x shape: {x.shape}")
        logger.debug(f"{name} batch y shape: {y.shape}")
        break
    logger.debug(f"Length of {name} dataloader: {len(dataloader)}")

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

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda is for nvidia GPUs
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # mps is for macos

LR = 1e-4
MOMENTUM = 0.99
WEIGHT_DECAY = 1e-5
OPTIMIZE_TYPE = "SGD"  # Options: "Adam", "SGD"
LOSS_FUNCTION = "SharpeRatioLoss"  # Options: "SharpeRatioLoss"

LATEST_MODEL_PATH = "saved/model_latest"


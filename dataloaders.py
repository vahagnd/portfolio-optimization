from config import *
import pandas as pd
from dataset import StockDataset
from data.preprocessing import Data
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

def get_dataloaders(df_raw: pd.DataFrame, years=[2018, 2019, 2020], prices='close', normalize=True, scaler='StandardScaler'):
    stock_data = Data(df_raw)

    (
        train_prices,
        train_returns,
        val_prices,
        val_returns,
        test_prices,
        test_returns
    ) = stock_data.preprocess(years=years, prices=prices, normalize=normalize, scaler=scaler)

    train_dataset = StockDataset(train_returns, train_prices, window_size=TIME_WINDOW, sharpe_window=SHARPE_WINDOW)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch_data, batch_returns in train_dataloader:
        logger.info(f"Train batch data shape: {batch_data.shape}")
        logger.info(f"Train batch returns shape: {batch_returns.shape}")
        break
    logger.info(f"Length of train dataloader: {len(train_dataloader)}")

    val_dataset = StockDataset(val_returns, val_prices, window_size=TIME_WINDOW, sharpe_window=SHARPE_WINDOW)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for batch_data, batch_returns in val_dataloader:
        logger.info(f"Validation batch data shape: {batch_data.shape}")
        logger.info(f"Train batch returns shape: {batch_returns.shape}")
        break
    logger.info(f"Length of validation dataloader: {len(val_dataloader)}")

    return train_dataloader, val_dataloader
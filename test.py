from models import sharpe_ratio_loss, SharpeLSTMModel, SharpeFCModel
from data.preprocessing import Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from config import (
    SEED, DEVICE, HIDDEN_SIZE, FEATURE_COUNT, TIME_WINDOW, STOCK_COUNT,
    MODEL_NAME_LSTM, MODEL_NAME_FC, LATEST_MODEL_PATH, fix_seed, PREPROCESS_KWARGS,
    DF_MAG7_RAW
)

logger = logging.getLogger(__name__)

def test_all(
    model: SharpeLSTMModel,
    model_fc: SharpeFCModel,
    test_prices: pd.DataFrame,
    test_returns: pd.DataFrame,
    T: int = 32,
    S: int = 7,
    device: torch.device = DEVICE
):
    logger.info("Testing models...")
    cumulative_returns_test = [1]
    cumulative_returns_equal = [1] # equal
    cumulative_returns_fc = [1] # fc

    test_predictions = []
    benchmark_returns = []
    test_predictions_fc = []

    testing_data = torch.stack([torch.tensor(test_prices.values, dtype=torch.float32),
                                torch.tensor(test_returns.values, dtype=torch.float32)], axis=2).to(device) # (days, S, 2)
    testing_returns = torch.tensor(test_returns.values, dtype=torch.float32).to(device)

    logger.debug(f"testing_data.shape: {testing_data.shape}, testing_returns.shape: {testing_returns.shape}")
    model.eval()
    with torch.no_grad():
        for t in range(T, testing_data.shape[0]):
            weights = model(testing_data[t - T:t].reshape(1, 1, T, S, 2)) # (T, S, 2) -> (S) LSTM
            weights_equal = (torch.ones(S) / S).to(device) # equal
            weights_fc = model_fc(testing_data[t - T:t].reshape(1, 1, T, S, 2)) # (T, S, 2) -> (S) FC

            next_day_return = weights @ testing_returns[t] # lstm
            next_day_return_equal = weights_equal @ testing_returns[t] # equal
            next_day_return_fc = weights_fc @ testing_returns[t] # fc

            cumulative_returns_test.append(cumulative_returns_test[-1] * (1 + next_day_return))
            cumulative_returns_equal.append(cumulative_returns_equal[-1] * (1 + next_day_return_equal)) # equal
            cumulative_returns_fc.append(cumulative_returns_fc[-1] * (1 + next_day_return_fc)) # fc

            test_predictions.append(next_day_return.cpu())
            benchmark_returns.append(next_day_return_equal.cpu())
            test_predictions_fc.append(next_day_return_fc.cpu())


    plt.figure(figsize=(10, 6))
    plt.plot(test_prices.index.values[T - 1:], np.array([r.cpu().item() if r != 1 else r for r in cumulative_returns_test]),
            label=MODEL_NAME_LSTM, color='blue')
    plt.plot(test_prices.index.values[T - 1:], np.array([r.cpu() if r != 1 else r for r in cumulative_returns_equal]),
            label='benchmark_equal_weights', color='red') # equal
    plt.plot(test_prices.index.values[T - 1:], np.array([r.cpu().item() if r != 1 else r for r in cumulative_returns_fc]),
            label=MODEL_NAME_FC, color='orange')
    # plt.plot(test_prices.index.values[T - 1:], portfolio_value_markowitz,
    #         label='Markowitz', color='green') # markowitz
    plt.title("Cumulative Return Over Time On Test Data")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{LATEST_MODEL_PATH}/test_cum_ret")
    plt.close()
    logger.info("Saved test cumulative returns plot.")


if __name__ == "__main__":
    logger.warning("test.py executed directly")

    model = SharpeLSTMModel(
                num_classes=1,
                input_size=2,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                feature_size=FEATURE_COUNT
            ).to(DEVICE)
    model_fc = SharpeFCModel(
                input_size=2,
                hidden_size=HIDDEN_SIZE,
                feature_size=FEATURE_COUNT
            ).to(DEVICE)

    model.load_state_dict(torch.load(f"{LATEST_MODEL_PATH}/weights_lstm.pth"))
    model_fc.load_state_dict(torch.load(f"{LATEST_MODEL_PATH}/weights_fc.pth"))

    stock_data = Data(DF_MAG7_RAW)

    test_prices, test_returns = stock_data.get_test_dataframes(
        **PREPROCESS_KWARGS
    )
    fix_seed(SEED)

    test_all(
        test_prices=test_prices,
        test_returns=test_returns,
        model=model,
        model_fc=model_fc,
        T=TIME_WINDOW,
        S=STOCK_COUNT,
        device=DEVICE
        )
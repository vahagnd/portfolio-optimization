from models import SharpeLSTMModel, SharpeFCModel
from data.preprocessing import Data
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from config import (
    DEVICE, HIDDEN_SIZE, FEATURE_COUNT, TIME_WINDOW, STOCK_COUNT, DF_MAG7_RAW,
    MODEL_NAME_LSTM, MODEL_NAME_FC, LATEST_MODEL_PATH, PREPROCESS_KWARGS
)

logger = logging.getLogger(__name__)

class Tester:
    def __init__(
        self,
        model_lstm: SharpeLSTMModel,
        model_fc: SharpeFCModel,
        test_prices: pd.DataFrame,
        test_returns: pd.DataFrame,
        cumulative_returns_markowitz: np.ndarray = None,
        markowitz_returns: np.ndarray = None,
        T: int = 32,
        S: int = 7,
        device: torch.device = torch.device("mps")
    ):
        self.model_lstm = model_lstm
        self.model_fc = model_fc
        self.test_prices = test_prices
        self.test_returns = test_returns
        self.cumulative_returns_markowitz = cumulative_returns_markowitz
        self.markowitz_returns = markowitz_returns
        self.T = T
        self.S = S
        self.device = device

    @staticmethod
    def compute_metrics(returns, cumulative_returns, risk_free_rate=0.0):
        expected_return = (cumulative_returns[-1] ** (252 / len(cumulative_returns)) - 1).item()
        std_dev = np.std(returns) * np.sqrt(252)
        downside_deviation = np.std(returns[returns < risk_free_rate]) * np.sqrt(252)
        sharpe_ratio = (expected_return - risk_free_rate) / std_dev if std_dev != 0 else np.nan
        sortino_ratio = (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
        cumulative_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        max_drawdown = np.min(drawdown)

        metrics = {
            "Expected Return": expected_return,
            "STD": std_dev,
            "Downside Deviation": downside_deviation,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Maximum Drawdown": max_drawdown
        }

        return metrics

    def run(self):
        logger.info("Testing models...")
        cumulative_returns_test = [1]
        cumulative_returns_equal = [1]
        cumulative_returns_fc = [1]
        cumulative_returns_markowitz = self.cumulative_returns_markowitz

        test_predictions = []
        benchmark_returns = []
        test_predictions_fc = []
        markowitz_returns = self.markowitz_returns

        testing_data = torch.stack([torch.tensor(self.test_prices.values, dtype=torch.float32),
                                    torch.tensor(self.test_returns.values, dtype=torch.float32)], axis=2).to(self.device)
        testing_returns = torch.tensor(self.test_returns.values, dtype=torch.float32).to(self.device)

        logger.debug(f"testing_data.shape: {testing_data.shape}, testing_returns.shape: {testing_returns.shape}")
        self.model_lstm.eval()
        self.model_fc.eval()
        with torch.no_grad():
            for t in range(self.T, testing_data.shape[0]):
                weights = self.model_lstm(testing_data[t - self.T:t].reshape(1, 1, self.T, self.S, 2))
                weights_equal = (torch.ones(self.S) / self.S).to(self.device)
                weights_fc = self.model_fc(testing_data[t - self.T:t].reshape(1, 1, self.T, self.S, 2))

                next_day_return = weights @ testing_returns[t]
                next_day_return_equal = weights_equal @ testing_returns[t]
                next_day_return_fc = weights_fc @ testing_returns[t]

                cumulative_returns_test.append(cumulative_returns_test[-1] * (1 + next_day_return))
                cumulative_returns_equal.append(cumulative_returns_equal[-1] * (1 + next_day_return_equal))
                cumulative_returns_fc.append(cumulative_returns_fc[-1] * (1 + next_day_return_fc))

                test_predictions.append(next_day_return.cpu())
                benchmark_returns.append(next_day_return_equal.cpu())
                test_predictions_fc.append(next_day_return_fc.cpu())

        plt.figure(figsize=(10, 6))
        plt.plot(self.test_prices.index.values[self.T - 1:], np.array([r.cpu().item() if r != 1 else r for r in cumulative_returns_test]),
                 label=MODEL_NAME_LSTM, color='blue')
        plt.plot(self.test_prices.index.values[self.T - 1:], np.array([r.cpu() if r != 1 else r for r in cumulative_returns_equal]),
                 label='benchmark_equal_weights', color='red')
        plt.plot(self.test_prices.index.values[self.T - 1:], np.array([r.cpu().item() if r != 1 else r for r in cumulative_returns_fc]),
                 label=MODEL_NAME_FC, color='orange')
        plt.plot(self.test_prices.index.values[self.T - 1:], cumulative_returns_markowitz,
                 label='rolling_markowitz', color='green')
        plt.title("Cumulative Return Over Time On Test Data")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{LATEST_MODEL_PATH}/plots/test_cum_ret")
        plt.close()
        logger.info("Saved test cumulative returns plot.")

        test_predictions = np.array(test_predictions)
        benchmark_returns = np.array(benchmark_returns)
        markowitz_returns = np.array(markowitz_returns)
        test_predctions_fc = np.array(test_predictions_fc)

        metrics_model = self.compute_metrics(test_predictions, cumulative_returns_test)
        metrics_benchmark = self.compute_metrics(benchmark_returns, cumulative_returns_equal)
        metrics_markowitz = self.compute_metrics(markowitz_returns, cumulative_returns_markowitz)
        metrics_fc = self.compute_metrics(test_predctions_fc, cumulative_returns_fc)

        metrics_df = pd.DataFrame({
            "LSTM": metrics_model,
            "Benchmark": metrics_benchmark,
            "Markowitz": metrics_markowitz,
            "FC": metrics_fc
        }).round(4)

        _, ax = plt.subplots(figsize=(6, 2))
        ax.axis('off')

        tbl = ax.table(
            cellText=metrics_df.values,
            rowLabels=metrics_df.index,
            colLabels=metrics_df.columns,
            loc='center',
            cellLoc='center'
        )

        tbl.scale(1, 1.5)

        for (row, _), cell in tbl.get_celld().items():
            if row % 2 == 1:
                cell.set_facecolor('#e0e0e0')
            else:
                cell.set_facecolor('#b0b0b0')

        plt.savefig(f"{LATEST_MODEL_PATH}/plots/metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved metrics.")

if __name__ == "__main__":
    logger.warning("test.py executed directly")

    model_lstm = SharpeLSTMModel(
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

    model_lstm.load_state_dict(torch.load(f"{LATEST_MODEL_PATH}/lstm/weights.pth"))
    model_fc.load_state_dict(torch.load(f"{LATEST_MODEL_PATH}/fc/weights.pth"))

    cumulative_returns_markowitz = np.load(f"{LATEST_MODEL_PATH}/markowitz/cumulative_returns_markowitz.npy")
    markowitz_returns = np.load(f"{LATEST_MODEL_PATH}/markowitz/markowitz_returns.npy")

    stock_data = Data(DF_MAG7_RAW)

    test_prices, test_returns = stock_data.get_test_dataframes(
        **PREPROCESS_KWARGS
    )

    tester = Tester(
        test_prices=test_prices,
        test_returns=test_returns,
        model_lstm=model_lstm,
        model_fc=model_fc,
        cumulative_returns_markowitz=cumulative_returns_markowitz,
        markowitz_returns=markowitz_returns,
        T=TIME_WINDOW,
        S=STOCK_COUNT,
        device=DEVICE
        )
    
    tester.run()
    
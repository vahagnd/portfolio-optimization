import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, returns_df, prices_df, window_size=32, sharpe_window=25):
        """
        returns_df: pandas DataFrame (days, 505) - daily returns
        prices_df: pandas DataFrame (days, 505) - daily prices
        window_size: int T (sequence length for LSTM)
        sharpe_window: int  R (periods over which to compute Sharpe)
        """
        assert returns_df.shape == prices_df.shape, "DataFrames must be same shape"
        self.returns = torch.tensor(returns_df, dtype=torch.float32)  # (days, S)
        self.prices = torch.tensor(prices_df, dtype=torch.float32)    # (days, S)

        # print(self.returns.shape)
        # print(self.prices.shape)

        self.T = window_size
        self.R = sharpe_window
        self.S = returns_df.shape[1]
        self.F = 2  # returns + prices
        self.max_start = len(returns_df) - self.R - self.T + 1

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        """
        Returns:
            x: (R, T, S, 2) - [returns, prices]
            y: (R, S) - next-day returns
        """
        x = []
        y = []
        for r in range(self.R):
            start = idx + r
            end = start + self.T

            ret_window = self.returns[start:end]  # (T, S)
            pri_window = self.prices[start:end]   # (T, S)

            window = torch.stack([ret_window, pri_window], dim=-1)  # (T, S, 2)
            x.append(window)
            y.append(self.returns[start + self.T])  # Next-day returns

        x = torch.stack(x)  # (R, T, S, 2)
        y = torch.stack(y)  # (R, S)
        return x, y
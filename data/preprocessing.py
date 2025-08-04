import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data.dataset import StockDataset

class Data:
    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw
        self.df_clean = self.__prepreprocess(df_raw)

    def __prepreprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepreprocess the raw DataFrame to clean and format it.
        """
        features = df.columns.values[1:]
        tickers = df.iloc[0].values[1:]
        multi_index = pd.MultiIndex.from_tuples(list(zip(features, tickers)))

        df_clean = df.iloc[2:].copy()
        df_clean.columns = ['Date'] + list(multi_index)
        df_clean.set_index('Date', inplace=True)
        df_clean.index = pd.to_datetime(df_clean.index)

        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

        # Normalize feature names by stripping suffixes like ".1"
        new_columns = [(feat.split('.')[0], ticker) for feat, ticker in df_clean.columns]
        df_clean.columns = pd.MultiIndex.from_tuples(new_columns)

        df_clean = df_clean.sort_index(axis=1)
        df_clean["date"] = pd.to_datetime(df_clean.index, errors="coerce")
        df_clean.set_index("date", inplace=True)
        df_clean = df_clean.round(4)

        return df_clean

    def __get_open(self) -> pd.DataFrame:
        return self.df_clean['Open']

    def __get_high(self) -> pd.DataFrame:
        return self.df_clean['High']

    def __get_low(self) -> pd.DataFrame:
        return self.df_clean['Low']

    def __get_close(self) -> pd.DataFrame:
        return self.df_clean['Close']

    def __get_volume(self) -> pd.DataFrame:
        return self.df_clean['Volume']

    def __preprocess(
        self,
        years: list[str], 
        prices: str = 'close',
        normalize: bool = True,
        scaler: str = 'StandardScaler'
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data by splitting into train, validation, and test sets,
        normalizing (or standardizing) the prices and returns, and returning the processed data.
        """
        if prices == 'open':
            df = self.__get_open()
        elif prices == 'high':
            df = self.__get_high()
        elif prices == 'low':
            df = self.__get_low()
        elif prices == 'close':
            df = self.__get_close()
        else:
            raise ValueError("Invalid price type")

        train_prices = df.loc[:str(years[0])]
        val_prices = df.loc[str(years[1])]
        test_prices = df.loc[str(years[2]):]

        train_returns = ((train_prices - train_prices.shift(1)) / train_prices.shift(1)).iloc[1:]
        val_returns = ((val_prices - val_prices.shift(1)) / val_prices.shift(1)).iloc[1:]
        test_returns = ((test_prices - test_prices.shift(1)) / test_prices.shift(1)).iloc[1:]

        train_prices = train_prices.iloc[1:]
        val_prices = val_prices.iloc[1:]
        test_prices = test_prices.iloc[1:]

        if normalize:
            if scaler == 'MinMaxScaler':
                scaler_prices = MinMaxScaler(feature_range=(0, 1))
                scaler_returns = MinMaxScaler(feature_range=(-1, 1))
            elif scaler == 'StandardScaler':
                scaler_prices = StandardScaler()
                scaler_returns = StandardScaler()
            else:
                raise ValueError("Unknown scaler")

            scaler_prices.fit(train_prices)
            scaler_returns.fit(train_returns)

            train_returns = scaler_returns.transform(train_returns)
            train_prices = scaler_prices.transform(train_prices)

            val_returns = scaler_returns.transform(val_returns)
            val_prices = scaler_prices.transform(val_prices)

            # test data is returned raw for now

        return train_prices, train_returns, val_prices, val_returns, test_prices, test_returns
    
    def get_train_val_datasets(
        self,
        window_size: int = 32,
        sharpe_window: int = 25,
        **kwargs
        ) -> tuple[StockDataset, StockDataset]:
        """
        Get training and validation datasets based on the preprocessed data.
        """
        train_p, train_r, val_p, val_r, _, _ = self.__preprocess(
            **kwargs
        )
        train_ds = StockDataset(train_r, train_p, window_size=window_size, sharpe_window=sharpe_window)
        val_ds = StockDataset(val_r, val_p, window_size=window_size, sharpe_window=sharpe_window)
        return train_ds, val_ds

    def get_train_val_dataloaders(
        self,
        batch_size: int = 64,
        window_size: int = 32,
        sharpe_window: int = 25,
        **kwargs
        ) -> tuple[DataLoader, DataLoader]:
        """
        Get training and validation dataloaders based on the preprocessed data.
        """
        train_ds, val_ds = self.get_train_val_datasets(
            window_size, sharpe_window, **kwargs
            )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl
    
    def get_test_dataframes(
        self,
        **kwargs
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get test prices and returns based on the preprocessed data.
        """
        _, _, _, _, test_prices, test_returns = self.__preprocess(
            **kwargs
        )
        return test_prices, test_returns

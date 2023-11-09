from datetime import datetime

import dxlib as dx
import pandas as pd
from numpy import ndarray


def custom_aggregation(x):
    value = x.dropna().values.tolist()
    if len(value) > 0:
        return value[-1]
    return None


class Dataset:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

        self.api = dx.api.AlpacaMarketsAPI(api_key, api_secret)
        self.tickers = self.get_tickers()

        self.history: dx.History | None = None
        self._timeframe = None

    @property
    def df(self):
        return self.history.df

    @staticmethod
    def to_date(x):
        return datetime.strptime(x, '%Y-%m-%d')

    def set_history(self, timeframe="1D", start="2021-01-01", end="2023-01-02"):
        self._timeframe = timeframe
        self.history = dx.History(self.get_bars(self.tickers, start, end))

    def get_bars(self, tickers, start: str | datetime, end: str | datetime) -> pd.DataFrame:
        start_date = self.to_date(start) if isinstance(start, str) else start
        end_date = self.to_date(end) if isinstance(end, str) else end
        historical_bars = self.api.get_historical_bars(tickers, start_date, end_date, self._timeframe)

        bars = historical_bars.groupby(historical_bars.index.date).agg(custom_aggregation)
        return bars.fillna(method="bfill")

    def get_tickers(self, n: int = 100) -> ndarray:
        return self.api.get_tickers(n)["symbol"].values

    def to_dict(self, df=None):
        if df is None:
            df = self.df

        # DO NOT DELETE first level of df for close, open, high, low, volume, num_trades, vwap
        # Edit second level of df columns to be security symbol instead of security object
        df = df.copy()
        df.columns.set_levels([security.symbol for security in df.columns.levels[1]], level=1, inplace=True)

        dataset = {
            "data": {
                "Close": df["Close"].to_dict(orient="list"),
                "Open": df["Open"].to_dict(orient="list"),
                "High": df["High"].to_dict(orient="list"),
                "Low": df["Low"].to_dict(orient="list"),
                "Volume": df["Volume"].to_dict(orient="list"),
                "NumTrades": df["NumTrades"].to_dict(orient="list"),
                "VWAP": df["VWAP"].to_dict(orient="list"),
            },
            "index": df.index.map(lambda x: x.strftime("%Y-%m-%d")).tolist(),
        }

        return dataset

    def extend_bars(self, start=None, end=None):
        if start is None and end is None:
            raise ValueError("Either start or end must be specified")

        if start is None:
            start = self.df.index[-1]

        if end is None:
            end = self.df.index[0]

        self.history += self.get_bars(self.tickers, start, end)

    def to_security(self, ticker: str | list[str]) -> list[dx.Security]:
        return list(self.history.security_manager.get_securities(ticker).values())

    def close(self):
        return self.df["Close"]

    def get_stock(self, ticker):
        stock_data = {
            'Ticker': ticker,
            'Close': self.df['Close'][ticker].tolist(),
            'Open': self.df['Open'][ticker].tolist(),
            'High': self.df['High'][ticker].tolist(),
            'Low': self.df['Low'][ticker].tolist(),
            'Volume': self.df['Volume'][ticker].tolist(),
        }
        return stock_data

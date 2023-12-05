from datetime import datetime

import dxlib as dx


def custom_aggregation(x):
    value = x.dropna().values.tolist()
    if len(value) > 0:
        return value[-1]
    return None


class Dataset:
    def __init__(self, api_key, api_secret, timeframe="1D"):
        self.api_key = api_key
        self.api_secret = api_secret

        self.api = dx.data.AlpacaMarketsData(api_key, api_secret)

        self.history: dx.History | None = None
        self.security_manager = dx.SecurityManager()
        self.tickers = None
        self._timeframe = timeframe

    @property
    def df(self):
        return self.history.df

    def get(self, *args, **kwargs):
        return self.history.get(*args, **kwargs)

    def set(self, start="2021-01-01", end="2023-01-01"):
        self.tickers = self.api.get_tickers(n=100)['ticker'].tolist()
        self.security_manager.add(self.tickers)
        bars = self.api.get_historical_bars(self.tickers, start, end, self._timeframe)
        self.history = dx.History(bars, self.security_manager)

    def get_bars(self, start="2021-01-01", end="2023-01-02"):
        return self.api.get_historical_bars(
            self.tickers,
            start,
            end,
            self._timeframe)

    def extend(self, start=None, end=None):
        if start is None:
            start = self.df.index[0][0]

        if end is None:
            end = self.df.index[-1][0]

        self.history += self.get_bars(start, end)

    def to_security(self, ticker: str | list[str]) -> list[dx.Security]:
        return list(self.history.security_manager.get(ticker).values())


def date(x):
    return datetime.strptime(x, '%Y-%m-%d')


def main():
    from config import API_KEY, API_SECRET

    dataset = Dataset(API_KEY, API_SECRET)
    dataset.set(start="2021-01-01", end="2023-01-01")
    print(dataset.df.head())


if __name__ == "__main__":
    main()

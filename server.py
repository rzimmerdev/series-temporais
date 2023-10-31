import dxlib as dx

from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn


available_columns = {
                "close": "Close",
                "volume": "Volume",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "vwap": "VWAP",
            }


class Server:
    def __init__(self):
        self.data = load_data()
        self.app = FastAPI()
        self.financial_indicators = FinancialIndicators()
        self.setup()

    def setup(self):
        @self.app.get("/")
        async def root():
            return {"message": "Hello World"}

        # Get available symbols
        @self.app.get("/stocks")
        async def get_symbols():
            return self.data["Close"].columns.values.tolist()

        # Get all symbols, one field
        @self.app.get("/stocks/{field}")
        async def get_symbols(field: str):
            field = available_columns.get(field, "Invalid field")
            return self.data[field].to_json()

        # Get one symbol, all fields
        @self.app.get("/stock/{symbol}")
        async def get_symbol(symbol: str):

            return self.get_symbol(symbol).to_json()

        self.financial_indicators.set_routes(self)

    def get_symbol(self, symbol: str):
        all_fields = self.data.columns.values.tolist()

        fields = [field for field in all_fields if field[1] == symbol]
        return self.data[fields]

    def get_close(self, symbol: str):
        return self.data["Close", symbol]

    def run(self, host="0.0.0.0"):
        uvicorn.run(self.app, host=host, port=8000)


# TODO: Financial Indicators - Volatility, Moving Averages, Bollinger Bands, etc.
# TODO: Time series analysis - autocorrelation, differencing, etc.
# TODO: ARIMA and GARCH.

class FinancialIndicators:
    # To calculate annualized volatility, pass daily returns, period=252 and window=20
    def volatility(self, series, window=252, period=252):
        volatility = series.rolling(window, min_periods=1).std()

        period_volatility = volatility * np.sqrt(period)
        return period_volatility

    def moving_average(self, series, window=20, exponential=False) -> pd.Series:
        if exponential:
            return series.ewm(span=window).mean()
        return series.rolling(window).mean()

    def bollinger_bands(self, series, window=20, exponential=False) -> (pd.Series, pd.Series):
        moving_average = self.moving_average(series, window, exponential)
        std = series.rolling(window).std()
        upper_band = moving_average + 2 * std
        lower_band = moving_average - 2 * std
        return upper_band, lower_band

    def returns(self, series, period=1):
        return series.pct_change(period)

    def log_change(self, series, period=1):
        return np.log(series).diff(period)

    def set_routes(self, server):
        @server.app.get("/volatility/{symbol}")
        async def get_volatility(symbol: str):
            return self.volatility(server.get_close(symbol)).to_json()

        @server.app.get("/volatility/{symbol}/window/{window}")
        async def get_volatility(symbol: str, window: int):
            return self.volatility(server.get_close(symbol), window).to_json()

        @server.app.get("/moving_average/{symbol}")
        async def get_moving_average(symbol: str):
            return self.moving_average(server.get_close(symbol)).to_json()

        @server.app.get("/moving_average/{symbol}/window/{window}")
        async def get_moving_average(symbol: str, window: int):
            return self.moving_average(server.get_close(symbol), window).to_json()

        @server.app.get("/bollinger_bands/{symbol}/window/{window}")
        async def get_bollinger_bands(symbol: str, window: int):
            return self.bollinger_bands(server.get_close(symbol), window).to_json()

        @server.app.get("/returns/{symbol}/period/{period}")
        async def get_returns(symbol: str, period: int):
            return self.returns(server.get_close(symbol), period).to_json()

        @server.app.get("/log_returns/{symbol}/period/{period}")
        async def get_log_change(symbol: str, period: int):
            return self.log_change(server.get_close(symbol), period).to_json()


class TechnicalIndicators:
    def __init__(self):
        self.financial_indicators = FinancialIndicators()

    def autocorrelation(self, series: pd.Series, lag):
        return series.autocorr(lag)

    def diff(self, series, period=1):
        return series.diff(period)

    def detrend(self, series):
        return series - self.financial_indicators.moving_average(series)

    def set_routes(self, server):
        @server.app.get("/autocorrelation/{symbol}/lag/{lag}")
        async def get_autocorrelation(symbol: str, lag: int):
            return self.autocorrelation(server.get_close(symbol), lag)

        @server.app.get("/diff/{symbol}")
        async def get_diff(symbol: str):
            return self.diff(server.get_close(symbol)).to_json()

        @server.app.get("/detrend/{symbol}")
        async def get_detrend(symbol: str):
            return self.detrend(server.get_close(symbol)).to_json()


def load_data(start="2021-01-01", end="2023-06-06"):
    data = get_api(start, end)

    def custom_aggregation(x):
        value = x.dropna().values.tolist()
        if len(value) > 0:
            return value[-1]
        return None

    data = data.groupby(data.index.date).agg(custom_aggregation)
    return data.fillna(method="bfill")


def date(x):
    return datetime.strptime(x, '%Y-%m-%d')


def get_api(start="2021-01-01", end="2023-06-06"):
    start_date = date(start)
    end_date = date(end)

    try:
        import config
        alpaca_api = dx.api.AlpacaMarketsAPI(config.api_key, config.api_secret)

    except ModuleNotFoundError:
        print("No API config found, trying to use cached data")
        alpaca_api = dx.api.AlpacaMarketsAPI()

    securities = alpaca_api.get_symbols(100)

    data = alpaca_api.get_historical_bars(securities["symbol"].values, start_date, end_date)

    return data


def main():
    server = Server()
    server.run()


if __name__ == "__main__":
    main()

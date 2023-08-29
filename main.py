from dxlib import api
from datetime import datetime
from plotly import express as px


def date(x):
    return datetime.strptime(x, '%Y-%m-%d')


def main():
    try:
        import config
    except ModuleNotFoundError:
        print("No API config found, trying to use cached data")
        config = None

    if config:
        alpaca_api = api.AlpacaMarketsAPI(config.api_key, config.api_secret)
    else:
        alpaca_api = api.AlpacaMarketsAPI()

    try:
        start_date = date("2022-08-28")
        end_date = date("2023-08-27")
        symbols = alpaca_api.get_symbols(100)

        print(symbols)

        historical_bars = alpaca_api.get_historical_bars(symbols["symbol"].values, start_date, end_date)

        print(historical_bars)
        px.line(historical_bars["Close"][symbols["symbol"][0]]).show()

    except KeyError as e:
        print("Could not get data from cache", e)


if __name__ == "__main__":
    main()

from dxlib import api, History
from datetime import datetime
from plotly import express as px

import numpy as np
import pandas as pd


def date(x):
    return datetime.strptime(x, '%Y-%m-%d')


def autocorrelation(data, lag):
    if isinstance(data, pd.Series):
        data = data.values
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy array or pandas.Series")

    n = len(data)
    mean = np.mean(data)

    autocorr = np.correlate(data - mean, data - mean, mode='full') / (n * np.var(data))
    at_lag = autocorr[n - 1 + lag]

    return at_lag


def plot_series(series, title, xaxis_title='Time', yaxis_title='Value'):
    fig = px.line(series)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.write_html(title + '.html')


def plot_heatmap(matrix):
    fig = px.imshow(matrix, color_continuous_scale='Viridis')
    fig.update_layout(
        title='Autocorrelation Heatmap',
        xaxis_title='Lag',
        yaxis_title='Lag',
    )
    fig.show()


def add_html(html, series, fig_name, plot_name):
    plot_series(series, title=fig_name)

    new_html = html + f"<h2>Plot - {plot_name}</h2>"
    new_html += f'<iframe src="{fig_name}.html" width="800" height="600"></iframe>'
    return new_html


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
        historical_bars = alpaca_api.get_historical_bars(symbols["symbol"].values, start_date, end_date)
        selected_symbol = symbols["symbol"][0]

        symbol_bars = historical_bars["Close"][selected_symbol].dropna()

        history = History(pd.DataFrame(symbol_bars, columns=[selected_symbol]))

        print(symbols)
        print(historical_bars)

        # close = historical_bars["Close"]
        # volume = historical_bars["Volume"]

        # close_with_large_volume = close[volume.mean() > 1e7]

        # corr = historical_bars[historical_bars["Volume"].mean() > 1e7]["Close"].dropna().corr()
        # plot_heatmap(corr)

        html_string = ("<html><head><title>Time Series analysis of a Financial Security</title></head><body><h1>Time "
                       "Series analysis of a Financial Security</h1>")

        security = history.security_manager.add_security(selected_symbol)
        security_close = history.indicators.series.log_change()[security].dropna()

        plot_name = "Adjusted Log Change"
        fig_name = f"adjusted_log_change_{selected_symbol}"

        html_string = add_html(html_string, security_close, fig_name, plot_name)

        selected_lags = np.array([1, 3, 5, 7, 14, 15, 20, 22, 30, 60])
        autocorrelations = np.array([autocorrelation(security_close, lag) for lag in selected_lags])
        significant_lags = np.argsort(autocorrelations)[-4:]

        for idx_lag in significant_lags:
            print(autocorrelations[idx_lag])
            original = security_close

            shifted = security_close.shift(periods=selected_lags[idx_lag])
            shifted.columns = [f"{security.symbol}_lag_{selected_lags[idx_lag]}"]

            df = pd.concat([original, shifted], axis=1).dropna()
            df.columns = [selected_symbol, f"{selected_symbol}_lag{selected_lags[idx_lag]}"]

            plot_name = f"Lag={selected_lags[idx_lag]} overlay, with autocorrelation={autocorrelations[idx_lag]}"
            fig_name = f"lag_{selected_lags[idx_lag]}_overlay"

            html_string = add_html(html_string, df, fig_name, plot_name)

        html_string += "</body></html>"

        with open("output.html", "w") as html_file:
            html_file.write(html_string)

    except KeyError as e:
        print("Could not get data from cache", e)


if __name__ == "__main__":
    main()

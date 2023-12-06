import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dxlib.indicators import TechnicalIndicators, SeriesIndicators
from dxlib import History

from statsmodels.tsa.seasonal import seasonal_decompose

from dataset import Dataset
import config


class Server:
    def __init__(self):
        self.dataset = Dataset(config.API_KEY, config.API_SECRET)
        self.dataset.set()

        self.analysis_tools = AnalysisTools(self)

        self.app = None
        self.set_app()

        self.set_routes()

    def set_app(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def set_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Hello World"}

        @self.app.get("/tickers")
        async def get_tickers():
            return self.dataset.tickers

        @self.app.post("/get")
        async def post_stocks(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get("tickers", None)
            fields = body.get("fields", None)
            interval = body.get("interval", None)

            if interval is not None:
                self.dataset.extend(interval[0], interval[1])
            else:
                self.dataset.set()

            try:
                data = self.dataset.history.get_interval(tickers, fields, [interval])
            except KeyError:
                raise HTTPException(status_code=400, detail="Invalid ticker")

            return data.serialized()['df']

        self.analysis_tools.set_routes()

    def run(self, host="0.0.0.0"):
        uvicorn.run(self.app, host=host, port=8000)


# TODO: ARIMA and GARCH.
class AnalysisTools:
    def __init__(self, server):
        self.tsi = TechnicalIndicators()
        self.ssi = SeriesIndicators()
        self.server = server

    def set_routes(self):
        @self.server.app.post("/analysis")
        async def analysis(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get("tickers", None)
            fields = body.get("fields", None)
            interval = body.get("interval", None)
            indicators = body.get("indicators", [])
            operations = body.get("operations", [])
            params = body.get("params", {})

            data = self.server.dataset.history.get_interval(tickers, fields, [interval])
            response = {}

            for indicator in indicators:
                response[indicator] = {}

                for operation in operations:
                    response[indicator][operation] = self.apply(data, fields, indicator, operation, params)

            return response

        @self.server.app.post("/autocorrelation")
        async def autocorrelation(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get("tickers", None)
            lag_range = body.get("range", None)

            data = self.server.dataset.history

            response = {"pacf": {}, "acf": {}}
            for ticker in tickers:
                df = data.get(securities=[ticker]).get_raw(fields=['close'])
                df = self.ssi.log_change(df, window=1).dropna()
                response["pacf"][ticker] = self.ssi.pacf(df, lag_range).dropna().tolist()
                response["acf"][ticker] = self.ssi.autocorrelation(df, lag_range)

            return response

        @self.server.app.post("/decompose")
        async def decompose(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get("tickers", None)
            period = body.get("period", None)
            interval = body.get("interval", None)

            data = self.server.dataset.history.get_interval(intervals=[interval])

            response = {}
            for ticker in tickers:
                response[ticker] = {}

                df = data.get(securities=[ticker]).get_raw(fields=['close'])
                df = self.ssi.log_change(df, window=1).dropna()

                decomposition = seasonal_decompose(df, period=period)

                response[ticker]["trend"] = decomposition.trend.dropna().tolist()
                response[ticker]["seasonal"] = decomposition.seasonal.dropna().tolist()
                response[ticker]["residual"] = decomposition.resid.dropna().tolist()

            return response

        @self.server.app.post("/arima")
        async def ar(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get("tickers", None)
            interval = body.get("interval", None)
            end = body.get("end", None)

            data = self.server.dataset.history.get_interval(intervals=[interval])
            end = min(end, int(len(data.df) * 0.05)) if end is not None else 20

            p = body.get("p", None)
            d = body.get("d", None)
            q = body.get("q", None)

            params = {
                'start_p': 0 if p is None else p,
                'd': 0 if d is None else d,
                'start_q': 0 if q is None else q,
                'max_p': 0 if p == 0 else 5,
                'max_d': 0 if d == 0 else 2,
                'max_q': 0 if q == 0 else 5,
                'seasonal': False,
                'stepwise': True
            }

            response = {}
            for ticker in tickers:
                response[ticker] = {}

                df = data.get(securities=[ticker]).get_raw(fields=['close'])

                train_df = df[:-end]
                test_size = len(df) - len(train_df)
                model = auto_arima(train_df, **params)

                predictions = model.predict(test_size).tolist()
                true_values = df[-end:].values.tolist()

                response[ticker]["forecast"] = {
                    "predictions": predictions,
                    "values": true_values
                }
                residuals = (predictions - df[-end:]).abs().tolist()
                response[ticker]["residuals"] = residuals
                response[ticker]["error"] = (predictions - df[-end:]).abs().mean().tolist()
                response[ticker]["order"] = model.order

            return response

    @staticmethod
    def formatter(output):
        return History(pd.DataFrame(output)).serialized()['df']

    def apply(self, data: History, fields, indicator, operation, params):
        match indicator:
            case "value":
                data = data.get(fields=fields).df
            case "volatility":
                window = params.get("window", 252)
                period = params.get("period", 252)
                data = self.tsi.volatility(data.df, window, period)
            case "moving_average":
                window = params.get("window", 20)
                data = self.ssi.sma(data.df, window)
            case "bollinger_bands":
                window = params.get("window", 20)
                data = self.tsi.bollinger_bands(data.df, window)

        match operation:
            case "value":
                pass
            case "returns":
                data = self.ssi.returns(data)
            case "log_change":
                data = self.ssi.log_change(data)
            case "autocorrelation":
                lags = params.get("lags", 15)
                data = self.ssi.autocorrelation(data, lags)
            case "diff":
                period = params.get("period", 1)
                data = self.ssi.diff(data, period)
            case "detrend":
                data = self.ssi.detrend(data)

        return self.formatter(data.dropna())

    @staticmethod
    def to_dict(df):
        # df.columns.set_levels([security.symbol for security in df.columns.levels[1]], level=1, inplace=True)
        df.columns = pd.Index([security.symbol for security in df.columns])

        dataset = {
            "data": df.to_dict(orient="list"),
            "index": df.index.map(lambda x: x.strftime("%Y-%m-%d")).tolist(),
        }

        return dataset


def main():
    server = Server()
    server.run()


if __name__ == "__main__":
    main()

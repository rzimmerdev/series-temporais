import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dxlib.indicators import TechnicalIndicators, SeriesIndicators

from dataset import Dataset
import config


class Server:
    def __init__(self):
        self.dataset = self.get_dataset(config.api_key, config.api_secret)
        self.analysis_tools = AnalysisTools(self)

        self.app = None
        self.set_app()

        self.set_routes()

    @staticmethod
    def get_dataset(api_key=None, api_secret=None):
        if api_key is None and api_secret is None:
            raise ValueError("Either dataset or api_key and api_secret must be specified")
        dataset = Dataset(api_key, api_secret)
        dataset.set_history()

        return dataset

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
            return self.dataset.tickers.tolist()

        @self.app.post("/stocks")
        async def post_stocks(request: Request):
            if request.headers.get("Content-Type") == "application/json":
                body = await request.json()
            else:
                body = {}

            tickers = body.get('tickers', None)

            if tickers is None:
                return self.dataset.to_dict()
            else:
                try:
                    stocks = self.dataset.history.get_by_ticker(tickers)
                    return self.dataset.to_dict(stocks)
                except KeyError:
                    raise HTTPException(status_code=404, detail="Tickers not found in the dataset")

        @self.app.post("/period")
        async def set_period(start: str = None, end: str = None):
            self.dataset.extend_bars(start, end)

        self.analysis_tools.set_routes()

    def run(self, host="0.0.0.0"):
        uvicorn.run(self.app, host=host, port=8000)


# TODO: ARIMA and GARCH. Maybe also LSTM?
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

            field = body.get("field", "Close")
            tickers = body.get("tickers", [])
            indicators = body.get("indicators", [])
            operations = body.get("operations", [])
            params = body.get("params", {})

            securities = self.server.dataset.to_security(tickers)

            data = self.server.dataset.df[field][securities]

            response = {}

            for indicator in indicators:
                response[indicator] = {}

                for operation in operations:
                    response[indicator][operation] = self.apply(data, indicator, operation, params)

            return response

    def apply(self, data, indicator, operation, params):
        match indicator:
            case "value":
                data = data
            case "volatility":
                window = params.get("window", 252)
                period = params.get("period", 252)
                data = self.tsi.volatility(data, window, period)
            case "moving_average":
                window = params.get("window", 20)
                data = self.ssi.sma(data, window)
            case "bollinger_bands":
                window = params.get("window", 20)
                data = self.tsi.bollinger_bands(data, window)

        match operation:
            case "returns":
                data = self.ssi.returns(data)
            case "log_change":
                data = self.ssi.log_change(data)
            case "auto_correlation":
                lags = params.get("lags", 15)
                data = self.ssi.autocorrelation(data, lags)
            case "diff":
                period = params.get("period", 1)
                data = self.ssi.diff(data, period)
            case "detrend":
                data = self.ssi.detrend(data)

        return self.to_dict(data.dropna())

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

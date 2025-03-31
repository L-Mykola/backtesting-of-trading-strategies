import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import vectorbt as vbt
from core.backtester import Backtester
from strategies.base import StrategyBase


class DummyWrapper:
    def __init__(self):
        self.index = pd.date_range("2020-01-01", periods=5, freq="D")


class DummyPortfolio:
    def __init__(self):
        self._value = pd.DataFrame(
            {"dummy": [10000, 10100, 10200, 10300, 10400]},
            index=pd.date_range("2020-01-01", periods=5, freq="D")
        )
        self.wrapper = DummyWrapper()

    def value(self):
        return self._value

    def total_return(self):
        return pd.Series([0.04], index=["dummy"])


class DummyStrategy(StrategyBase):
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data
        self.backtest_result = None

    def generate_signals(self) -> dict:
        close = self.price_data.xs("close", level=1, axis=1)
        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        return {"entries": entries, "exits": exits}

    def run_backtest(self) -> vbt.Portfolio:
        return DummyPortfolio()

    def get_metrics(self, metrics_csv_path: str) -> dict:
        df = pd.DataFrame({"metric": ["dummy_metric"], "value": [42]})
        df.to_csv(metrics_csv_path, index=False)
        return {"dummy_metric": 42}


# Тест для Backtester
def test_backtester(tmp_path: Path):

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    symbols = ["dummy"]

    arrays = [np.repeat(symbols, 1), ["close"]]
    cols = pd.MultiIndex.from_arrays(arrays, names=["symbol", "ohlcv"])

    data = {"close": [10000, 10100, 10200, 10300, 10400],
            "open": [9990, 10090, 10190, 10290, 10390],
            "high": [10010, 10110, 10210, 10310, 10410],
            "low": [9980, 10080, 10180, 10280, 10380],
            "volume": [10, 20, 30, 40, 50]}
    price_data = pd.DataFrame(data, index=dates)

    price_data_close = price_data[["close"]].copy()
    price_data_close.columns = cols

    dummy_strategy = DummyStrategy(price_data_close)

    project_dir = tmp_path
    backtester = Backtester(dummy_strategy, "DummyStrategy", project_dir)

    metrics = backtester.run()

    metrics_csv_path = os.path.join(backtester.results_dir, "DummyStrategy_metrics.csv")
    assert os.path.exists(metrics_csv_path)

    equity_curve_html = os.path.join(backtester.html_dir, "DummyStrategy_equity_curve.html")
    equity_curve_png = os.path.join(backtester.screenshots_dir, "DummyStrategy_equity_curve.png")
    heatmap_html = os.path.join(backtester.html_dir, "DummyStrategy_heatmap.html")
    heatmap_png = os.path.join(backtester.screenshots_dir, "DummyStrategy_heatmap.png")
    for file_path in [equity_curve_html, equity_curve_png, heatmap_html, heatmap_png]:
        assert os.path.exists(file_path)

    assert metrics.get("dummy_metric") == 42

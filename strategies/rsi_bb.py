import os
from typing import List, Dict

import pandas as pd
import vectorbt as vbt
import ta

from strategies.base import StrategyBase
from core.metrics import Metrics


class RSIBBStrategy(StrategyBase):
    """
    Strategy using RSI and confirmation through Bollinger Bands.
    """

    def __init__(self, pairs: List[str], price_data: pd.DataFrame, rsi_period: int = 14, bb_window: int = 20, bb_std: float = 2):
        super().__init__(price_data)
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.pairs = pairs
        self.backtest_result = None

    def generate_signals(self) -> Dict:
        """
        Generates signals:
         - Entry: when RSI < 30 and the price is near the lower boundary of the Bollinger Bands.
         - Exit: when RSI > 70 or the price is approaching the upper boundary of the Bollinger Bands.
         :return: dict for generated signals
        """
        close = self.price_data.xs("close", level=1, axis=1)
        rsi = close.apply(lambda s: ta.momentum.RSIIndicator(s, window=self.rsi_period).rsi())

        bb_indicator = close.rolling(self.bb_window)
        middle_band = bb_indicator.mean()
        std = close.rolling(self.bb_window).std()
        lower_band = middle_band - self.bb_std * std
        upper_band = middle_band + self.bb_std * std

        entries = ((rsi < 30) & (close <= lower_band * 1.01)).fillna(False)
        exits = ((rsi > 70) | (close >= upper_band * 0.99)).fillna(False)

        signals = {"entries": entries, "exits": exits}
        return signals

    def run_backtest(self) -> vbt.Portfolio:
        """
        Launches a strategy backtest
        :return: vbt.Portfolio for backtest results
        """
        signals = self.generate_signals()
        close = self.price_data.xs("close", level=1, axis=1)
        close = close.bfill().clip(lower=0.01)
        self.backtest_result = vbt.Portfolio.from_signals(
            close,
            entries=signals["entries"],
            exits=signals["exits"],
            init_cash=10000,
            fees=0.001,
            slippage=0.001,
            freq="1min",
        )
        return self.backtest_result

    def get_metrics(self, path: os.path) -> Dict:
        """
        Aggregate strategy performance metrics and save it to csv.
        :return: dict with strategy metrics
        """
        metrics = Metrics(self.backtest_result, self.pairs)
        agg_metrics = metrics.aggregate_metrics()
        metrics.save_to_csv(agg_metrics, path)
        return agg_metrics
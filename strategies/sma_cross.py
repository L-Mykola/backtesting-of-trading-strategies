import os
from typing import Dict, List

import pandas as pd
import vectorbt as vbt

from strategies.base import StrategyBase
from core.metrics import Metrics


class SMACrossStrategy(StrategyBase):
    """
    The strategy of crossing two moving averages (SMA Crossover).
    """

    def __init__(self, price_data: pd.DataFrame, pairs: List[str], fast_window: int = 10, slow_window: int = 30):
        super().__init__(price_data)
        self.pairs = pairs
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signals = None
        self.backtest_result = None

    def generate_signals(self) -> Dict:
        """
        It generates signals based on the intersection of a short and a long SMA.
        :return: dict for generated signals
        """

        close = self.price_data.xs("close", level=1, axis=1)
        fast_sma = close.rolling(self.fast_window).mean()
        slow_sma = close.rolling(self.slow_window).mean()

        entries = fast_sma > slow_sma
        exits = fast_sma < slow_sma
        signals = {"entries": entries, "exits": exits}
        return signals

    def run_backtest(self) -> vbt.Portfolio:
        """
        Launches a strategy backtest
        :return: vbt.Portfolio for backtest results
        """

        signals = self.generate_signals()

        close = self.price_data.xs("close", level=1, axis=1)
        self.backtest_result = vbt.Portfolio.from_signals(
            close,
            entries=signals["entries"],
            exits=signals["exits"],
            init_cash=10000,
            fees=0.001,
            slippage=0.001,
            freq="1min"
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


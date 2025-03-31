import os
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import vectorbt as vbt


class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> Dict:
        """
        Generates signals for trading.
        :return: dict for generated signals
        """
        pass

    @abstractmethod
    def run_backtest(self) -> vbt.Portfolio:
        """
        Launches a strategy backtest
        :return: vbt.Portfolio for backtest results
        """
        pass

    @abstractmethod
    def get_metrics(self, path: os.path) -> Dict:
        """
        Aggregate strategy performance metrics and save it to csv.
        :return: dict with strategy metrics
        """
        pass

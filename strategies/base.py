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
        :return: pd Dataframe for generated signals
        """
        pass

    @abstractmethod
    def run_backtest(self) -> vbt.Portfolio:
        """
        Launches a strategy backtest
        :return: pd Dataframe for backtest results
        """
        pass

    @abstractmethod
    def get_metrics(self, path: os.path) -> Dict:
        """
        Calculates strategy performance metrics.
        :return: dict with strategy metrics
        """
        pass

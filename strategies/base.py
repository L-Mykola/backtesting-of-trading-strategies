from abc import ABC, abstractmethod
import pandas as pd


class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generates signals for trading.
        """
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """
        Launches a strategy backtest
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Calculates strategy performance metrics.
        """
        pass

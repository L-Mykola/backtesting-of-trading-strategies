import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase


class VWAPReversionStrategy(StrategyBase):
    """
    VWAP Reversion strategy.
    Entry occurs when the price deviates significantly from VWAP with an expected return.
    """
    def __init__(self, price_data: pd.DataFrame, deviation: float = 0.01):
        super().__init__(price_data)
        self.deviation = deviation
        self.signals = None
        self.backtest_result = None

    def calculate_vwap(self) -> pd.Series:
        """
        Calculates VWAP.
        """

        price = self.price_data['close']
        volume = self.price_data['volume']
        vwap = (price * volume).cumsum() / volume.cumsum()
        return vwap

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates a signal when the price deviates significantly from the VWAP.
        :return: pd Dataframe for generated signals
        """

        price = self.price_data['close']
        vwap = self.calculate_vwap()

        signal = np.where(price > vwap * (1 + self.deviation), -1,
                          np.where(price < vwap * (1 - self.deviation), 1, 0))
        self.signals = pd.DataFrame({
            "vwap": vwap,
            "signal": signal
        }, index=self.price_data.index)
        return self.signals

    def run_backtest(self) -> pd.DataFrame:
        """
        Launches a strategy backtest
        :return: pd Dataframe for backtest results
        """

        if self.signals is None:
            self.generate_signals()

        price = self.price_data['close']
        entries = self.signals['signal'] > 0
        exits = self.signals['signal'] < 0

        pf = vbt.Portfolio.from_signals(price, entries, exits,
                                        freq='1min',
                                        init_cash=10000,
                                        fees=0.001,
                                        slippage=0.001)
        self.backtest_result = pf
        return pf.stats()

    def get_metrics(self) -> dict:
        """
        Calculates strategy performance metrics.
        :return: dict with strategy metrics
        """

        if self.backtest_result is None:
            self.run_backtest()
        stats = self.backtest_result.stats()
        metrics = {
            "total_return": stats["Total Return"],
            "sharpe_ratio": stats["Sharpe Ratio"],
            "max_drawdown": stats["Max Drawdown"],
            "winrate": None,
            "expectancy": None,
            "exposure_time": None
        }
        return metrics

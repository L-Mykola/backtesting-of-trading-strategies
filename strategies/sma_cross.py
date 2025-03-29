import pandas as pd
import numpy as np
import vectorbt as vbt
from .base import StrategyBase


class SMACrossStrategy(StrategyBase):
    """
    The strategy of crossing two moving averages (SMA Crossover).
    """

    def __init__(self, price_data: pd.DataFrame, short_window: int = 10, long_window: int = 30):
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
        self.backtest_result = None

    def generate_signals(self) -> pd.DataFrame:
        """
        It generates signals based on the intersection of a short and a long SMA.
        :return: pd Dataframe for generated signals
        """

        price = self.price_data['close']
        sma_short = price.rolling(window=self.short_window, min_periods=1).mean()
        sma_long = price.rolling(window=self.long_window, min_periods=1).mean()
        signal = np.where(sma_short > sma_long, 1, -1)
        self.signals = pd.DataFrame({
            "sma_short": sma_short,
            "sma_long": sma_long,
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
        entries = self.signals['signal'] == 1
        exits = self.signals['signal'] == -1

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
            "winrate": None,  # Розрахунок winrate можна доповнити
            "expectancy": None,  # Розрахунок expectancy можна доповнити
            "exposure_time": None  # Розрахунок exposure_time можна доповнити
        }
        return metrics

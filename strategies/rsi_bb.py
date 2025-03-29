import pandas as pd
import numpy as np
import vectorbt as vbt
import ta
from .base import StrategyBase


class RSIBBStrategy(StrategyBase):
    """
    A strategy based on RSI with Bollinger Bands confirmation.
    Enter when RSI < 30 and rebound from the lower boundary of BB.
    """

    def __init__(self, price_data: pd.DataFrame, rsi_period: int = 14, bb_window: int = 20):
        super().__init__(price_data)
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.signals = None
        self.backtest_result = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates a buy signal when the RSI < 30 and the price exceeds the lower boundary of BB
        :return: pd Dataframe for generated signals
        """
        price = self.price_data['close']
        rsi = ta.momentum.RSIIndicator(price, window=self.rsi_period).rsi()
        bb = ta.volatility.BollingerBands(price, window=self.bb_window)
        bb_lower = bb.bollinger_lband()

        signal = np.where((rsi < 30) & (price > bb_lower), 1, -1)
        self.signals = pd.DataFrame({
            "rsi": rsi,
            "bb_lower": bb_lower,
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
            "winrate": None,
            "expectancy": None,
            "exposure_time": None
        }
        return metrics

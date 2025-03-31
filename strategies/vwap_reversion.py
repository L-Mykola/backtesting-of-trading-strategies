import os
from typing import List, Dict

import pandas as pd
import vectorbt as vbt

from strategies.base import StrategyBase
from core.metrics import Metrics


class VWAPReversionStrategy(StrategyBase):
    """
    VWAP Reversion Intraday strategy.
    """

    def __init__(self, price_data: pd.DataFrame, pairs: List[str], threshold: float = 0.01):
        super().__init__(price_data)
        self.pairs = pairs
        self.threshold = threshold
        self.backtest_result = None

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculates the intraday VWAP for a single DataFrame of an asset.
        :return: pd.Series for calculated VWAP
        """
        df = df.copy()
        df['typical_price'] = df['close']
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        return df['vwap']

    def generate_signals(self) -> Dict:
        """
        Generates signals based on VWAP:
         - Entry: when the price < VWAP * (1 - threshold).
         - Output: when the price > VWAP * (1 + threshold).
         :return: dict for generated signals
        """
        close = self.price_data.xs("close", level=1, axis=1)
        volume = self.price_data.xs("volume", level=1, axis=1)

        # Розрахунок VWAP для кожного активу окремо
        vwap_df = pd.DataFrame(index=close.index, columns=close.columns)
        for col in close.columns:
            df_asset = pd.DataFrame({
                'close': close[col],
                'volume': volume[col]
            })
            df_asset['date'] = df_asset.index.date
            vwap_list = []
            for date, group in df_asset.groupby('date'):
                vwap_series = self.calculate_vwap(group)
                vwap_list.append(vwap_series)
            vwap_asset = pd.concat(vwap_list)
            vwap_df[col] = vwap_asset.sort_index()

        entries = close < vwap_df * (1 - self.threshold)
        exits = close > vwap_df * (1 + self.threshold)
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


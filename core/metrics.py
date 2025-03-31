import csv
import os
from typing import List, Dict
from loguru import logger
import numpy as np
import vectorbt as vbt


class Metrics:
    """
    Class for aggregating and save metrics
    """

    def __init__(self, portfolio: vbt.Portfolio, pairs: List):
        self.portfolio = portfolio
        self.pairs = pairs

    def _compute_exposure_time(self, threshold: float = 1e-6) -> float:
        """
        Calculates Exposure Time as the percentage of time the portfolio had open positions.
        It uses the difference between portfolio.value and portfolio.cash.
        :param threshold: a numerical threshold for determining an open position (1e-6 by default)
        :return: Exposure Time as a percentage (float)
        """
        is_exposed = (self.portfolio.value() - self.portfolio.cash()).abs() > threshold
        exposure_fraction = is_exposed.mean()
        return exposure_fraction.mean() * 100

    def aggregate_metrics(self) -> Dict:
        """
        Aggregates metrics for a portfolio of many trading pairs.
        :return: aggregated metrics as dict
        """
        sharpe_ratio_sum = 0
        for pair in self.pairs:
            sharpe_ratio = self.portfolio.stats(column=f'{pair}')['Sharpe Ratio']
            if not np.isinf(sharpe_ratio):
                sharpe_ratio_sum += sharpe_ratio
            else:
                sharpe_ratio_sum += 0

        agg_exposure = self._compute_exposure_time()

        aggregated = {
            "Total Return": self.portfolio.stats(agg_func=np.mean, silence_warnings=True)['Total Return [%]'],
            "Sharpe Ratio": sharpe_ratio_sum/len(self.pairs),
            "Max Drawdown": self.portfolio.stats(agg_func=np.mean, silence_warnings=True)['Max Drawdown [%]'],
            "Win Rate": self.portfolio.stats(agg_func=np.mean, silence_warnings=True)['Win Rate [%]'],
            "Expectancy": self.portfolio.stats(agg_func=np.mean, silence_warnings=True)['Expectancy'],
            "Exposure Time": agg_exposure
        }

        return aggregated

    @staticmethod
    def save_to_csv(metrics: Dict, path: os.path):
        with open(path, 'w') as file:
            w = csv.DictWriter(file, metrics.keys())
            w.writeheader()
            w.writerow(metrics)
        logger.info(f"Metrics are saved in {path}")


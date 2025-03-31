import os

import pandas as pd
from loguru import logger
from tabulate import tabulate

from core.data_loader import DataLoader
from strategies.sma_cross import SMACrossStrategy
from strategies.rsi_bb import RSIBBStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from core.backtester import Backtester


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    data_loader = DataLoader(project_dir=project_dir, start_date="2025-02-01", end_date="2025-02-28")
    pairs, price_data = data_loader.process(num_of_pairs=100)

    strategies = {
        "SMACrossoverStrategy": SMACrossStrategy(price_data=price_data, fast_window=10, slow_window=30, pairs=pairs),
        "RSIBBStrategy": RSIBBStrategy(price_data=price_data, rsi_period=14, bb_window=20, bb_std=2, pairs=pairs),
        "VWAPReversionStrategy": VWAPReversionStrategy(price_data=price_data, threshold=0.01, pairs=pairs)
    }
    results = []
    for name, strategy in strategies.items():
        logger.info(f"Start backtest for {name}")
        backtester = Backtester(strategy=strategy, strategy_name=name, project_dir=project_dir)
        result = backtester.run()
        results.append(result)
        logger.info(f"The backtest for {name} is complete. The results have been saved in the 'results' folder.")

    result_df = pd.DataFrame(results, index=list(strategies.keys()))
    result_table = tabulate(result_df, headers='keys', tablefmt="fancy_grid", showindex=True)
    logger.info("\n\nBacktest results\n" + result_table)


if __name__ == "__main__":
    main()

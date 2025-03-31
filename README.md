# Backtesting Trading Strategies

This project is a modular and scalable system for backtesting multiple trading strategies on 1-minute OHLCV data. It supports multiple trading pairs (e.g., 100 pairs against BTC) and leverages vectorbt for fast backtesting and performance analysis.

## Features

- **Data Loader**: Downloads and preprocesses historical 1-minute OHLCV data for selected trading pairs.
- **Strategy Interface**: Provides an abstract base class (StrategyBase) for defining trading strategies.
- **Multiple Strategies**: Includes sample strategies such as SMA Crossover, RSI with Bollinger Band confirmation, and VWAP Reversion.
- **Backtesting Framework**: Utilizes vectorbt to simulate trades while accounting for commission, slippage, and execution delay.
- **Metrics & Visualization**: Calculates key performance metrics (Total Return, Sharpe Ratio, Max Drawdown, etc.) and generates interactive Plotly graphs (equity curve and heatmaps).
- **Testing**: Unit tests for critical components using pytest.
- **Reporting**: Aggregates strategy metrics and outputs tables (using tabulate and loguru) as well as CSV and HTML files for further analysis


## Setup and Usage

1. **Clone the repository:**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Prepare the Data**: Ensure that the historical OHLCV data for your trading pairs is available in the data/ directory (e.g., btc_1m_feb25.parquet). The data loader in core/data_loader.py can also download and cache data if configured.
4. **Run Backtests**: Execute the main script to run the backtests for all implemented strategies:
    ```bash
   python main.py
    ```
    This will:
     - Load and preprocess data.
     - Run each strategy’s backtest.
     - Generate visualizations (equity curves and heatmaps) saved as HTML and PNG files in the results/ directory.
     - Save aggregated metrics in CSV format.


## Running Tests
To run the unit tests, ensure you have pytest installed and then execute:
```bash
pytest
```
This will run all tests in the tests/ folder and output the results.

## Logging and Output
The project uses loguru for logging. When running tests or the main script, log messages (including nicely formatted tables using tabulate) will be printed to the console for easy debugging and reporting.

## Strategy description and test results
![Screenshot of the backtest results](/results/readme_backtest_results.png)

1. **SMACrossoverStrategy**

   Concept:

   - This strategy is based on a classic moving average crossover system. It computes two simple moving averages (typically a fast and a slow one) using historical price data. A buy signal is generated when the fast moving average crosses above the slow moving average, and a sell signal is generated when it crosses below. The underlying assumption is that such crossovers signal a change in momentum and trend continuation.

   Backtest Results Analysis:

     - Total Return: -92.53%
     The strategy has resulted in a substantial loss over the test period.

     - Sharpe Ratio: -50.17
     A very negative Sharpe ratio indicates that the risk-adjusted performance is extremely poor.

     - Max Drawdown: 93.66%
     The portfolio experienced a very deep drawdown, meaning the capital was severely eroded at its worst point.

     - Win Rate: 14.64%
     A low win rate suggests that winning trades were rare.

     - Expectancy: -17.39
     Negative expectancy indicates that, on average, each trade loses money.

     - Exposure Time: 47.83%
     The strategy was in the market about half of the time.

   Conclusion:
      - SMACrossoverStrategy performed very poorly, with high drawdowns and infrequent winning trades. This might suggest that the parameters used were not well tuned for the market conditions during the test period, or that the trend-following approach did not work well in a choppy or sideways market.

2. **RSIBBStrategy**

   Concept:
   - RSIBBStrategy combines the Relative Strength Index (RSI) with Bollinger Bands. The strategy typically enters a trade when the RSI falls below a threshold (commonly 30), indicating oversold conditions, and the price is near the lower Bollinger Band (suggesting a potential rebound). Conversely, it exits when the RSI rises above a threshold (commonly 70) or when the price approaches the upper Bollinger Band. The idea is to capture short-term mean reversion moves.

   Backtest Results Analysis:

   - Total Return: -57.06%
   While still negative, the loss is smaller compared to the SMA strategy.

   - Sharpe Ratio: -76.00
   An even more negative Sharpe ratio here indicates that the risk-adjusted performance is worse than expected.

   - Max Drawdown: 58.23%
   The drawdown is less severe than the SMA strategy, but still significant.

   - Win Rate: 0.40%
   The win rate is extremely low, implying almost no winning trades.

   - Expectancy: -26.36
   Each trade, on average, leads to a loss.

   - Exposure Time: 12.78%
   The strategy was active in the market only for a short period, suggesting that it rarely entered trades.

   Conclusion:
   - RSIBBStrategy appears to have been too conservative (or misconfigured) during the test period, resulting in very few trades and almost no wins. The low exposure time, combined with a very low win rate, indicates that either the thresholds were set too extreme or market conditions did not trigger the signals effectively.

3. VWAPReversionStrategy

   Concept:
   - VWAPReversionStrategy is a mean-reversion approach that uses the Volume-Weighted Average Price (VWAP) as a reference. The strategy enters trades when the current price significantly deviates from the VWAP, betting that the price will revert back towards it. This approach is often used to exploit temporary mispricings.

   Backtest Results Analysis:

   - Total Return: 5870.19%
   The strategy achieved an enormous gain, indicating extremely successful trades.

   - Sharpe Ratio: -3.26
   Despite the high return, the Sharpe ratio is negative, which is unusual. This could be due to extremely high volatility in returns or a potential issue with how the ratio is computed for this strategy.

   - Max Drawdown: 33.86%
   The drawdown is moderate compared to the other strategies, suggesting better downside protection.

   - Win Rate: 58.96%
   A win rate above 50% indicates that more than half of the trades were profitable.

   - Expectancy: 4171.61
   Extremely high expectancy implies that profitable trades yielded very large gains on average.

   - Exposure Time: 50.18%
   The strategy was active about half of the time, which is a typical exposure for a reversion strategy.

   Conclusion:
   - VWAPReversionStrategy outperformed the other strategies significantly, generating very high returns with a reasonable drawdown. However, the negative Sharpe ratio raises concerns about the volatility or risk profile of the returns, and it might require further investigation. Overall, the strategy appears to be highly effective under the test conditions, though risk-adjusted performance metrics should be carefully interpreted.

Summary
SMACrossoverStrategy: A trend-following method that resulted in deep losses and high drawdowns, indicating it did not perform well in the tested market conditions.

RSIBBStrategy: A mean-reversion approach using RSI and Bollinger Bands that rarely traded and produced poor results, suggesting that signal thresholds or market conditions were suboptimal.

VWAPReversionStrategy: A mean-reversion strategy based on VWAP that produced extraordinary returns with a moderate drawdown, although the negative Sharpe ratio suggests high return volatility.






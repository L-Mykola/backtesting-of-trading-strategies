import os
from pathlib import Path

import pandas as pd
from strategies.base import StrategyBase
import plotly.express as px
import plotly.graph_objects as go
import vectorbt as vbt


class Backtester:
    """
    A class for backtesting strategies on all trading pairs.
    """
    def __init__(self, strategy: StrategyBase, strategy_name: str, project_dir: Path, results_dir: str = "results"):
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.results_dir = os.path.join(project_dir, results_dir)
        self.screenshots_dir = os.path.join(results_dir, "screenshots")
        self.html_dir = os.path.join(results_dir, "html")

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
        if not os.path.exists(self.html_dir):
            os.makedirs(self.html_dir)

    def run(self):
        """
        Performs backtests, calculates metrics, and generates graphical results.
        return: result metrics
        """
        portfolio = self.strategy.run_backtest()
        self._plot_equity_curve(portfolio)
        self._plot_heatmap(portfolio)

        metrics_csv_path = os.path.join(self.results_dir, f"{self.strategy_name}_metrics.csv")
        metrics = self.strategy.get_metrics(metrics_csv_path)

        return metrics

    def _plot_equity_curve(self, portfolio: vbt.Portfolio):
        """
        Building an equity curve graph with Plotly.
        :param portfolio: backtest results
        """
        eq_curve = portfolio.value()

        if isinstance(eq_curve, (pd.DataFrame)):
            asset = eq_curve.columns[0]
            fig = px.line(eq_curve, x=eq_curve.index, y=asset, title="Equity Curve")
        elif isinstance(eq_curve, pd.Series):
            fig = px.line(eq_curve.reset_index(), x='index', y=eq_curve.name or 0, title="Equity Curve")
        else:
            dummy = pd.Series([eq_curve], index=[portfolio.wrapper.index[0]])
            fig = px.line(dummy.reset_index(), x='index', y=0, title="Equity Curve")

        html_filename = os.path.join(self.html_dir, f"{self.strategy_name}_equity_curve.html")
        fig.write_html(html_filename)
        png_filename = os.path.join(self.screenshots_dir, f"{self.strategy_name}_equity_curve.png")
        fig.write_image(png_filename)

    def _plot_heatmap(self, portfolio: vbt.Portfolio):
        """
        Build a heatmap with performance for all assets using Plotly.
        :param portfolio: backtest results
        """
        total_returns = portfolio.total_return()
        if isinstance(total_returns, pd.Series):
            total_returns = pd.DataFrame([total_returns.values], columns=total_returns.index)

        z = [total_returns.iloc[-1].values]
        x = total_returns.columns.tolist()
        y = ["Total Return"]

        fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale="Viridis"))
        fig.update_layout(title="Heatmap Total Returns")

        html_filename = os.path.join(self.html_dir, f"{self.strategy_name}_heatmap.html")
        fig.write_html(html_filename)
        png_filename = os.path.join(self.screenshots_dir, f"{self.strategy_name}_heatmap.png")
        fig.write_image(png_filename)
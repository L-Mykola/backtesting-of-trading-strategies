import pandas as pd
import numpy as np
import pytest
from strategies.sma_cross import SMACrossStrategy
from strategies.rsi_bb import RSIBBStrategy
from strategies.vwap_reversion import VWAPReversionStrategy


@pytest.fixture
def sample_data():
    rng = pd.date_range("2025-02-01", periods=50, freq="min")
    data = {}
    for symbol in ["PAIR1BTC", "PAIR2BTC"]:
        df = pd.DataFrame(index=rng)
        df["open"] = np.linspace(100, 105, 50)
        df["high"] = df["open"] + 0.5
        df["low"] = df["open"] - 0.5
        df["close"] = np.linspace(100, 105, 50)
        df["volume"] = 5
        data[symbol] = df
    combined = pd.concat(data, axis=1)
    return combined


def test_sma_crossover_signals(sample_data):
    strategy = SMACrossStrategy(price_data=sample_data, fast_window=3, slow_window=5, pairs=["PAIR1BTC", "PAIR2BTC"])
    signals = strategy.generate_signals()
    assert "entries" in signals and "exits" in signals
    close = sample_data.xs("close", level=1, axis=1)
    assert signals["entries"].shape == close.shape


def test_rsi_bb_signals(sample_data):
    strategy = RSIBBStrategy(
        price_data=sample_data,
        rsi_period=14,
        bb_window=20,
        bb_std=2,
        pairs=["PAIR1BTC", "PAIR2BTC"])
    signals = strategy.generate_signals()
    assert "entries" in signals and "exits" in signals
    close = sample_data.xs("close", level=1, axis=1)
    assert signals["entries"].shape == close.shape


def test_vwap_reversion_signals(sample_data):
    strategy = VWAPReversionStrategy(price_data=sample_data, threshold=0.01, pairs=["PAIR1BTC", "PAIR2BTC"])
    signals = strategy.generate_signals()
    assert "entries" in signals and "exits" in signals
    close = sample_data.xs("close", level=1, axis=1)
    assert signals["entries"].shape == close.shape

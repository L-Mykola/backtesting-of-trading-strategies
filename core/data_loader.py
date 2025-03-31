import os
import time
from typing import List, Tuple

import pandas as pd
import ccxt
from loguru import logger


class DataLoader:

    def __init__(self, project_dir: os.path, start_date: str, end_date: str, data_dir: str = "data"):
        self.project_dir = project_dir
        self.data_dir = data_dir
        self.output_folder = os.path.join(self.project_dir, self.data_dir)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        self.pairs = []

    def get_top_liquid_pairs(self, n) -> list:
        """
        Gets the top n liquid pairs to BTC using ticker data from Binance
        :param n: num of pairs
        :return: list of top n liquid pairs to BTC
        """
        markets = self.exchange.load_markets()
        tickers = self.exchange.fetch_tickers()
        btc_pairs = []
        for symbol, market in markets.items():
            if '/BTC' in symbol:
                ticker = tickers.get(symbol)
                if ticker and ticker.get('quoteVolume'):
                    btc_pairs.append((symbol, ticker['quoteVolume']))
        btc_pairs = sorted(btc_pairs, key=lambda x: x[1], reverse=True)
        top_pairs = [pair[0] for pair in btc_pairs[:n]]
        self.pairs = top_pairs.copy()
        return top_pairs

    def fetch_ohlcv_for_symbol(self, symbol: str, timeframe: str = '1m') -> pd.DataFrame:
        """
        Downloads OHLCV data for a given trading pair for a specified period.
        :param symbol: symbol
        :param timeframe: timeframe
        :return: dataframe of OHLCV data for a given trading pair for a specified period
        """
        since = int(self.start_date.timestamp() * 1000)
        end_time = int(self.end_date.timestamp() * 1000)
        all_ohlcv = []
        while since < end_time:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                self.pairs.remove(symbol)
                break
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            if last_timestamp == since:
                break
            since = last_timestamp + 1
            time.sleep(self.exchange.rateLimit / 1000)

        complete_index = pd.date_range(self.start_date, self.end_date, freq='min')

        if not all_ohlcv:
            raise ValueError(f"Could not load data for {symbol}")

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        df = df.reindex(complete_index)
        df.fillna(0, inplace=True)
        return df

    def download_data(self, n) -> pd.DataFrame:
        """
        Uploads data for the top 100 pairs to BTC and returns a summary DataFrame with multi-index columns
        :param n: num of pairs
        :return: summary DataFrame with multi-index columns
        """
        top_pairs = self.get_top_liquid_pairs(n)
        logger.info(f"Топ {n} pairs: {top_pairs}")
        data_frames = []
        for symbol in top_pairs:
            logger.info(f"Loading data for {symbol}...")
            try:
                df = self.fetch_ohlcv_for_symbol(symbol)
                df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                data_frames.append(df)
            except Exception as e:
                logger.error(f"Missing {symbol} through the error: {e}")
                self.pairs.remove(symbol)
        if data_frames:
            combined = pd.concat(data_frames, axis=1)
            combined.sort_index(inplace=True)
            return combined
        else:
            raise ValueError("Unable to download data for any pair")

    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Saves DataFrame in Parquet format with compression
        :param df: pandas DataFrame
        :param filename: file to save data
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        filepath = os.path.join(self.output_folder, filename)
        df.to_parquet(filepath, compression='snappy')

    def load_cached_data(self, filename: str) -> pd.DataFrame | None:
        """
        Loads data from the cache if the file exists.
        :param filename: name of cache file
        :return: dataframe with multi-index columns of OHLCV data or None
        """
        filepath = os.path.join(self.output_folder, filename)
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        return None

    @staticmethod
    def check_data_integrity(df: pd.DataFrame) -> bool:
        """
        Checks data integrity (no missing values).
        :param df: pandas DataFrame
        :return: bool
        """
        return not df.isnull().any().any()

    def process(self, num_of_pairs: int, filename: str = "btc_1m_feb25.parquet") -> Tuple[List, pd.DataFrame]:
        """
        The main method for downloading, processing, and caching data.
        :param num_of_pairs: number of pairs
        :param filename: name of cache file
        :return: result pandas dataframe
        """
        cached_df = self.load_cached_data(filename)
        if cached_df is not None and self.check_data_integrity(cached_df):
            logger.info("Use data from the cache!")
            pairs = cached_df.columns.get_level_values(0).unique().tolist()
            logger.info(f"Pairs: {pairs}")
            return pairs, cached_df

        df = self.download_data(num_of_pairs)
        if not self.check_data_integrity(df):
            raise ValueError("Data has integrity issues.")
        self.save_data(df, filename)
        return self.pairs, df



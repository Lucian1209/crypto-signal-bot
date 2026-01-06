"""
Binance Data Collector
Collects OHLCV data and calculates technical indicators
"""

import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
from typing import Optional, List


class BinanceCollector:
    """Collect and process data from Binance (lazy client init)"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client: Optional[Client] = None

    def _get_client(self) -> Client:
        """Lazy initialization of Binance client"""
        if self.client is None:
            if self.api_key and self.api_secret:
                self.client = Client(self.api_key, self.api_secret)
            else:
                self.client = Client()
        return self.client

    def get_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days_back: int = 30
    ) -> pd.DataFrame:

        client = self._get_client()

        try:
            start_time = datetime.utcnow() - timedelta(days=days_back)
            start_str = start_time.strftime("%d %b %Y %H:%M:%S")

            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df[['open', 'high', 'low', 'close', 'volume']]

        except BinanceAPIException as e:
            raise RuntimeError(f"Binance API error: {e}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['rsi'] = ta.rsi(df['close'], length=14)

        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]
            df['macd_signal'] = macd.iloc[:, 1]
            df['macd_histogram'] = macd.iloc[:, 2]

        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_50'] = ta.ema(df['close'], length=50)

        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['bb_lower'] = bb.iloc[:, 0]
            df['bb_middle'] = bb.iloc[:, 1]
            df['bb_upper'] = bb.iloc[:, 2]

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        df['price_change_1h'] = df['close'].pct_change(1)
        df['price_change_24h'] = df['close'].pct_change(24)

        df.dropna(inplace=True)
        return df

    def get_current_price(self, symbol: str) -> dict:
        client = self._get_client()

        ticker = client.get_ticker(symbol=symbol)
        return {
            "symbol": symbol,
            "price": float(ticker["lastPrice"]),
            "change_24h": float(ticker["priceChangePercent"]),
            "volume_24h": float(ticker["volume"]),
            "high_24h": float(ticker["highPrice"]),
            "low_24h": float(ticker["lowPrice"]),
        }

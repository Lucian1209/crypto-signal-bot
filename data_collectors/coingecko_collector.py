"""
CoinGecko Data Collector - альтернатива Binance без геоблокування
Використовує публічний API CoinGecko
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class CoinGeckoCollector:
    """
    Збирає дані з CoinGecko API
    + Без геоблокінгу
    + Безкоштовний
    - Обмеження: 10-50 запитів/хвилину
    """

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'
        })

        # Mapping symbols to CoinGecko IDs
        self.symbol_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'BNBUSDT': 'binancecoin',
            'ADAUSDT': 'cardano',
            'SOLUSDT': 'solana',
            'XRPUSDT': 'ripple',
            'DOGEUSDT': 'dogecoin',
            'MATICUSDT': 'matic-network',
            'DOTUSDT': 'polkadot',
            'LTCUSDT': 'litecoin',
            'AVAXUSDT': 'avalanche-2',
            'LINKUSDT': 'chainlink',
            'UNIUSDT': 'uniswap',
            'ATOMUSDT': 'cosmos',
            'XMRUSDT': 'monero',
        }

        logger.info("CoinGecko Collector initialized")

    def _get_coin_id(self, symbol: str) -> str:
        """Конвертує BTCUSDT -> bitcoin"""
        symbol = symbol.upper().replace('USDT', 'USDT')

        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        # Якщо невідомий символ, спробуємо пошукати
        try:
            search_url = f"{self.base_url}/search"
            response = self.session.get(search_url, params={'query': symbol[:3]})
            data = response.json()

            if data.get('coins'):
                coin_id = data['coins'][0]['id']
                logger.info(f"Found coin ID for {symbol}: {coin_id}")
                self.symbol_map[symbol] = coin_id
                return coin_id
        except Exception as e:
            logger.error(f"Error finding coin ID: {e}")

        raise ValueError(f"Unknown symbol: {symbol}")

    def get_current_price(self, symbol: str) -> dict:
        """
        Отримати поточну ціну
        """
        try:
            coin_id = self._get_coin_id(symbol)

            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if coin_id not in data:
                raise ValueError(f"No data for {coin_id}")

            coin_data = data[coin_id]

            return {
                'symbol': symbol,
                'price': coin_data['usd'],
                'change_24h': coin_data.get('usd_24h_change', 0),
                'volume_24h': coin_data.get('usd_24h_vol', 0)
            }

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, interval: str = "1h", days_back: int = 3) -> pd.DataFrame:
        """
        Отримати історичні дані

        interval: не використовується для CoinGecko (завжди погодинні дані)
        days_back: скільки днів історії
        """
        try:
            coin_id = self._get_coin_id(symbol)

            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days_back,
                'interval': 'hourly'
            }

            logger.info(f"Fetching {days_back} days of data for {symbol}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Конвертувати в DataFrame
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])

            if not prices:
                raise ValueError(f"No price data for {symbol}")

            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Додати volume
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                df = df.merge(vol_df, on='timestamp', how='left')
            else:
                df['volume'] = 0

            # Створити OHLC з close (апроксимація)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * 1.001  # Додати 0.1% варіації
            df['low'] = df[['open', 'close']].min(axis=1) * 0.999

            # Сортувати по часу
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Заповнити NaN
            df = df.fillna(method='ffill').fillna(method='bfill')

            logger.info(f"Got {len(df)} rows for {symbol}")

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахувати технічні індикатори
        """
        df = df.copy()

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], period=14)

        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # EMA
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Заповнити NaN
        df = df.fillna(method='bfill').fillna(0)

        logger.info("Technical indicators calculated")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI розрахунок"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def test_connection(self) -> bool:
        """Тест підключення"""
        try:
            url = f"{self.base_url}/ping"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Тест
    logging.basicConfig(level=logging.INFO)

    collector = CoinGeckoCollector()

    print("Testing connection...")
    if collector.test_connection():
        print("✓ Connected to CoinGecko")

    print("\nGetting BTC price...")
    price = collector.get_current_price("BTCUSDT")
    print(f"BTC: ${price['price']:,.2f} ({price['change_24h']:+.2f}%)")

    print("\nGetting historical data...")
    df = collector.get_historical_data("BTCUSDT", days_back=2)
    print(f"Got {len(df)} rows")
    print(df.head())

    print("\nCalculating indicators...")
    df = collector.calculate_technical_indicators(df)
    print(f"RSI: {df.iloc[-1]['rsi']:.2f}")
    print(f"MACD: {df.iloc[-1]['macd']:.4f}")

"""
CryptoCompare Data Collector - безкоштовний без API ключа
Альтернатива CoinGecko для Railway deployment
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class CryptoCompareCollector:
    """
    Збирає дані з CryptoCompare API
    + Безкоштовний без API ключа
    + Без геоблокінгу
    + Погодинні та хвилинні дані
    """

    def __init__(self):
        self.base_url = "https://min-api.cryptocompare.com/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'
        })

        # Symbol mapping
        self.symbol_map = {
            'BTCUSDT': 'BTC',
            'ETHUSDT': 'ETH',
            'BNBUSDT': 'BNB',
            'ADAUSDT': 'ADA',
            'SOLUSDT': 'SOL',
            'XRPUSDT': 'XRP',
            'DOGEUSDT': 'DOGE',
            'MATICUSDT': 'MATIC',
            'DOTUSDT': 'DOT',
            'LTCUSDT': 'LTC',
            'AVAXUSDT': 'AVAX',
            'LINKUSDT': 'LINK',
            'UNIUSDT': 'UNI',
            'ATOMUSDT': 'ATOM',
            'XMRUSDT': 'XMR',
        }

        logger.info("CryptoCompare Collector initialized (no API key required)")

    def _get_symbol(self, trading_pair: str) -> str:
        """Конвертує BTCUSDT -> BTC"""
        if trading_pair in self.symbol_map:
            return self.symbol_map[trading_pair]

        # Спробувати видалити USDT
        if trading_pair.endswith('USDT'):
            symbol = trading_pair[:-4]
            self.symbol_map[trading_pair] = symbol
            return symbol

        return trading_pair

    def get_current_price(self, symbol: str) -> dict:
        """Отримати поточну ціну"""
        try:
            crypto_symbol = self._get_symbol(symbol)

            url = f"{self.base_url}/pricemultifull"
            params = {
                'fsyms': crypto_symbol,
                'tsyms': 'USD'
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'RAW' not in data or crypto_symbol not in data['RAW']:
                raise ValueError(f"No data for {symbol}")

            price_data = data['RAW'][crypto_symbol]['USD']

            return {
                'symbol': symbol,
                'price': float(price_data['PRICE']),
                'change_24h': float(price_data.get('CHANGEPCT24HOUR', 0)),
                'volume_24h': float(price_data.get('VOLUME24HOURTO', 0))
            }

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, interval: str = "1h", days_back: int = 3) -> pd.DataFrame:
        """
        Отримати історичні дані

        interval: '1h', '1m', '5m', '15m', '30m', '1d'
        days_back: скільки днів історії
        """
        try:
            crypto_symbol = self._get_symbol(symbol)

            # Визначити endpoint і параметри
            if interval in ['1m', '5m', '15m', '30m']:
                endpoint = 'histominute'
                if interval == '1m':
                    limit = days_back * 24 * 60
                elif interval == '5m':
                    limit = days_back * 24 * 12
                elif interval == '15m':
                    limit = days_back * 24 * 4
                else:  # 30m
                    limit = days_back * 24 * 2
                limit = min(limit, 2000)  # API limit
            elif interval == '1d':
                endpoint = 'histoday'
                limit = days_back
            else:  # Default to hourly
                endpoint = 'histohour'
                limit = days_back * 24
                limit = min(limit, 2000)  # API limit

            url = f"{self.base_url}/v2/{endpoint}"
            params = {
                'fsym': crypto_symbol,
                'tsym': 'USD',
                'limit': limit
            }

            logger.info(f"Fetching {days_back} days of {interval} data for {symbol}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Error':
                raise ValueError(f"API Error: {data.get('Message')}")

            # Конвертувати в DataFrame
            candles = data['Data']['Data']

            if not candles:
                raise ValueError(f"No data for {symbol}")

            df = pd.DataFrame(candles)

            # Перейменувати колонки
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumeto': 'volume'  # Volume in USD
            })

            # Конвертувати timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Вибрати потрібні колонки
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Заповнити NaN
            df = df.fillna(method='ffill').fillna(method='bfill')

            logger.info(f"Got {len(df)} rows for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахувати технічні індикатори"""
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
            url = f"{self.base_url}/price"
            params = {'fsym': 'BTC', 'tsyms': 'USD'}
            response = self.session.get(url, params=params, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Тест
    logging.basicConfig(level=logging.INFO)

    collector = CryptoCompareCollector()

    print("Testing connection...")
    if collector.test_connection():
        print("✓ Connected to CryptoCompare")

    print("\nGetting BTC price...")
    price = collector.get_current_price("BTCUSDT")
    print(f"BTC: ${price['price']:,.2f} ({price['change_24h']:+.2f}%)")

    print("\nGetting historical data...")
    df = collector.get_historical_data("BTCUSDT", interval="1h", days_back=2)
    print(f"Got {len(df)} rows")
    print(df.head())
    print(df.tail())

    print("\nCalculating indicators...")
    df = collector.calculate_technical_indicators(df)
    print(f"RSI: {df.iloc[-1]['rsi']:.2f}")
    print(f"MACD: {df.iloc[-1]['macd']:.4f}")

    print("\n✓ ALL TESTS PASSED!")

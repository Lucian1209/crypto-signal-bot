"""
Binance Data Collector з підтримкою API ключів
Працює БЕЗ ключів локально, З ключами на Railway
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import time

logger = logging.getLogger(__name__)


class BinanceCollector:
    """
    Збирає дані з Binance
    Підтримує роботу БЕЗ API ключів (локально) та З ключами (Railway)
    """

    def __init__(self):
        # Отримати API ключі з environment
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        try:
            if api_key and api_secret:
                # З автентифікацією (для Railway)
                self.client = Client(api_key, api_secret)
                self.authenticated = True
                logger.info("✓ Binance Client initialized WITH API key")
                logger.info("  Recommended for Railway/Cloud deployments")
            else:
                # Без автентифікації (локально, може бути geo-blocked)
                self.client = Client()
                self.authenticated = False
                logger.info("✓ Binance Client initialized WITHOUT API key")
                logger.warning("  May be geo-blocked on some servers")
                logger.info("  For Railway, add BINANCE_API_KEY and BINANCE_API_SECRET")

            # Test connection
            self.client.ping()
            logger.info("✓ Binance connection successful")

        except BinanceAPIException as e:
            logger.error(f"✗ Binance connection failed: {e}")
            if "restricted location" in str(e).lower():
                logger.error("")
                logger.error("=" * 70)
                logger.error("BINANCE GEO-BLOCKED ERROR")
                logger.error("=" * 70)
                logger.error("Solutions:")
                logger.error("1. Add Binance API keys (recommended for Railway):")
                logger.error("   BINANCE_API_KEY=your_key")
                logger.error("   BINANCE_API_SECRET=your_secret")
                logger.error("   Get keys: https://www.binance.com/ → API Management")
                logger.error("")
                logger.error("2. Or use alternative data source:")
                logger.error("   DATA_SOURCE=cryptocompare")
                logger.error("=" * 70)
            raise

    def get_current_price(self, symbol: str) -> dict:
        """Отримати поточну ціну"""
        try:
            ticker = self.client.get_ticker(symbol=symbol)

            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'change_24h': float(ticker['priceChangePercent']),
                'volume_24h': float(ticker['quoteVolume'])
            }
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise

    def get_historical_data(self, symbol: str, interval: str = "1h", days_back: int = 3) -> pd.DataFrame:
        """
        Отримати історичні дані

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Candlestick interval
                1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            days_back: How many days of history

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Calculate start time
            start_time = datetime.now() - timedelta(days=days_back)
            start_str = start_time.strftime("%d %b %Y %H:%M:%S")

            logger.info(f"Fetching {days_back} days of {interval} data for {symbol}...")

            # Get klines (candlestick data)
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )

            if not klines:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Select needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"✓ Got {len(df)} rows for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахувати технічні індикатори

        Calculates:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - EMA (Exponential Moving Averages)
        - Bollinger Bands
        - Volume indicators
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

        # Fill NaN
        df = df.bfill().fillna(0)

        logger.info("✓ Technical indicators calculated")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def test_connection(self) -> bool:
        """Test connection to Binance"""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Тест
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    print("="*70)
    print("TESTING BINANCE COLLECTOR")
    print("="*70)

    # Check if API keys are set
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if api_key and api_secret:
        print(f"\n✓ API Key found: {api_key[:10]}...")
        print("✓ API Secret found")
        print("\nTesting WITH authentication...")
    else:
        print("\n⚠ No API keys found")
        print("Testing WITHOUT authentication (may fail if geo-blocked)")
        print("\nTo test with keys:")
        print("  export BINANCE_API_KEY='your_key'")
        print("  export BINANCE_API_SECRET='your_secret'")

    print("\n" + "-"*70 + "\n")

    try:
        collector = BinanceCollector()

        print("\n1. Getting BTC price...")
        price = collector.get_current_price("BTCUSDT")
        print(f"✓ BTC: ${price['price']:,.2f} ({price['change_24h']:+.2f}%)")
        print(f"  24h Volume: ${price['volume_24h']:,.0f}")

        print("\n2. Getting historical data (1h, 2 days)...")
        df = collector.get_historical_data("BTCUSDT", interval="1h", days_back=2)
        print(f"✓ Got {len(df)} rows")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nLast 3 rows:")
        print(df.tail(3))

        print("\n3. Calculating indicators...")
        df = collector.calculate_technical_indicators(df)
        latest = df.iloc[-1]
        print(f"✓ RSI: {latest['rsi']:.2f}")
        print(f"✓ MACD: {latest['macd']:.4f}")
        print(f"✓ EMA(9): ${latest['ema_9']:,.2f}")
        print(f"✓ EMA(21): ${latest['ema_21']:,.2f}")

        print("\n4. Testing different intervals...")
        for interval in ['5m', '15m', '1h', '4h', '1d']:
            df_test = collector.get_historical_data("BTCUSDT", interval=interval, days_back=1)
            print(f"✓ {interval}: {len(df_test)} candles")

        print("\n5. Testing other symbols...")
        for symbol in ['ETHUSDT', 'BNBUSDT']:
            price_test = collector.get_current_price(symbol)
            print(f"✓ {symbol}: ${price_test['price']:,.2f}")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)

        if api_key and api_secret:
            print("\nBinance with API keys is working!")
            print("Ready for Railway deployment 🚀")
        else:
            print("\nBinance without API keys is working locally!")
            print("For Railway, add API keys to avoid geo-blocking")

    except BinanceAPIException as e:
        if "restricted location" in str(e).lower():
            print("\n" + "="*70)
            print("✗ GEO-BLOCKED ERROR")
            print("="*70)
            print("\nBinance is blocked in your location.")
            print("\nSolutions:")
            print("1. Get Binance API keys (recommended):")
            print("   - https://www.binance.com/ → API Management")
            print("   - Create Read-only API key")
            print("   - Set environment variables:")
            print("     export BINANCE_API_KEY='your_key'")
            print("     export BINANCE_API_SECRET='your_secret'")
            print("\n2. Use alternative data source:")
            print("   - CryptoCompare (no API key): DATA_SOURCE=cryptocompare")
            print("   - CoinGecko (free API key): DATA_SOURCE=coingecko")
        else:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

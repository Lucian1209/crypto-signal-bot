import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

print(f"BASE_DIR: {BASE_DIR}")
print(f"Python path: {sys.path}")

try:
    from data_collectors.binance_collector import BinanceCollector
    print("✓ BinanceCollector imported")
    collector = BinanceCollector()
    print("✓ BinanceCollector initialized")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

"""
ML Signal Generation Service - Production Ready
Підтримує Binance, CoinGecko та CryptoCompare
За замовчуванням використовує CryptoCompare (БЕЗ API KEY!)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import json
import logging
import sys
import os
import traceback
from pathlib import Path

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "signal_service.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("signal_service")

# -----------------------------------------------------------------------------
# PATH FIX
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
logger.info(f"BASE_DIR: {BASE_DIR}")

# -----------------------------------------------------------------------------
# IMPORTS - Dynamic data source
# -----------------------------------------------------------------------------

DATA_SOURCE = os.getenv("DATA_SOURCE", "cryptocompare")  # binance, coingecko, or cryptocompare

try:
    if DATA_SOURCE == "binance":
        from data_collectors.binance_collector import BinanceCollector as DataCollector
        logger.info("✓ BinanceCollector will be used (best for local)")
    elif DATA_SOURCE == "coingecko":
        from data_collectors.coingecko_collector import CoinGeckoCollector as DataCollector
        logger.info("✓ CoinGeckoCollector will be used (requires API key)")
    else:  # cryptocompare (default)
        from data_collectors.cryptocompare_collector import CryptoCompareCollector as DataCollector
        logger.info("✓ CryptoCompareCollector will be used (no API key needed, perfect for Railway)")
except Exception as e:
    logger.error(f"✗ DataCollector import failed: {e}")
    logger.exception("Full import error:")
    raise

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Crypto Signal ML Service",
    description="ML-powered crypto trading signal generation with multiple data sources",
    version="2.1.0"
)

# -----------------------------------------------------------------------------
# GLOBALS - Lazy loading
# -----------------------------------------------------------------------------

model = None
scaler = None
feature_names = None
data_collector = None
ml_loaded = False
ml_load_attempted = False

# -----------------------------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------------------------

class SignalRequest(BaseModel):
    symbol: str
    interval: str = "1h"

class SignalResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    analysis: str
    timestamp: str
    technical_indicators: dict
    mode: str  # "ml" or "technical"
    data_source: str  # "binance", "coingecko" or "cryptocompare"

# -----------------------------------------------------------------------------
# LAZY ML LOADER
# -----------------------------------------------------------------------------

def load_ml_model():
    """Завантажує ML модель тільки коли потрібно"""
    global model, scaler, feature_names, ml_loaded, ml_load_attempted

    if ml_load_attempted:
        return ml_loaded

    ml_load_attempted = True

    # Check if ML is disabled
    if os.getenv("DISABLE_ML", "false").lower() == "true":
        logger.info("ML DISABLED via DISABLE_ML env variable")
        ml_loaded = False
        return False

    logger.info("=" * 70)
    logger.info("Attempting to load ML model...")

    model_path = os.getenv("MODEL_PATH", "models/crypto_signal_model.h5")
    scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
    features_path = os.getenv("FEATURES_PATH", "models/feature_names.json")

    logger.info(f"Model path: {Path(model_path).resolve()}")
    logger.info(f"Scaler path: {Path(scaler_path).resolve()}")
    logger.info(f"Features path: {Path(features_path).resolve()}")

    # CHECK FILES FIRST
    model_exists = Path(model_path).exists()
    scaler_exists = Path(scaler_path).exists()
    features_exists = Path(features_path).exists()

    logger.info(f"Files check: Model={model_exists}, Scaler={scaler_exists}, Features={features_exists}")

    if not (model_exists and scaler_exists and features_exists):
        logger.warning("✗ Not all ML files exist - SKIPPING ML LOAD")
        logger.warning("Will use technical analysis fallback")
        logger.info("=" * 70)
        ml_loaded = False
        return False

    # FILES EXIST - TRY TO LOAD
    try:
        logger.info("All files exist, proceeding with load...")

        # Load features FIRST (smallest file)
        logger.info("Loading feature names...")
        with open(features_path, "r") as f:
            feature_names = json.load(f)
        logger.info(f"✓ Features loaded: {len(feature_names)} features")

        # Load scaler (medium file)
        logger.info("Loading scaler...")
        import joblib
        scaler = joblib.load(scaler_path)
        logger.info(f"✓ Scaler loaded: {scaler_path}")

        # Load TensorFlow model (largest, slowest)
        logger.info("Importing TensorFlow...")
        import tensorflow as tf
        logger.info(f"✓ TensorFlow {tf.__version__} imported")

        logger.info("Loading Keras model (this may take 10-30 seconds)...")
        logger.info("Setting TensorFlow to minimal logging...")
        tf.get_logger().setLevel('ERROR')

        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"✓ Model loaded: {model_path}")

        ml_loaded = True
        logger.info("✓ ML MODEL FULLY LOADED")
        logger.info("=" * 70)
        return True

    except Exception as e:
        logger.error(f"✗ ML loading failed: {e}")
        logger.error(traceback.format_exc())
        ml_loaded = False
        logger.info("Will use technical fallback instead")
        logger.info("=" * 70)
        return False

# -----------------------------------------------------------------------------
# STARTUP
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global data_collector

    logger.info("=" * 70)
    logger.info(f"STARTING CRYPTO SIGNAL SERVICE")
    logger.info(f"Data Source: {DATA_SOURCE}")
    logger.info("=" * 70)

    # Ініціалізувати data collector
    try:
        data_collector = DataCollector()
        logger.info(f"✓ {DATA_SOURCE.title()} Collector initialized")

        # Test connection
        if hasattr(data_collector, 'test_connection'):
            if data_collector.test_connection():
                logger.info(f"✓ {DATA_SOURCE.title()} connection test OK")
            else:
                logger.warning(f"⚠ {DATA_SOURCE.title()} connection test failed")

        # Test price fetch
        test = data_collector.get_current_price("BTCUSDT")
        logger.info(f"✓ Test price fetch OK: BTC = ${test['price']:,.2f}")

    except Exception as e:
        logger.error(f"✗ Data collector init failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Cannot start without data collector: {e}")

    logger.info("✓ SERVICE STARTED (ML will load on first request)")
    logger.info("=" * 70)

# -----------------------------------------------------------------------------
# HEALTH
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Crypto Signal ML Service",
        "version": "2.1.0",
        "status": "running",
        "data_source": DATA_SOURCE,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "data_source": DATA_SOURCE,
        "data_collector_ready": data_collector is not None,
        "ml_load_attempted": ml_load_attempted,
        "ml_loaded": ml_loaded,
        "ml_available": model is not None and scaler is not None and feature_names is not None,
        "mode": "ml" if (model and scaler and feature_names) else "technical_analysis"
    }

# -----------------------------------------------------------------------------
# PREDICT
# -----------------------------------------------------------------------------

@app.post("/predict", response_model=SignalResponse)
async def predict(request: SignalRequest):
    """
    Generate trading signal for given symbol

    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
        interval: Timeframe (1h, 4h, 1d, etc.)

    Returns:
        SignalResponse with action (BUY/SELL), confidence, prices, and indicators
    """
    try:
        symbol = request.symbol.upper()
        logger.info("=" * 70)
        logger.info(f"NEW REQUEST: {symbol} | {request.interval}")

        if data_collector is None:
            logger.error("Data collector not available!")
            raise HTTPException(503, "Data collector not available")

        # Get data
        logger.info("Fetching historical data...")
        df = data_collector.get_historical_data(
            symbol=symbol,
            interval=request.interval,
            days_back=3
        )
        logger.info(f"✓ Got {len(df)} data rows")

        if len(df) < 30:
            raise HTTPException(400, f"Not enough data: got {len(df)} rows, need at least 30")

        logger.info("Calculating technical indicators...")
        df = data_collector.calculate_technical_indicators(df)
        logger.info("✓ Technical indicators calculated")

        logger.info("Fetching current price...")
        price_data = data_collector.get_current_price(symbol)
        current_price = float(price_data["price"])
        logger.info(f"✓ Current price: ${current_price:,.2f}")

        # Generate signal
        logger.info("Determining signal generation method...")
        ml_ready = load_ml_model()

        if ml_ready:
            logger.info("Using ML prediction...")
            signal = predict_with_ml(df)
            mode = "ml"
        else:
            logger.info("Using technical analysis fallback...")
            signal = predict_with_technical_analysis(df)
            mode = "technical"

        logger.info(f"✓ Signal generated: {signal['action']} (confidence: {signal['confidence']:.2f})")

        # Risk management
        if signal["action"] == "BUY":
            stop_loss = current_price * 0.97   # -3%
            take_profit = current_price * 1.05  # +5%
        else:
            stop_loss = current_price * 1.03   # +3%
            take_profit = current_price * 0.95  # -5%

        latest = df.iloc[-1]

        response = SignalResponse(
            symbol=symbol,
            action=signal["action"],
            confidence=round(signal["confidence"], 4),
            entry_price=round(current_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            analysis=signal["analysis"],
            timestamp=datetime.utcnow().isoformat(),
            technical_indicators={
                "rsi": float(latest.get("rsi", 0)),
                "macd": float(latest.get("macd", 0)),
                "ema_9": float(latest.get("ema_9", 0)),
                "ema_21": float(latest.get("ema_21", 0)),
            },
            mode=mode,
            data_source=DATA_SOURCE
        )

        logger.info("✓ REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"✗ ERROR in predict: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 70)
        raise HTTPException(500, f"Internal error: {str(e)}")

# -----------------------------------------------------------------------------
# ML PREDICTION
# -----------------------------------------------------------------------------

def predict_with_ml(df: pd.DataFrame) -> dict:
    """ML-based prediction using trained model"""
    try:
        sequence_length = 24

        logger.info(f"Preparing data for ML (sequence_length={sequence_length})...")

        # Check if we have enough data
        if len(df) < sequence_length:
            raise ValueError(f"Not enough data: {len(df)} < {sequence_length}")

        # Check if all features exist
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X = df[feature_names].values[-sequence_length:]
        logger.info(f"✓ Data shape: {X.shape}")

        logger.info("Scaling data...")
        X_scaled = scaler.transform(X)

        logger.info("Reshaping data...")
        X_seq = X_scaled.reshape(1, sequence_length, len(feature_names))
        logger.info(f"✓ Final shape: {X_seq.shape}")

        logger.info("Running model prediction...")
        proba = float(model.predict(X_seq, verbose=0)[0][0])
        logger.info(f"✓ Raw prediction: {proba:.4f}")

        action = "BUY" if proba > 0.5 else "SELL"
        confidence = proba if action == "BUY" else 1 - proba

        return {
            "action": action,
            "confidence": confidence,
            "analysis": f"ML probability={proba:.2%}"
        }

    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        logger.error(traceback.format_exc())
        raise

# -----------------------------------------------------------------------------
# TECH FALLBACK
# -----------------------------------------------------------------------------

def predict_with_technical_analysis(df: pd.DataFrame) -> dict:
    """Technical analysis-based prediction (fallback when ML unavailable)"""
    try:
        latest = df.iloc[-1]
        rsi = latest.get("rsi", 50)
        macd = latest.get("macd", 0)
        macd_signal = latest.get("macd_signal", 0)

        logger.info(f"Technical indicators: RSI={rsi:.2f}, MACD={macd:.4f}")

        score = 0
        reasons = []

        # RSI analysis
        if rsi < 30:
            score += 2
            reasons.append("RSI oversold")
            logger.info("RSI < 30: +2 (oversold)")
        elif rsi > 70:
            score -= 2
            reasons.append("RSI overbought")
            logger.info("RSI > 70: -2 (overbought)")
        elif rsi < 45:
            score += 1
            reasons.append("RSI low")
        elif rsi > 55:
            score -= 1
            reasons.append("RSI high")

        # MACD analysis
        if macd > macd_signal:
            score += 1
            reasons.append("MACD bullish")
            logger.info("MACD > Signal: +1 (bullish)")
        else:
            score -= 1
            reasons.append("MACD bearish")
            logger.info("MACD < Signal: -1 (bearish)")

        # EMA trend
        ema_9 = latest.get("ema_9", 0)
        ema_21 = latest.get("ema_21", 0)
        if ema_9 > ema_21:
            score += 1
            reasons.append("EMA uptrend")
        else:
            score -= 1
            reasons.append("EMA downtrend")

        action = "BUY" if score >= 2 else "SELL"
        confidence = min(0.5 + abs(score) * 0.1, 0.95)

        analysis = f"Score={score} ({', '.join(reasons)}), RSI={rsi:.1f}"

        logger.info(f"Final score: {score} -> {action} (confidence: {confidence:.2f})")

        return {
            "action": action,
            "confidence": confidence,
            "analysis": analysis
        }

    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        logger.error(traceback.format_exc())
        raise

# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

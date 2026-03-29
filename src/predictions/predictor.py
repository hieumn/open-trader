"""Multi-signal prediction engine.

Combines several orthogonal models to forecast next-day price direction
and magnitude. Each model votes independently; the ensemble averages
their signals weighted by recent accuracy.

Models:
  1. Momentum regression  — linear regression on 5-day returns → extrapolate
  2. Mean-reversion       — RSI + Bollinger %B extremes → contrarian signal
  3. Trend-following      — MA stack alignment + MACD direction
  4. Volume-price         — OBV trend + unusual volume detection
  5. Volatility regime    — ATR contraction → breakout anticipation

Output: direction (-1 to +1), confidence (0–100%), predicted move (%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features.indicators import (
    ema, sma, rsi, macd, atr, bollinger_bands, adx, pivot_points,
)


# ─── Feature engineering ──────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical features needed by the prediction models."""
    df = df.copy()

    # EMAs
    for n in [10, 20, 50, 100, 150, 200]:
        df[f"ema{n}"] = ema(df, n)

    # Momentum
    df["rsi"] = rsi(df, 14)
    _m = macd(df, 12, 26, 9)
    df["macd"] = _m["macd"]
    df["macd_signal"] = _m["macd_signal"]
    df["macd_hist"] = _m["macd_hist"]
    df["adx"] = adx(df, 14)

    # Volatility
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"]
    _bb = bollinger_bands(df, 20, 2.0)
    df["bb_pct"] = _bb["bb_pct"]
    df["bb_upper"] = _bb["bb_upper"]
    df["bb_lower"] = _bb["bb_lower"]
    df["bb_mid"] = _bb["bb_mid"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Volume
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()

    # Price changes
    for n in [1, 3, 5, 10]:
        df[f"ret_{n}d"] = df["close"].pct_change(n)

    # Pivots
    _piv = pivot_points(df)
    df["pivot"] = _piv["pivot"]
    df["r1"] = _piv["r1"]
    df["s1"] = _piv["s1"]

    return df


# ─── Individual models ────────────────────────────────────────────────────

def _momentum_regression(df: pd.DataFrame) -> dict:
    """Linear regression on recent returns → extrapolate 1-day."""
    window = 10
    if len(df) < window + 1:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}

    recent = df["close"].iloc[-window:].values
    x = np.arange(window, dtype=float)
    # Normalize to avoid numerical issues
    x_norm = x - x.mean()
    y = (recent / recent[0] - 1) * 100  # % change from start

    slope = np.sum(x_norm * (y - y.mean())) / (np.sum(x_norm ** 2) + 1e-10)
    r_squared = 1.0 - np.sum((y - (slope * x_norm + y.mean())) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-10)
    r_squared = max(0, min(1, r_squared))

    predicted_move = slope  # next-day continuation
    direction = np.clip(slope / (abs(slope) + 0.1), -1, 1)

    return {
        "direction": float(direction),
        "magnitude": abs(float(predicted_move)),
        "confidence": float(r_squared) * 100,
    }


def _mean_reversion(df: pd.DataFrame) -> dict:
    """RSI + Bollinger %B extremes suggest pullback."""
    if len(df) < 20:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}

    latest = df.iloc[-1]
    rsi_val = float(latest["rsi"])
    bb_pct = float(latest["bb_pct"])

    # Oversold → bullish reversion; overbought → bearish reversion
    rsi_signal = 0.0
    if rsi_val < 30:
        rsi_signal = (30 - rsi_val) / 30   # 0 to 1
    elif rsi_val > 70:
        rsi_signal = -(rsi_val - 70) / 30  # 0 to -1

    bb_signal = 0.0
    if bb_pct < 0.1:
        bb_signal = (0.1 - bb_pct) / 0.1
    elif bb_pct > 0.9:
        bb_signal = -(bb_pct - 0.9) / 0.1

    combined = np.clip((rsi_signal + bb_signal) / 2, -1, 1)
    conf = min(abs(rsi_signal) + abs(bb_signal), 1.0) * 100

    return {
        "direction": float(combined),
        "magnitude": abs(float(combined)) * 1.5,  # mean-reversion moves ~1.5%
        "confidence": float(conf),
    }


def _trend_following(df: pd.DataFrame) -> dict:
    """MA stack alignment + MACD momentum."""
    if len(df) < 200:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}

    latest = df.iloc[-1]

    # Count how many EMA pairs are properly ordered (bullish)
    ema_keys = ["ema10", "ema20", "ema50", "ema100", "ema150", "ema200"]
    bullish_pairs = 0
    total_pairs = 0
    for i in range(len(ema_keys) - 1):
        total_pairs += 1
        if latest[ema_keys[i]] > latest[ema_keys[i + 1]]:
            bullish_pairs += 1

    stack_score = bullish_pairs / total_pairs  # 0 to 1

    # MACD direction
    macd_dir = 1.0 if latest["macd_hist"] > 0 else -1.0
    macd_strength = min(abs(float(latest["macd_hist"])) / (float(latest["atr"]) + 1e-10), 1.0)

    # ADX for trend strength
    adx_val = float(latest["adx"])
    trend_strength = min(adx_val / 50, 1.0)  # normalize 0–50 → 0–1

    # If stack mostly bullish → bullish direction; else bearish
    if stack_score >= 0.6:
        direction = stack_score * macd_dir
    elif stack_score <= 0.4:
        direction = -(1 - stack_score) * macd_dir
    else:
        direction = 0.0

    direction = np.clip(direction, -1, 1)
    conf = (stack_score * 0.4 + macd_strength * 0.3 + trend_strength * 0.3) * 100

    return {
        "direction": float(direction),
        "magnitude": abs(float(direction)) * 0.8,
        "confidence": float(conf),
    }


def _volume_pressure(df: pd.DataFrame) -> dict:
    """OBV trend + unusual volume detection."""
    if len(df) < 20:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}

    latest = df.iloc[-1]

    # OBV above/below its EMA → accumulation/distribution
    obv_diff = float(latest["obv"] - latest["obv_ema"])
    obv_norm = obv_diff / (abs(float(latest["obv_ema"])) + 1e-10)
    obv_signal = np.clip(obv_norm * 5, -1, 1)

    # Volume spike on up/down day → confirms direction
    vol_ratio = float(latest["vol_ratio"])
    price_chg = float(latest["ret_1d"]) if not pd.isna(latest["ret_1d"]) else 0.0
    vol_confirm = 0.0
    if vol_ratio > 1.5:
        vol_confirm = np.sign(price_chg) * min((vol_ratio - 1) / 2, 1.0)

    combined = np.clip((obv_signal * 0.6 + vol_confirm * 0.4), -1, 1)
    conf = min(abs(obv_signal) + abs(vol_confirm), 1.0) * 0.7 * 100

    return {
        "direction": float(combined),
        "magnitude": abs(float(combined)) * 0.6,
        "confidence": float(conf),
    }


def _volatility_regime(df: pd.DataFrame) -> dict:
    """ATR contraction → breakout likelihood (direction from other signals)."""
    if len(df) < 50:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}

    atr_pcts = df["atr_pct"].iloc[-50:]
    current_atr = float(atr_pcts.iloc[-1])
    median_atr = float(atr_pcts.median())
    bb_width = float(df["bb_width"].iloc[-1])
    bb_width_med = float(df["bb_width"].iloc[-50:].median())

    # Squeeze detection: both ATR and BB width below median
    atr_ratio = current_atr / (median_atr + 1e-10)
    bb_ratio = bb_width / (bb_width_med + 1e-10)

    squeeze = 0.0
    if atr_ratio < 0.8 and bb_ratio < 0.8:
        squeeze = 1.0 - (atr_ratio + bb_ratio) / 2  # higher when tighter

    # Direction hint from recent price position relative to BB mid
    close = float(df["close"].iloc[-1])
    bb_mid = float(df["bb_mid"].iloc[-1])
    dir_hint = 1.0 if close > bb_mid else -1.0

    return {
        "direction": float(dir_hint * squeeze * 0.5),
        "magnitude": float(squeeze) * 2.0,  # breakouts can be large
        "confidence": float(squeeze) * 60,
    }


# ─── Ensemble ─────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame) -> dict:
    """Run all models and produce an ensemble prediction.

    Returns:
        dict with keys:
            symbol_date  — last date in the dataframe
            direction    — float in [-1, +1] (negative = bearish, positive = bullish)
            confidence   — 0–100 (%)
            predicted_move_pct — expected move magnitude (%)
            models       — dict of individual model outputs
            close        — latest close price
            rsi          — latest RSI
            macd_hist    — latest MACD histogram
            adx          — latest ADX
            bb_pct       — latest Bollinger %B
            atr_pct      — latest ATR%
    """
    df = _prepare_features(df)

    models = {
        "momentum": _momentum_regression(df),
        "mean_reversion": _mean_reversion(df),
        "trend": _trend_following(df),
        "volume": _volume_pressure(df),
        "volatility": _volatility_regime(df),
    }

    # Weighted ensemble: weight by each model's confidence
    weights = {
        "momentum": 0.25,
        "mean_reversion": 0.15,
        "trend": 0.30,
        "volume": 0.15,
        "volatility": 0.15,
    }

    total_w = 0
    w_direction = 0.0
    w_magnitude = 0.0
    w_confidence = 0.0

    for name, m in models.items():
        w = weights[name] * (m["confidence"] / 100 + 0.1)  # floor so zero-conf still contributes slightly
        w_direction += m["direction"] * w
        w_magnitude += m["magnitude"] * w
        w_confidence += m["confidence"] * w
        total_w += w

    if total_w > 0:
        direction = np.clip(w_direction / total_w, -1, 1)
        magnitude = w_magnitude / total_w
        confidence = w_confidence / total_w
    else:
        direction = 0.0
        magnitude = 0.0
        confidence = 0.0

    latest = df.iloc[-1]

    return {
        "symbol_date": latest["date"],
        "close": float(latest["close"]),
        "direction": float(direction),
        "confidence": round(float(confidence), 1),
        "predicted_move_pct": round(float(magnitude), 3),
        "rsi": round(float(latest["rsi"]), 1),
        "macd_hist": round(float(latest["macd_hist"]), 4),
        "adx": round(float(latest["adx"]), 1),
        "bb_pct": round(float(latest["bb_pct"]), 3),
        "atr_pct": round(float(latest["atr_pct"]) * 100, 2),
        "models": models,
    }

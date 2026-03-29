"""Multi-signal prediction engine v2 — optimized for daily price-based suggestions.

Models (7 total):
  1. Momentum regression  — linear regression on recent returns
  2. Mean-reversion       — RSI + Bollinger %B extremes
  3. Trend-following      — MA stack alignment + MACD direction
  4. Volume-price         — OBV trend + unusual volume detection
  5. Volatility regime    — ATR contraction → breakout anticipation
  6. Price-action         — candlestick patterns + swing structure
  7. Support/Resistance   — proximity to key levels → bounce/break

Additional exports:
  - find_support_resistance(df) — key price levels from swing points
  - find_swing_points(df)       — recent swing highs and lows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.features.indicators import (
    ema, sma, rsi, macd, atr, bollinger_bands, adx, pivot_points,
)


# ─── Feature engineering ──────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for n in [10, 20, 50, 100, 150, 200]:
        df[f"ema{n}"] = ema(df, n)

    df["sma20"] = sma(df, 20)
    df["sma50"] = sma(df, 50)
    df["rsi"] = rsi(df, 14)

    _m = macd(df, 12, 26, 9)
    df["macd"] = _m["macd"]
    df["macd_signal"] = _m["macd_signal"]
    df["macd_hist"] = _m["macd_hist"]
    df["adx"] = adx(df, 14)

    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / df["close"]
    _bb = bollinger_bands(df, 20, 2.0)
    df["bb_pct"] = _bb["bb_pct"]
    df["bb_upper"] = _bb["bb_upper"]
    df["bb_lower"] = _bb["bb_lower"]
    df["bb_mid"] = _bb["bb_mid"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()

    for n in [1, 3, 5, 10]:
        df[f"ret_{n}d"] = df["close"].pct_change(n)

    _piv = pivot_points(df)
    df["pivot"] = _piv["pivot"]
    df["r1"] = _piv["r1"]
    df["r2"] = _piv["r2"]
    df["s1"] = _piv["s1"]
    df["s2"] = _piv["s2"]

    body = df["close"] - df["open"]
    full_range = df["high"] - df["low"]
    df["body_pct"] = body / (full_range + 1e-10)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (full_range + 1e-10)
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (full_range + 1e-10)

    return df


# ─── Swing point detection ────────────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> dict:
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []

    for i in range(lookback, n - lookback):
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((i, float(highs[i])))
        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((i, float(lows[i])))

    for i in range(max(lookback, n - lookback), n):
        if i >= lookback:
            if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)):
                swing_highs.append((i, float(highs[i])))
            if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)):
                swing_lows.append((i, float(lows[i])))

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def find_support_resistance(df: pd.DataFrame, n_levels: int = 5) -> dict:
    """Key S/R from swing points, round numbers, EMAs, pivots, volume clusters."""
    close = float(df["close"].iloc[-1])
    atr_val = float(df["high"].iloc[-20:].max() - df["low"].iloc[-20:].min()) / 20
    if atr_val == 0:
        atr_val = close * 0.02
    tolerance = atr_val * 0.3

    raw = []

    # 1) Swing points — 3-bar and 5-bar
    recent = df.iloc[-min(100, len(df)):]
    sw3 = find_swing_points(recent, lookback=3)
    for _, p in sw3["swing_highs"][-10:]:
        raw.append((p, 3.0, "swing_high"))
    for _, p in sw3["swing_lows"][-10:]:
        raw.append((p, 3.0, "swing_low"))
    if len(df) > 50:
        sw5 = find_swing_points(df.iloc[-min(200, len(df)):], lookback=5)
        for _, p in sw5["swing_highs"][-8:]:
            raw.append((p, 4.0, "major_swing_high"))
        for _, p in sw5["swing_lows"][-8:]:
            raw.append((p, 4.0, "major_swing_low"))

    # 2) Round numbers
    magnitude = 10 ** max(0, int(np.log10(max(close, 1))) - 1)
    base = (close // magnitude - 3) * magnitude
    for i in range(7):
        level = base + i * magnitude
        if abs(level - close) < close * 0.1:
            raw.append((level, 1.5, "round_number"))

    # 3) EMAs
    latest = df.iloc[-1]
    for name in ["ema10", "ema20", "ema50", "ema100", "ema150", "ema200"]:
        if name in latest.index and not pd.isna(latest[name]):
            raw.append((float(latest[name]), 2.0, f"ema_{name}"))

    # 4) Pivots
    for name, s in [("pivot", 2.5), ("r1", 2.0), ("r2", 1.5), ("s1", 2.0), ("s2", 1.5)]:
        if name in latest.index and not pd.isna(latest[name]):
            raw.append((float(latest[name]), s, f"pivot_{name}"))

    # 5) Volume profile (top-3 bins by volume in last 20 bars)
    if len(df) >= 20:
        r20 = df.iloc[-20:]
        pr = r20["high"].max() - r20["low"].min()
        if pr > 0:
            nb = 10
            bs = pr / nb
            bb = r20["low"].min()
            vb = np.zeros(nb)
            for _, row in r20.iterrows():
                mid = (row["high"] + row["low"]) / 2
                b = min(int((mid - bb) / bs), nb - 1)
                vb[b] += row["volume"]
            for b in np.argsort(vb)[-3:]:
                raw.append((float(bb + (b + 0.5) * bs), 2.0, "volume_cluster"))

    # 6) Recent extremes
    if len(df) >= 5:
        raw.append((float(df["high"].iloc[-5:].max()), 2.5, "5d_high"))
        raw.append((float(df["low"].iloc[-5:].min()), 2.5, "5d_low"))
    if len(df) >= 20:
        raw.append((float(df["high"].iloc[-20:].max()), 3.0, "20d_high"))
        raw.append((float(df["low"].iloc[-20:].min()), 3.0, "20d_low"))

    # Merge nearby levels
    raw.sort(key=lambda x: x[0])
    merged = []
    for price, strength, source in raw:
        found = False
        for j, (mp, ms, msrc) in enumerate(merged):
            if abs(price - mp) < tolerance:
                new_price = (mp * ms + price * strength) / (ms + strength)
                merged[j] = (new_price, ms + strength, msrc + "+" + source)
                found = True
                break
        if not found:
            merged.append((price, strength, source))

    support = sorted(
        [(p, s, src) for p, s, src in merged if p < close],
        key=lambda x: -x[1],
    )[:n_levels]
    resistance = sorted(
        [(p, s, src) for p, s, src in merged if p >= close],
        key=lambda x: -x[1],
    )[:n_levels]

    return {
        "support": sorted(support, key=lambda x: -x[0]),
        "resistance": sorted(resistance, key=lambda x: x[0]),
    }


# ─── Individual models ────────────────────────────────────────────────────

def _momentum_regression(df):
    window = 10
    if len(df) < window + 1:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    recent = df["close"].iloc[-window:].values
    x = np.arange(window, dtype=float)
    x_n = x - x.mean()
    y = (recent / recent[0] - 1) * 100
    slope = np.sum(x_n * (y - y.mean())) / (np.sum(x_n**2) + 1e-10)
    ss_res = np.sum((y - (slope * x_n + y.mean()))**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-10
    r2 = max(0, min(1, 1 - ss_res / ss_tot))
    return {
        "direction": float(np.clip(slope / (abs(slope) + 0.1), -1, 1)),
        "magnitude": abs(float(slope)),
        "confidence": float(r2) * 100,
    }


def _mean_reversion(df):
    if len(df) < 20:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    latest = df.iloc[-1]
    rsi_val, bb_pct = float(latest["rsi"]), float(latest["bb_pct"])
    rsi_sig = (30 - rsi_val) / 30 if rsi_val < 30 else (-(rsi_val - 70) / 30 if rsi_val > 70 else 0.0)
    bb_sig = (0.1 - bb_pct) / 0.1 if bb_pct < 0.1 else (-(bb_pct - 0.9) / 0.1 if bb_pct > 0.9 else 0.0)
    combined = np.clip((rsi_sig + bb_sig) / 2, -1, 1)
    conf = min(abs(rsi_sig) + abs(bb_sig), 1.0) * 100
    return {"direction": float(combined), "magnitude": abs(float(combined)) * 1.5, "confidence": float(conf)}


def _trend_following(df):
    if len(df) < 200:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    latest = df.iloc[-1]
    emas = ["ema10", "ema20", "ema50", "ema100", "ema150", "ema200"]
    bp = sum(1 for i in range(len(emas)-1) if latest[emas[i]] > latest[emas[i+1]])
    ss = bp / (len(emas) - 1)
    md = 1.0 if latest["macd_hist"] > 0 else -1.0
    ms = min(abs(float(latest["macd_hist"])) / (float(latest["atr"]) + 1e-10), 1.0)
    ts = min(float(latest["adx"]) / 50, 1.0)
    d = ss * md if ss >= 0.6 else (-(1 - ss) * md if ss <= 0.4 else 0.0)
    d = np.clip(d, -1, 1)
    return {"direction": float(d), "magnitude": abs(float(d)) * 0.8, "confidence": (ss*0.4+ms*0.3+ts*0.3)*100}


def _volume_pressure(df):
    if len(df) < 20:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    latest = df.iloc[-1]
    od = float(latest["obv"] - latest["obv_ema"])
    on = od / (abs(float(latest["obv_ema"])) + 1e-10)
    osig = np.clip(on * 5, -1, 1)
    vr = float(latest["vol_ratio"])
    pc = float(latest["ret_1d"]) if not pd.isna(latest["ret_1d"]) else 0.0
    vc = np.sign(pc) * min((vr - 1) / 2, 1.0) if vr > 1.5 else 0.0
    c = np.clip(osig * 0.6 + vc * 0.4, -1, 1)
    return {"direction": float(c), "magnitude": abs(float(c)) * 0.6, "confidence": min(abs(osig)+abs(vc), 1.0)*70}


def _volatility_regime(df):
    if len(df) < 50:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    ca = float(df["atr_pct"].iloc[-1])
    ma = float(df["atr_pct"].iloc[-50:].median())
    bw = float(df["bb_width"].iloc[-1])
    bwm = float(df["bb_width"].iloc[-50:].median())
    ar, br = ca/(ma+1e-10), bw/(bwm+1e-10)
    sq = max(0, 1-(ar+br)/2) if (ar < 0.8 and br < 0.8) else 0.0
    dh = 1.0 if float(df["close"].iloc[-1]) > float(df["bb_mid"].iloc[-1]) else -1.0
    return {"direction": float(dh*sq*0.5), "magnitude": float(sq)*2.0, "confidence": float(sq)*60}


def _price_action(df):
    if len(df) < 10:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    latest, prev = df.iloc[-1], df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else prev
    sigs, confs = [], []
    body = float(latest["body_pct"])
    uw, lw = float(latest["upper_wick"]), float(latest["lower_wick"])
    pb = float(prev["body_pct"])
    vr = float(latest["vol_ratio"]) if not pd.isna(latest["vol_ratio"]) else 1.0

    # Bullish patterns
    if lw > 0.6 and abs(body) < 0.3 and uw < 0.15:
        sigs.append(0.7); confs.append(65)
    if body > 0.5 and pb < -0.3 and latest["close"] > prev["open"] and latest["open"] < prev["close"]:
        sigs.append(0.8); confs.append(70)
    if float(prev2["body_pct"]) < -0.4 and abs(pb) < 0.25 and body > 0.4:
        sigs.append(0.7); confs.append(60)
    if body > 0.6 and vr > 1.3:
        sigs.append(0.5); confs.append(55)

    # Bearish patterns
    if uw > 0.6 and abs(body) < 0.3 and lw < 0.15:
        sigs.append(-0.7); confs.append(65)
    if body < -0.5 and pb > 0.3 and latest["close"] < prev["open"] and latest["open"] > prev["close"]:
        sigs.append(-0.8); confs.append(70)
    if float(prev2["body_pct"]) > 0.4 and abs(pb) < 0.25 and body < -0.4:
        sigs.append(-0.7); confs.append(60)
    if body < -0.6 and vr > 1.3:
        sigs.append(-0.5); confs.append(55)

    # Swing structure (HH/HL vs LH/LL)
    if len(df) >= 10:
        h5 = df["high"].iloc[-5:].values
        l5 = df["low"].iloc[-5:].values
        hp = df["high"].iloc[-10:-5].values
        lp = df["low"].iloc[-10:-5].values
        if len(hp) >= 5:
            if h5.max() > hp.max() and l5.min() > lp.min():
                sigs.append(0.4); confs.append(45)
            if h5.max() < hp.max() and l5.min() < lp.min():
                sigs.append(-0.4); confs.append(45)

    if not sigs:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    tw = sum(confs)
    d = sum(s*c for s, c in zip(sigs, confs)) / tw
    return {"direction": float(np.clip(d, -1, 1)), "magnitude": abs(d)*1.0, "confidence": float(max(confs))}


def _sr_proximity(df):
    if len(df) < 30:
        return {"direction": 0.0, "magnitude": 0.0, "confidence": 0.0}
    close = float(df["close"].iloc[-1])
    sr = find_support_resistance(df, n_levels=3)
    sig, conf = 0.0, 0.0
    if sr["support"]:
        ns = sr["support"][0]
        d = (close - ns[0]) / close
        if d < 0.015:
            sig += 0.5 * ns[1] / 5
            conf = max(conf, 50 + ns[1] * 5)
    if sr["resistance"]:
        nr = sr["resistance"][0]
        d = (nr[0] - close) / close
        if d < 0.015:
            sig -= 0.5 * nr[1] / 5
            conf = max(conf, 50 + nr[1] * 5)
    return {"direction": float(np.clip(sig, -1, 1)), "magnitude": abs(sig)*1.2, "confidence": float(min(conf, 100))}


# ─── Ensemble ─────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame) -> dict:
    df = _prepare_features(df)

    models = {
        "momentum": _momentum_regression(df),
        "mean_reversion": _mean_reversion(df),
        "trend": _trend_following(df),
        "volume": _volume_pressure(df),
        "volatility": _volatility_regime(df),
        "price_action": _price_action(df),
        "sr_proximity": _sr_proximity(df),
    }

    weights = {
        "momentum": 0.15, "mean_reversion": 0.12, "trend": 0.25,
        "volume": 0.12, "volatility": 0.10, "price_action": 0.16, "sr_proximity": 0.10,
    }

    tw = 0
    wd, wm, wc = 0.0, 0.0, 0.0
    for name, m in models.items():
        w = weights[name] * (m["confidence"] / 100 + 0.1)
        wd += m["direction"] * w
        wm += m["magnitude"] * w
        wc += m["confidence"] * w
        tw += w

    if tw > 0:
        direction = np.clip(wd / tw, -1, 1)
        magnitude = wm / tw
        confidence = wc / tw
    else:
        direction, magnitude, confidence = 0.0, 0.0, 0.0

    latest = df.iloc[-1]
    sr = find_support_resistance(df)
    swings = find_swing_points(df.iloc[-min(60, len(df)):], lookback=3)

    return {
        "symbol_date": latest["date"],
        "close": float(latest["close"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "open": float(latest["open"]),
        "direction": float(direction),
        "confidence": round(float(confidence), 1),
        "predicted_move_pct": round(float(magnitude), 3),
        "rsi": round(float(latest["rsi"]), 1),
        "macd_hist": round(float(latest["macd_hist"]), 4),
        "adx": round(float(latest["adx"]), 1),
        "bb_pct": round(float(latest["bb_pct"]), 3),
        "bb_upper": float(latest["bb_upper"]),
        "bb_lower": float(latest["bb_lower"]),
        "bb_mid": float(latest["bb_mid"]),
        "atr_pct": round(float(latest["atr_pct"]) * 100, 2),
        "atr": float(latest["atr"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "pivot": float(latest["pivot"]),
        "r1": float(latest["r1"]),
        "r2": float(latest["r2"]),
        "s1": float(latest["s1"]),
        "s2": float(latest["s2"]),
        "support_resistance": sr,
        "swing_points": swings,
        "models": models,
    }

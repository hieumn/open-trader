"""Confluence Momentum Strategy — ultra-high win-rate daily trading.

Supports two MA-stack modes:

6-MA mode (default — US large-caps, strong trending stocks):
  Stack:   EMA10 > EMA20 > EMA50 > EMA100 > EMA150 > EMA200
  Pullback target: EMA10
  Score threshold: 9/10

3-MA mode (--ma-stack 3 — emerging/frontier markets, e.g. Vietnam):
  Stack:   EMA20 > EMA50 > EMA200
  Pullback target: EMA20 (wider, suits choppier markets)
  Score threshold: 7/9
  RSI range:  45-68  (slightly wider)
  ATR/spike:  < 3%  (relaxed for higher-volatility markets)

EXIT (managed in runner.py):
  - Take-profit at +0.5% from entry (default)
  - Time stop: force-exit after 7 bars if TP not hit
  - max-loss cap effectively disabled (let time stop handle it)
"""

import pandas as pd
from src.features.indicators import ema, rsi, macd, atr, adx, bollinger_bands, pivot_points


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 6-tier EMA stack
    for n in [10, 20, 50, 100, 150, 200]:
        df[f"ema{n}"] = ema(df, n)

    df["adx"]  = adx(df, 14)
    df["rsi"]  = rsi(df, 14)

    _macd = macd(df, fast=12, slow=26, signal=9)
    df["macd"]        = _macd["macd"]
    df["macd_signal"] = _macd["macd_signal"]
    df["macd_hist"]   = _macd["macd_hist"]

    df["atr"] = atr(df, 14)

    _bb = bollinger_bands(df, 20, 2.0)
    df["bb_mid"]   = _bb["bb_mid"]
    df["bb_upper"] = _bb["bb_upper"]
    df["bb_lower"] = _bb["bb_lower"]
    df["bb_pct"]   = _bb["bb_pct"]

    _piv = pivot_points(df)
    df["pivot"] = _piv["pivot"]
    df["r1"]    = _piv["r1"]
    df["r2"]    = _piv["r2"]
    df["s1"]    = _piv["s1"]
    df["s2"]    = _piv["s2"]

    # Volume ratio
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Volatility: ATR as % of close
    df["atr_pct"] = df["atr"] / df["close"]

    # Daily % change (absolute)
    df["daily_chg"] = (df["close"] / df["close"].shift(1) - 1).abs()

    # Max absolute daily change in last 5 bars
    df["max_chg_5"] = df["daily_chg"].rolling(5).max()

    return df


def generate_signals(df: pd.DataFrame, ma_stack: int = 6) -> pd.DataFrame:
    """Generate entry/exit signals.

    Parameters
    ----------
    ma_stack : int
        6  — full 6-MA perfect stack (best for US large-caps with strong trends)
        3  — lighter 3-MA stack (better for emerging/frontier markets like Vietnam)
    """
    if ma_stack not in (3, 6):
        raise ValueError("ma_stack must be 3 or 6")

    df = compute_indicators(df)

    if ma_stack == 6:
        # ── 6-MA mode: EMA10 > EMA20 > EMA50 > EMA100 > EMA150 > EMA200 ──────
        stack_cond = (
            (df["ema10"] > df["ema20"])
            & (df["ema20"] > df["ema50"])
            & (df["ema50"] > df["ema100"])
            & (df["ema100"] > df["ema150"])
            & (df["ema150"] > df["ema200"])
        )
        emas_rising = (
            (df["ema10"] > df["ema10"].shift(3))
            & (df["ema20"] > df["ema20"].shift(3))
            & (df["ema50"] > df["ema50"].shift(5))
            & (df["ema100"] > df["ema100"].shift(5))
        )
        pullback_ref = df["ema10"]
        rsi_hi  = 65
        vol_cap = 0.02
        spk_cap = 0.02
        min_score = 9
    else:
        # ── 3-MA mode: EMA20 > EMA50 > EMA200 (relaxed for choppy markets) ──
        stack_cond = (
            (df["ema20"] > df["ema50"])
            & (df["ema50"] > df["ema200"])
        )
        emas_rising = (
            (df["ema20"] > df["ema20"].shift(3))
            & (df["ema50"] > df["ema50"].shift(5))
            & (df["ema200"] > df["ema200"].shift(10))
        )
        pullback_ref = df["ema20"]   # wider pullback target
        rsi_hi  = 68
        vol_cap = 0.03   # relax vol filter for higher-vol markets
        spk_cap = 0.03
        min_score = 7

    # ── Pullback: low touched pullback_ref within last 2 bars ─────────────
    touched = (
        (df["low"] <= pullback_ref)
        | (df["low"].shift(1) <= pullback_ref.shift(1))
    )

    # ── Bounce candle: green, close > pullback_ref ────────────────────────
    bounce = (df["close"] > df["open"]) & (df["close"] > pullback_ref)

    # ── RSI in sweet spot ──────────────────────────────────────────────────
    rsi_ok = (df["rsi"] >= 45) & (df["rsi"] <= rsi_hi)

    # ── MACD histogram positive ────────────────────────────────────────────
    macd_pos = df["macd_hist"] > 0

    # ── Bollinger %B in the middle zone ────────────────────────────────────
    bb_ok = (df["bb_pct"] >= 0.3) & (df["bb_pct"] <= 0.80)

    # ── Above yesterday's pivot ────────────────────────────────────────────
    above_pivot = df["close"] > df["pivot"]

    # ── Low volatility ─────────────────────────────────────────────────────
    low_vol  = df["atr_pct"] < vol_cap

    # ── No recent spike ─────────────────────────────────────────────────────
    no_spike = df["max_chg_5"] < spk_cap

    # ── Score ───────────────────────────────────────────────────────────────
    score = (
        stack_cond.astype(int)
        + emas_rising.astype(int)
        + touched.astype(int)
        + bounce.astype(int)
        + rsi_ok.astype(int)
        + macd_pos.astype(int)
        + bb_ok.astype(int)
        + above_pivot.astype(int)
        + low_vol.astype(int)
        + no_spike.astype(int)
    )
    df["entry_score"] = score

    # ENTRY: stack + calm market conditions + score threshold
    df["signal_entry"] = (
        stack_cond & low_vol & no_spike & (score >= min_score)
    ).astype(int)

    # EXIT: only RSI extreme (most exits handled by runner time-stop & TP)
    df["signal_exit"] = (df["rsi"] > 78).astype(int)

    cols = [
        "date", "open", "high", "low", "close", "volume",
        "ema10", "ema20", "ema50", "ema100", "ema150", "ema200",
        "adx", "rsi", "macd", "macd_signal", "macd_hist",
        "atr", "atr_pct", "bb_mid", "bb_upper", "bb_lower", "bb_pct",
        "pivot", "r1", "r2", "s1", "s2", "vol_ratio",
        "daily_chg", "max_chg_5",
        "entry_score", "signal_entry", "signal_exit",
    ]
    return df[cols]




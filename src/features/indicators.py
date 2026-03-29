import pandas as pd


# ── Moving Averages ────────────────────────────────────────────────────────────

def sma(df: pd.DataFrame, length: int, price_col: str = "close") -> pd.Series:
    return df[price_col].rolling(length).mean()


def ema(df: pd.DataFrame, length: int, price_col: str = "close") -> pd.Series:
    return df[price_col].ewm(span=length, adjust=False).mean()


# ── Momentum ───────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, length: int = 14, price_col: str = "close") -> pd.Series:
    """Wilder RSI (EWM-based, same as TradingView default)."""
    delta = df[price_col].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / down.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """Returns DataFrame with columns: macd, macd_signal, macd_hist."""
    fast_ema = df[price_col].ewm(span=fast, adjust=False).mean()
    slow_ema = df[price_col].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line},
        index=df.index,
    )


# ── Volatility ─────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (EWM Wilder smoothing)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def bollinger_bands(
    df: pd.DataFrame, length: int = 20, std_dev: float = 2.0, price_col: str = "close"
) -> pd.DataFrame:
    """Returns DataFrame with columns: bb_mid, bb_upper, bb_lower, bb_width, bb_pct."""
    mid = df[price_col].rolling(length).mean()
    std = df[price_col].rolling(length).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid
    pct = (df[price_col] - lower) / (upper - lower)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct": pct},
        index=df.index,
    )


# ── Support / Resistance ───────────────────────────────────────────────────────

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (>25 = strong trend)."""
    high = df["high"]
    low  = df["low"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = df["close"].shift(1)

    # True Range
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    # Directional movement
    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    # When +DM > -DM, minus_dm = 0 and vice-versa
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    atr_s     = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean()  / atr_s
    minus_di  = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_s
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")))
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic floor pivot points calculated from the *previous* day's H/L/C.

    Returns DataFrame with columns: pivot, r1, s1, r2, s2, r3, s3.
    All values are forward-filled so each row carries *yesterday's* pivots.
    """
    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low  - 2 * (prev_high - pivot)

    return pd.DataFrame(
        {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3},
        index=df.index,
    )
import pandas as pd
from src.features.indicators import sma, rsi, atr


def compute_indicators(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = sma(df, fast)
    df["sma_slow"] = sma(df, slow)
    df["rsi"] = rsi(df, rsi_period)
    df["atr"] = atr(df, atr_period)
    return df


def generate_signals(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 200,
    rsi_period: int = 14,
    atr_period: int = 14,
    rsi_entry_max: float = 70.0,
    rsi_entry_min: float = 40.0,
) -> pd.DataFrame:
    """SMA crossover with RSI confirmation.

    Enter long only when:
      - fast SMA crosses above slow SMA  (trend filter)
      - RSI is between rsi_entry_min and rsi_entry_max  (avoid overbought entries)
    Exit when fast SMA crosses below slow SMA regardless of RSI.
    """
    df = compute_indicators(df, fast, slow, rsi_period, atr_period)

    raw_signal = pd.Series(0, index=df.index)
    raw_signal[df["sma_fast"] > df["sma_slow"]] = 1
    raw_signal[df["sma_fast"] < df["sma_slow"]] = -1

    crossover = raw_signal.diff().fillna(0)

    # Apply RSI filter: only allow a BUY crossover when RSI is not overbought
    filtered_crossover = crossover.copy()
    buy_cross = crossover > 0
    rsi_overbought = df["rsi"] > rsi_entry_max
    rsi_oversold = df["rsi"] < rsi_entry_min
    filtered_crossover[buy_cross & (rsi_overbought | rsi_oversold)] = 0

    df["signal"] = raw_signal
    df["signal_change"] = filtered_crossover
    return df[["date", "close", "sma_fast", "sma_slow", "rsi", "atr", "signal", "signal_change"]]

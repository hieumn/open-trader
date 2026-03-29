import pandas as pd
import yfinance as yf


def fetch_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a normalised DataFrame.

    Returns a DataFrame with columns: date, open, high, low, close, volume.
    """
    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start, end=end, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for symbol '{symbol}' between {start} and {end}.")
    df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.reset_index(drop=True)

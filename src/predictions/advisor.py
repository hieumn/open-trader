"""Trading advisor — converts predictions into actionable BUY / SELL / HOLD.

Uses the ensemble prediction plus support/resistance levels to generate:
  - Action:       BUY / SELL / HOLD
  - Entry price:  suggested limit-order price
  - Take-profit:  target exit price
  - Stop-loss:    risk management level
  - Risk/reward:  ratio
  - Reasoning:    plain-English explanation
"""

from __future__ import annotations

import pandas as pd
from src.predictions.predictor import predict, _prepare_features
from src.features.indicators import pivot_points


def advise(df: pd.DataFrame) -> dict:
    """Produce a full trading recommendation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe (date, open, high, low, close, volume).

    Returns
    -------
    dict with keys:
        action, entry_price, take_profit, stop_loss,
        risk_reward, confidence, reasons[], prediction{}
    """
    pred = predict(df)

    direction = pred["direction"]
    confidence = pred["confidence"]
    close = pred["close"]
    rsi_val = pred["rsi"]
    adx_val = pred["adx"]
    bb_pct = pred["bb_pct"]
    atr_pct = pred["atr_pct"]
    predicted_move = pred["predicted_move_pct"]

    # Get support/resistance from pivots
    feat_df = _prepare_features(df)
    latest = feat_df.iloc[-1]
    pivot = float(latest["pivot"])
    r1 = float(latest["r1"])
    s1 = float(latest["s1"])

    reasons = []

    # ── Decision logic ────────────────────────────────────────────────────

    # Strong BUY: confident bullish prediction + supporting indicators
    if direction > 0.25 and confidence > 35:
        action = "BUY"
        # Entry slightly below close (limit order on pullback)
        entry_price = round(close * 0.998, 4)
        # TP based on predicted move + pivot resistance
        tp_by_prediction = close * (1 + predicted_move / 100)
        tp_by_pivot = r1
        take_profit = round(max(tp_by_prediction, close * 1.005), 4)  # at least +0.5%
        if tp_by_pivot > close:
            take_profit = round(min(take_profit, tp_by_pivot), 4)
        # Stop below support or ATR-based
        sl_by_atr = close * (1 - atr_pct / 100 * 1.5)
        sl_by_pivot = s1
        stop_loss = round(max(sl_by_atr, sl_by_pivot, close * 0.97), 4)  # at most -3%

        reasons.append(f"Bullish signal: ensemble direction {direction:+.2f}")
        if rsi_val < 50:
            reasons.append(f"RSI {rsi_val:.0f} — room to run")
        if adx_val > 25:
            reasons.append(f"ADX {adx_val:.0f} — strong trend")
        if bb_pct < 0.5:
            reasons.append(f"BB%B {bb_pct:.2f} — below midline, upside potential")

        # Model agreement
        bull_models = sum(1 for m in pred["models"].values() if m["direction"] > 0.1)
        reasons.append(f"{bull_models}/5 models bullish")

    # Strong SELL: confident bearish prediction
    elif direction < -0.25 and confidence > 35:
        action = "SELL"
        entry_price = round(close, 4)
        take_profit = round(min(close * (1 - predicted_move / 100), s1, close * 0.995), 4)
        stop_loss = round(min(close * (1 + atr_pct / 100 * 1.5), r1, close * 1.03), 4)

        reasons.append(f"Bearish signal: ensemble direction {direction:+.2f}")
        if rsi_val > 60:
            reasons.append(f"RSI {rsi_val:.0f} — overbought territory")
        if bb_pct > 0.7:
            reasons.append(f"BB%B {bb_pct:.2f} — near upper band")

        bear_models = sum(1 for m in pred["models"].values() if m["direction"] < -0.1)
        reasons.append(f"{bear_models}/5 models bearish")

    # HOLD: weak or conflicting signals
    else:
        action = "HOLD"
        entry_price = round(close, 4)
        take_profit = round(r1 if r1 > close else close * 1.01, 4)
        stop_loss = round(s1 if s1 < close else close * 0.99, 4)

        reasons.append(f"Mixed signals: direction {direction:+.2f}, confidence {confidence:.0f}%")
        if abs(direction) < 0.1:
            reasons.append("Models disagree on direction")
        if confidence < 30:
            reasons.append("Low prediction confidence — wait for clearer setup")
        if 45 <= rsi_val <= 55:
            reasons.append(f"RSI {rsi_val:.0f} — neutral zone")

    # Risk/Reward ratio
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    # Strength label
    if confidence >= 60:
        strength = "STRONG"
    elif confidence >= 40:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    return {
        "action": action,
        "strength": strength,
        "entry_price": entry_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "risk_reward": risk_reward,
        "confidence": round(confidence, 1),
        "reasons": reasons,
        "prediction": pred,
    }


def advise_multi(symbols: list[str], start: str, end: str) -> list[dict]:
    """Run advise() for multiple symbols. Returns list of results with symbol key."""
    from src.data.fetcher import fetch_history

    results = []
    for sym in symbols:
        try:
            df = fetch_history(sym, start, end)
            adv = advise(df)
            adv["symbol"] = sym
            results.append(adv)
        except Exception as e:
            results.append({
                "symbol": sym,
                "action": "ERROR",
                "error": str(e),
            })
    return results

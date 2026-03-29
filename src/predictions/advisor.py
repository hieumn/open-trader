"""Trading advisor v2 — ATR-dynamic + S/R-anchored price suggestions.

Long-only model (suitable for Vietnam and most retail markets):
  BUY  → enter long with pullback entry, ATR-based TP/SL
  SELL → exit existing long position, shows re-buy level
  HOLD → wait, shows key levels to watch

Price levels derived from:
  - Nearest support/resistance zones (swing-based, volume-based)
  - ATR-scaled entry pullback, TP, and stop-loss
  - EMA cluster proximity
  - Pivot levels as confirmation
  - Multi-target TP (partial exits at TP1 and TP2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.predictions.predictor import predict, _prepare_features, find_support_resistance


def _nearest_below(levels: list, price: float, default: float) -> float:
    """Return the highest level below price, or default."""
    below = [l[0] for l in levels if l[0] < price]
    return below[0] if below else default  # levels already sorted desc


def _nearest_above(levels: list, price: float, default: float) -> float:
    """Return the lowest level above price, or default."""
    above = [l[0] for l in levels if l[0] > price]
    return above[0] if above else default  # levels already sorted asc


def advise(df: pd.DataFrame) -> dict:
    """Produce a full trading recommendation with optimized price levels.

    Returns dict with:
      action, strength, entry_price, take_profit_1, take_profit_2,
      stop_loss, risk_reward, confidence, reasons[], key_levels{},
      prediction{}
    """
    pred = predict(df)

    direction = pred["direction"]
    confidence = pred["confidence"]
    close = pred["close"]
    high = pred["high"]
    low = pred["low"]
    atr_val = pred["atr"]
    atr_pct = pred["atr_pct"]
    rsi_val = pred["rsi"]
    adx_val = pred["adx"]
    bb_pct = pred["bb_pct"]
    bb_upper = pred["bb_upper"]
    bb_lower = pred["bb_lower"]
    ema20 = pred["ema20"]
    ema50 = pred["ema50"]
    predicted_move = pred["predicted_move_pct"]
    sr = pred["support_resistance"]

    reasons = []

    # ── Key S/R levels ────────────────────────────────────────────────────
    sup_default = close * 0.97
    res_default = close * 1.03
    nearest_support = _nearest_below(sr["support"], close, sup_default)
    nearest_resistance = _nearest_above(sr["resistance"], close, res_default)

    # Second levels for multi-target
    sup_levels = [l[0] for l in sr["support"] if l[0] < nearest_support]
    res_levels = [l[0] for l in sr["resistance"] if l[0] > nearest_resistance]
    support_2 = sup_levels[0] if sup_levels else nearest_support - atr_val
    resistance_2 = res_levels[0] if res_levels else nearest_resistance + atr_val

    # ── BUY logic ─────────────────────────────────────────────────────────
    if direction > 0.20 and confidence > 30:
        action = "BUY"

        # Entry: pullback toward nearest support or 0.5×ATR below close
        entry_by_atr = close - atr_val * 0.5
        entry_by_support = nearest_support + atr_val * 0.1  # just above support
        entry_price = max(entry_by_atr, entry_by_support)
        # Don't set entry above close (that defeats the pullback idea)
        entry_price = min(entry_price, close * 0.999)

        # TP1: nearest resistance or 1.5×ATR above entry
        tp1_by_sr = nearest_resistance
        tp1_by_atr = entry_price + atr_val * 1.5
        take_profit_1 = min(tp1_by_sr, tp1_by_atr)
        take_profit_1 = max(take_profit_1, entry_price * 1.005)  # at least +0.5%

        # TP2: second resistance or 2.5×ATR above entry
        tp2_by_sr = resistance_2
        tp2_by_atr = entry_price + atr_val * 2.5
        take_profit_2 = min(tp2_by_sr, tp2_by_atr)
        take_profit_2 = max(take_profit_2, take_profit_1 * 1.003)

        # Stop: below nearest support by 0.3×ATR, or 1.5×ATR below entry
        sl_by_sr = nearest_support - atr_val * 0.3
        sl_by_atr = entry_price - atr_val * 1.5
        stop_loss = max(sl_by_sr, sl_by_atr)
        # Cap max loss at ~3%
        stop_loss = max(stop_loss, entry_price * 0.97)

        reasons.append(f"Bullish signal: direction {direction:+.2f}, confidence {confidence:.0f}%")
        if rsi_val < 45:
            reasons.append(f"RSI {rsi_val:.0f} — oversold, high bounce potential")
        elif rsi_val < 55:
            reasons.append(f"RSI {rsi_val:.0f} — room to run")
        if adx_val > 25:
            reasons.append(f"ADX {adx_val:.0f} — strong trend supports momentum")
        if bb_pct < 0.3:
            reasons.append(f"BB%B {bb_pct:.2f} — near lower band, upside likely")
        if close > ema20:
            reasons.append("Price above EMA20 — short-term trend is up")

        # S/R context
        dist_to_sup = (close - nearest_support) / close * 100
        dist_to_res = (nearest_resistance - close) / close * 100
        reasons.append(f"Support at {nearest_support:,.0f} ({dist_to_sup:.1f}% below)")
        reasons.append(f"Resistance at {nearest_resistance:,.0f} ({dist_to_res:.1f}% above)")

        bull_models = sum(1 for m in pred["models"].values() if m["direction"] > 0.1)
        reasons.append(f"{bull_models}/7 models bullish")

    # ── SELL logic (long-only: "exit your position / avoid buying") ──────
    elif direction < -0.20 and confidence > 30:
        action = "SELL"

        # Sell at current close
        entry_price = close

        # Re-buy level: nearest strong support (where we'd consider buying back)
        rebuy_level = nearest_support
        rebuy_level_2 = support_2

        # These become "targets" — where price is likely heading
        take_profit_1 = rebuy_level
        take_profit_2 = rebuy_level_2

        # "Invalidation" — if price goes above this, the sell was wrong
        # Use nearest resistance (where bears get invalidated)
        stop_loss = nearest_resistance

        reasons.append(f"Bearish signal: direction {direction:+.2f}, confidence {confidence:.0f}%")
        reasons.append("Recommendation: EXIT long position or AVOID buying")
        if rsi_val > 65:
            reasons.append(f"RSI {rsi_val:.0f} — overbought, pullback likely")
        if bb_pct > 0.7:
            reasons.append(f"BB%B {bb_pct:.2f} — near upper band, pressure building")
        if close < ema20:
            reasons.append("Price below EMA20 — short-term trend is down")
        if close > ema20:
            reasons.append("Price still above EMA20 — consider trailing stop")

        dist_to_sup = (close - nearest_support) / close * 100
        dist_to_res = (nearest_resistance - close) / close * 100
        reasons.append(f"Downside target at {nearest_support:,.0f} ({dist_to_sup:.1f}% below)")
        reasons.append(f"Invalidation above {nearest_resistance:,.0f} ({dist_to_res:.1f}% above)")
        reasons.append(f"Re-buy zone: {rebuy_level:,.0f} – {rebuy_level_2:,.0f}")

        bear_models = sum(1 for m in pred["models"].values() if m["direction"] < -0.1)
        reasons.append(f"{bear_models}/7 models bearish")

    # ── HOLD logic ────────────────────────────────────────────────────────
    else:
        action = "HOLD"
        entry_price = close
        take_profit_1 = nearest_resistance
        take_profit_2 = resistance_2
        stop_loss = nearest_support

        reasons.append(f"Mixed/weak signal: direction {direction:+.2f}, confidence {confidence:.0f}%")
        if abs(direction) < 0.1:
            reasons.append("Models disagree on direction — wait for clarity")
        if confidence < 25:
            reasons.append("Low confidence — no edge detected")
        if 45 <= rsi_val <= 55:
            reasons.append(f"RSI {rsi_val:.0f} — neutral zone, no momentum")

        # Still show levels for reference
        dist_to_sup = (close - nearest_support) / close * 100
        dist_to_res = (nearest_resistance - close) / close * 100
        reasons.append(f"Watch support at {nearest_support:,.0f} ({dist_to_sup:.1f}% below)")
        reasons.append(f"Watch resistance at {nearest_resistance:,.0f} ({dist_to_res:.1f}% above)")

        if direction > 0.1:
            reasons.append(f"Leaning bullish — consider BUY on pullback to {nearest_support:,.0f}")
        elif direction < -0.1:
            reasons.append(f"Leaning bearish — consider SELL on rally to {nearest_resistance:,.0f}")

    # ── Risk / Reward ─────────────────────────────────────────────────────
    if action == "SELL":
        # For SELL (long-only exit): reward = distance to downside target,
        # risk = distance to invalidation (resistance)
        reward = abs(entry_price - take_profit_1)
        risk = abs(stop_loss - entry_price)
        reward2 = abs(entry_price - take_profit_2)
    else:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit_1 - entry_price)
        reward2 = abs(take_profit_2 - entry_price)

    risk_reward = round(reward / risk, 2) if risk > 0 else 0.0
    risk_reward_2 = round(reward2 / risk, 2) if risk > 0 else 0.0

    # Strength — require minimum R/R of 1.0 for any action to be STRONG
    if action == "HOLD":
        strength = "MODERATE" if confidence >= 35 else "WEAK"
    elif confidence >= 55 and risk_reward >= 1.5:
        strength = "STRONG"
    elif confidence >= 35 and risk_reward >= 1.0:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    # ── Key levels summary ────────────────────────────────────────────────
    key_levels = {
        "support_1": round(nearest_support, 2),
        "support_2": round(support_2, 2),
        "resistance_1": round(nearest_resistance, 2),
        "resistance_2": round(resistance_2, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "pivot": round(pred["pivot"], 2),
        "r1": round(pred["r1"], 2),
        "s1": round(pred["s1"], 2),
    }

    return {
        "action": action,
        "strength": strength,
        "entry_price": round(entry_price, 2),
        "take_profit_1": round(take_profit_1, 2),
        "take_profit_2": round(take_profit_2, 2),
        "stop_loss": round(stop_loss, 2),
        "risk_reward": risk_reward,
        "risk_reward_2": risk_reward_2,
        "confidence": round(confidence, 1),
        "reasons": reasons,
        "key_levels": key_levels,
        "prediction": pred,
    }


def advise_multi(symbols: list[str], start: str, end: str) -> list[dict]:
    from src.data.fetcher import fetch_history
    results = []
    for sym in symbols:
        try:
            df = fetch_history(sym, start, end)
            adv = advise(df)
            adv["symbol"] = sym
            results.append(adv)
        except Exception as e:
            results.append({"symbol": sym, "action": "ERROR", "error": str(e)})
    return results

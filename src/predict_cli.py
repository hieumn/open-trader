#!/usr/bin/env python3
"""CLI for stock prediction and buy/sell suggestions.

Usage:
    python -m src.predict_cli --symbol AAPL
    python -m src.predict_cli --symbol AAPL MSFT GOOGL SHB.VN
    python -m src.predict_cli --symbol AAPL --from 2024-01-01
    python -m src.predict_cli --symbol AAPL --verbose
"""

import argparse
from datetime import datetime, timedelta

from src.data.fetcher import fetch_history
from src.predictions.advisor import advise, advise_multi


def _print_recommendation(adv: dict) -> None:
    sym = adv.get("symbol", "???")
    if adv["action"] == "ERROR":
        print(f"\n  {sym}: ERROR — {adv.get('error', 'unknown')}")
        return

    pred = adv["prediction"]
    close = pred["close"]
    action = adv["action"]
    strength = adv["strength"]

    if action == "BUY":
        action_str = f"\033[92m★ {action}\033[0m"
    elif action == "SELL":
        action_str = f"\033[91m★ {action}\033[0m"
    else:
        action_str = f"\033[93m● {action}\033[0m"

    direction = pred["direction"]
    dir_bar = _direction_bar(direction)

    print(f"\n{'═' * 66}")
    print(f"  {sym}  —  {close:,.0f} ₫  |  {action_str} ({strength})")
    print(f"{'═' * 66}")
    print(f"  Direction :{dir_bar}  ({direction:+.2f})")
    print(f"  Confidence: {adv['confidence']:.0f}%")
    print(f"  Pred. move: {pred['predicted_move_pct']:+.2f}%")

    # ── Price action plan ─────────────────────────────────────────────────
    print(f"\n  {'─── Price Action Plan ─────────────────────────────────':}")
    entry = adv['entry_price']
    tp1 = adv['take_profit_1']
    tp2 = adv['take_profit_2']
    sl = adv['stop_loss']

    if action == "BUY":
        print(f"  \033[91m  Stop-loss : {sl:>12,.0f} ₫  ({(sl/close-1)*100:+.2f}%)\033[0m")
        print(f"  \033[93m  Entry     : {entry:>12,.0f} ₫  ({(entry/close-1)*100:+.2f}%)\033[0m")
        print(f"       Close   : {close:>12,.0f} ₫")
        print(f"  \033[92m  TP1 (50%) : {tp1:>12,.0f} ₫  ({(tp1/close-1)*100:+.2f}%)\033[0m")
        print(f"  \033[92m  TP2 (50%) : {tp2:>12,.0f} ₫  ({(tp2/close-1)*100:+.2f}%)\033[0m")
    elif action == "SELL":
        print(f"  \033[91m  ⚠ SELL / EXIT your long position at current price\033[0m")
        print(f"       Sell at : {close:>12,.0f} ₫")
        print(f"  \033[93m  Re-buy 1  : {tp1:>12,.0f} ₫  ({(tp1/close-1)*100:+.2f}%)\033[0m")
        print(f"  \033[93m  Re-buy 2  : {tp2:>12,.0f} ₫  ({(tp2/close-1)*100:+.2f}%)\033[0m")
        print(f"  \033[91m  Invalid.  : {sl:>12,.0f} ₫  ({(sl/close-1)*100:+.2f}%) ← sell was wrong if above\033[0m")
    else:
        print(f"  \033[91m  Stop      : {sl:>12,.0f} ₫  ({(sl/close-1)*100:+.2f}%)\033[0m")
        print(f"       Close   : {close:>12,.0f} ₫")
        print(f"  \033[92m  Target 1  : {tp1:>12,.0f} ₫  ({(tp1/close-1)*100:+.2f}%)\033[0m")
        print(f"  \033[92m  Target 2  : {tp2:>12,.0f} ₫  ({(tp2/close-1)*100:+.2f}%)\033[0m")

    print(f"\n  R/R (TP1)  : {adv['risk_reward']:.2f}")
    print(f"  R/R (TP2)  : {adv['risk_reward_2']:.2f}")

    # ── Key levels ────────────────────────────────────────────────────────
    kl = adv.get("key_levels", {})
    if kl:
        print(f"\n  {'─── Key Levels ───────────────────────────────────────':}")
        print(f"    Resistance 2 : {kl.get('resistance_2', 0):>12,.0f} ₫")
        print(f"    Resistance 1 : {kl.get('resistance_1', 0):>12,.0f} ₫")
        print(f"    \033[1mCurrent      : {close:>12,.0f} ₫\033[0m")
        print(f"    Support 1    : {kl.get('support_1', 0):>12,.0f} ₫")
        print(f"    Support 2    : {kl.get('support_2', 0):>12,.0f} ₫")
        print(f"    ────────────────────────────────")
        print(f"    EMA20 : {kl.get('ema20', 0):>10,.0f}    EMA50 : {kl.get('ema50', 0):>10,.0f}")
        print(f"    BB ↑  : {kl.get('bb_upper', 0):>10,.0f}    BB ↓  : {kl.get('bb_lower', 0):>10,.0f}")
        print(f"    Pivot : {kl.get('pivot', 0):>10,.0f}    R1    : {kl.get('r1', 0):>10,.0f}    S1: {kl.get('s1', 0):>10,.0f}")

    # ── Indicators ────────────────────────────────────────────────────────
    print(f"\n  {'─── Indicators ───────────────────────────────────────':}")
    print(f"    RSI : {pred['rsi']:>6.1f}    ADX : {pred['adx']:>6.1f}")
    print(f"    BB% : {pred['bb_pct']:>6.3f}    ATR%: {pred['atr_pct']:>5.2f}%")
    print(f"    MACD hist: {pred['macd_hist']:>+.4f}")

    # ── Reasons ───────────────────────────────────────────────────────────
    print(f"\n  {'─── Analysis ─────────────────────────────────────────':}")
    for r in adv["reasons"]:
        print(f"    • {r}")


def _print_verbose(adv: dict) -> None:
    pred = adv.get("prediction", {})
    models = pred.get("models", {})
    if not models:
        return
    print(f"\n  {'Model':<18} {'Direction':>10} {'Magnitude':>10} {'Confidence':>10}")
    print(f"  {'─' * 52}")
    for name, m in models.items():
        d = m["direction"]
        label = "bullish" if d > 0.1 else "bearish" if d < -0.1 else "neutral"
        print(f"  {name:<18} {d:>+10.3f} {m['magnitude']:>10.3f} {m['confidence']:>9.1f}%  [{label}]")

    # S/R levels detail
    sr = pred.get("support_resistance", {})
    if sr.get("support") or sr.get("resistance"):
        print(f"\n  {'─── Support / Resistance Zones ──────────────────────':}")
        for p, s, src in sr.get("resistance", [])[:5]:
            print(f"    R  {p:>12,.0f} ₫  strength={s:.1f}  ({src})")
        close = pred["close"]
        print(f"    ── {close:>12,.0f} ₫  ◄ CURRENT PRICE")
        for p, s, src in sr.get("support", [])[:5]:
            print(f"    S  {p:>12,.0f} ₫  strength={s:.1f}  ({src})")


def _print_summary_table(results: list[dict]) -> None:
    print(f"\n{'═' * 95}")
    print(f"  {'Symbol':<10} {'Price':>12} {'Action':<8} {'Str':<8} "
          f"{'Conf':>5} {'Dir':>7} {'Entry':>12} {'TP1':>12} {'SL':>12} {'R/R':>5}")
    print(f"{'═' * 95}")

    for adv in results:
        sym = adv.get("symbol", "?")
        if adv["action"] == "ERROR":
            print(f"  {sym:<10} {'ERROR':>12}  {adv.get('error', '')[:60]}")
            continue
        pred = adv["prediction"]
        action = adv["action"]
        if action == "BUY":
            a = f"\033[92m{action:<8}\033[0m"
        elif action == "SELL":
            a = f"\033[91m{action:<8}\033[0m"
        else:
            a = f"\033[93m{action:<8}\033[0m"

        print(f"  {sym:<10} {pred['close']:>11,.0f}₫ {a} {adv['strength']:<8} "
              f"{adv['confidence']:>4.0f}% {pred['direction']:>+7.2f} "
              f"{adv['entry_price']:>11,.0f}₫ {adv['take_profit_1']:>11,.0f}₫ "
              f"{adv['stop_loss']:>11,.0f}₫ {adv['risk_reward']:>5.2f}")


def _direction_bar(direction: float) -> str:
    width = 20
    center = width // 2
    pos = int((direction + 1) / 2 * width)
    pos = max(0, min(width, pos))
    bar = [' '] * (width + 1)
    bar[center] = '│'
    if pos > center:
        for i in range(center + 1, pos + 1):
            bar[i] = '▓'
    elif pos < center:
        for i in range(pos, center):
            bar[i] = '▓'
    return ' [' + ''.join(bar) + ']'


def main():
    parser = argparse.ArgumentParser(description="Stock prediction & buy/sell advisor")
    parser.add_argument("--symbol", "-s", nargs="+", required=True, help="Stock symbol(s)")
    parser.add_argument("--from", dest="start", default=None, help="Start date (default: 1y ago)")
    parser.add_argument("--to", dest="end", default=None, help="End date (default: today)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show model breakdown + S/R zones")
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    results = advise_multi(args.symbol, start, end)

    if len(args.symbol) > 1:
        _print_summary_table(results)

    for adv in results:
        _print_recommendation(adv)
        if args.verbose:
            _print_verbose(adv)

    print()


if __name__ == "__main__":
    main()

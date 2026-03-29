#!/usr/bin/env python3
"""CLI for stock prediction and buy/sell suggestions.

Usage:
    # Single stock
    python -m src.predict_cli --symbol AAPL

    # Multiple symbols
    python -m src.predict_cli --symbol AAPL MSFT GOOGL SHB.VN

    # Custom date range for analysis
    python -m src.predict_cli --symbol AAPL --from 2024-01-01

    # Detailed model breakdown
    python -m src.predict_cli --symbol AAPL --verbose
"""

import argparse
from datetime import datetime, timedelta

from src.data.fetcher import fetch_history
from src.predictions.advisor import advise, advise_multi


def _print_recommendation(adv: dict) -> None:
    """Pretty-print a single stock recommendation."""
    sym = adv.get("symbol", "???")

    if adv["action"] == "ERROR":
        print(f"\n  {sym}: ERROR — {adv.get('error', 'unknown')}")
        return

    pred = adv["prediction"]
    close = pred["close"]

    # Action with color-coding (ANSI)
    action = adv["action"]
    strength = adv["strength"]
    if action == "BUY":
        action_str = f"\033[92m★ {action}\033[0m"   # green
    elif action == "SELL":
        action_str = f"\033[91m★ {action}\033[0m"   # red
    else:
        action_str = f"\033[93m● {action}\033[0m"   # yellow

    direction = pred["direction"]
    dir_bar = _direction_bar(direction)

    print(f"\n{'═' * 62}")
    print(f"  {sym}  —  ${close:,.2f}  |  {action_str} ({strength})")
    print(f"{'═' * 62}")
    print(f"  Direction :{dir_bar}  ({direction:+.2f})")
    print(f"  Confidence: {adv['confidence']:.0f}%")
    print(f"  Pred. move: {pred['predicted_move_pct']:+.2f}%")
    print(f"{'─' * 62}")
    print(f"  Entry price : ${adv['entry_price']:>10,.4f}")
    print(f"  Take-profit : ${adv['take_profit']:>10,.4f}  ({(adv['take_profit']/close-1)*100:+.2f}%)")
    print(f"  Stop-loss   : ${adv['stop_loss']:>10,.4f}  ({(adv['stop_loss']/close-1)*100:+.2f}%)")
    print(f"  Risk/Reward : {adv['risk_reward']:.2f}")
    print(f"{'─' * 62}")
    print(f"  Indicators:")
    print(f"    RSI : {pred['rsi']:>6.1f}    ADX : {pred['adx']:>6.1f}")
    print(f"    BB% : {pred['bb_pct']:>6.3f}    ATR%: {pred['atr_pct']:>5.2f}%")
    print(f"    MACD hist: {pred['macd_hist']:>+.4f}")
    print(f"{'─' * 62}")
    print(f"  Reasons:")
    for r in adv["reasons"]:
        print(f"    • {r}")


def _print_verbose(adv: dict) -> None:
    """Print detailed model breakdown."""
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


def _print_summary_table(results: list[dict]) -> None:
    """Print a compact comparison table for multiple stocks."""
    print(f"\n{'═' * 82}")
    print(f"  {'Symbol':<10} {'Price':>10} {'Action':<8} {'Strength':<10} "
          f"{'Conf%':>6} {'Direction':>10} {'R/R':>6} {'RSI':>6}")
    print(f"{'═' * 82}")

    for adv in results:
        sym = adv.get("symbol", "?")
        if adv["action"] == "ERROR":
            print(f"  {sym:<10} {'ERROR':>10}  {adv.get('error', '')[:50]}")
            continue

        pred = adv["prediction"]
        action = adv["action"]
        if action == "BUY":
            act_str = f"\033[92m{action:<8}\033[0m"
        elif action == "SELL":
            act_str = f"\033[91m{action:<8}\033[0m"
        else:
            act_str = f"\033[93m{action:<8}\033[0m"

        print(f"  {sym:<10} ${pred['close']:>9,.2f} {act_str} {adv['strength']:<10} "
              f"{adv['confidence']:>5.0f}% {pred['direction']:>+10.3f} "
              f"{adv['risk_reward']:>6.2f} {pred['rsi']:>6.1f}")


def _direction_bar(direction: float) -> str:
    """Visual bar for direction -1 to +1."""
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
    parser = argparse.ArgumentParser(
        description="Stock prediction & buy/sell advisor"
    )
    parser.add_argument(
        "--symbol", "-s", nargs="+", required=True,
        help="Stock symbol(s), e.g. AAPL SHB.VN",
    )
    parser.add_argument(
        "--from", dest="start", default=None,
        help="Analysis start date (default: 1 year ago)",
    )
    parser.add_argument(
        "--to", dest="end", default=None,
        help="Analysis end date (default: today)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show individual model breakdown",
    )
    args = parser.parse_args()

    if args.end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    else:
        end = args.end

    if args.start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        start = args.start

    symbols = args.symbol
    results = advise_multi(symbols, start, end)

    # Summary table for multi-symbol
    if len(symbols) > 1:
        _print_summary_table(results)

    # Detailed view for each
    for adv in results:
        _print_recommendation(adv)
        if args.verbose:
            _print_verbose(adv)

    print()


if __name__ == "__main__":
    main()

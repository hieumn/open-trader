# small CLI to run backtest
import argparse
from src.backtest.runner import main as backtest_main

if __name__ == "__main__":
    backtest_main()
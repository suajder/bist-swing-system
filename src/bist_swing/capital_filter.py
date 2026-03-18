import pandas as pd


def capital_filter(equity_series, lookback=50):

    if len(equity_series) < lookback:
        return True

    ma = equity_series.rolling(lookback).mean()

    if equity_series.iloc[-1] < ma.iloc[-1]:
        return False

    return True
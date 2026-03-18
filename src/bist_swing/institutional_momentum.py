import numpy as np
import pandas as pd


def add_institutional_momentum_cols(df: pd.DataFrame) -> pd.DataFrame:

    close = df["Close"].astype(float)

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    mom60 = close.pct_change(60)

    trend_spread = (ema50 - ema200) / ema200

    inst_mom_score = (
        0.4 * mom60 +
        0.4 * trend_spread +
        0.2 * (close / ema50 - 1)
    )

    df["ema50"] = ema50
    df["ema200"] = ema200
    df["mom60"] = mom60
    df["inst_mom_score"] = inst_mom_score

    df["inst_mom_ok"] = (
        (close > ema50) &
        (ema50 > ema200) &
        (mom60 > 0)
    )

    return df

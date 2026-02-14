from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

@dataclass(frozen=True)
class SignalParams:
    d_ema_fast: int = 20
    d_ema_slow: int = 50
    rsi_n: int = 14
    rsi_entry_max: float = 70.0
    mom_n: int = 20
    w_ema_fast: int = 10
    w_ema_slow: int = 30

class SignalEngine:
    def build(self, df: pd.DataFrame, sp: SignalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["d_ema20"] = ema(df["Close"], sp.d_ema_fast)
        out["d_ema50"] = ema(df["Close"], sp.d_ema_slow)
        out["d_rsi14"] = rsi(df["Close"], sp.rsi_n)
        out["atr14"] = atr(df, 14)
        out["mom20"] = df["Close"].pct_change(sp.mom_n)

        # weekly regime: compute on resampled weekly closes
        w = df.resample("W-FRI").agg({"Close":"last"})
        w["w_ema_fast"] = ema(w["Close"], sp.w_ema_fast)
        w["w_ema_slow"] = ema(w["Close"], sp.w_ema_slow)
        w["w_exit_regime"] = w["w_ema_fast"] < w["w_ema_slow"]
        out["w_exit_regime"] = w["w_exit_regime"].reindex(out.index, method="ffill").fillna(False)

        # entry rule (weekly not in exit, daily trend up, RSI not extreme)
        trend_up = out["d_ema20"] > out["d_ema50"]
        rsi_ok = out["d_rsi14"] < sp.rsi_entry_max
        out["entry_signal"] = (trend_up & rsi_ok & (~out["w_exit_regime"])).fillna(False)

        return out

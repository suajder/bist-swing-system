from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from bist_swing.indicators import ema, rsi, atr, zscore

@dataclass(frozen=True)
class SignalParams:
    d_ema_fast: int = 20
    d_ema_slow: int = 50
    rsi_n: int = 14
    rsi_entry_max: float = 70.0
    mom_n: int = 20
    w_ema_fast: int = 10
    w_ema_slow: int = 30
    
    # Yeni eklendi: Mean Reversion Parametreleri
    rsi_short_n: int = 2
    rsi_short_max: float = 15.0
    zscore_n: int = 20
    zscore_min: float = -2.0

class SignalEngine:
    def build(self, df: pd.DataFrame, sp: SignalParams, market_regime_ok: pd.Series = None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["d_ema20"] = ema(df["Close"], sp.d_ema_fast)
        out["d_ema50"] = ema(df["Close"], sp.d_ema_slow)
        out["d_ema200"] = ema(df["Close"], 200) # Uzun vade trend
        out["d_rsi14"] = rsi(df["Close"], sp.rsi_n)
        out["d_rsi_short"] = rsi(df["Close"], sp.rsi_short_n)
        out["zscore"] = zscore(df["Close"], sp.zscore_n)
        out["atr14"] = atr(df, 14)
        out["mom20"] = df["Close"].pct_change(sp.mom_n)

        # weekly regime: compute on resampled weekly closes
        w = df.resample("W-FRI").agg({"Close":"last"})
        w["w_ema_fast"] = ema(w["Close"], sp.w_ema_fast)
        w["w_ema_slow"] = ema(w["Close"], sp.w_ema_slow)
        w["w_exit_regime"] = w["w_ema_fast"] < w["w_ema_slow"]
        out["w_exit_regime"] = w["w_exit_regime"].reindex(out.index, method="ffill").fillna(False)

        # 1. Trend Takibi Sinyali
        trend_up = out["d_ema20"] > out["d_ema50"]
        rsi_ok = out["d_rsi14"] < sp.rsi_entry_max
        trend_entry = trend_up & rsi_ok

        # 2. Mean Reversion Sinyali (Kısa RSI Dibi + Z-Score Sapması)
        long_term_up = out["d_ema50"] > out["d_ema200"]
        mr_entry = long_term_up & (out["d_rsi_short"] < sp.rsi_short_max) & (out["zscore"] < sp.zscore_min)

        # Sinyalleri Birleştir
        raw_entry = trend_entry | mr_entry

        # Rejim Vetosu (Haftalık trend VE isteğe bağlı Endeks Vetosu)
        regime_ok = ~out["w_exit_regime"]
        if market_regime_ok is not None:
             # Eğer Endeks (XU100) verisi sağlanmışsa, Endeks EMA50 üzerindeyse True'dur
             aligned_market_regime = market_regime_ok.reindex(out.index, method="ffill").fillna(False)
             regime_ok = regime_ok & aligned_market_regime

        out["entry_signal"] = (raw_entry & regime_ok).fillna(False)

        return out

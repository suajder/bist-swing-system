from __future__ import annotations

import time
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import requests


class DataProvider:
    def __init__(self, period_days: int = 365 * 3, session: Optional[requests.Session] = None):
        self.period_days = int(period_days)
        self.sess = session or requests.Session()
        self.sess.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        })

    def get(self, ticker: str) -> pd.DataFrame:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

        params = {
            "range": f"{self.period_days}d",
            "interval": "1d",
        }

        r = self.sess.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()

        result = js.get("chart", {}).get("result")
        if not result:
            raise RuntimeError(f"No data for {ticker}")

        res0 = result[0]
        ts = res0["timestamp"]
        q = res0["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Volume": q["volume"],
        })

        df.index = pd.to_datetime(ts, unit="s")
        df = df.dropna()

        # =========================
        # 🔥 CORE FEATURES
        # =========================

        # ADV20
        df["ADV20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

        # EMA
        df["ema50"] = df["Close"].ewm(span=50).mean()
        df["ema200"] = df["Close"].ewm(span=200).mean()

        # ATR (14)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()

        # Volume mean
        df["vol_mean20"] = df["Volume"].rolling(20).mean()

        # Breakout high
        df["high_20"] = df["High"].rolling(20).max()

        # =========================
        # 🔥 SMART FLAGS
        # =========================

        # Trend
        df["trend_ok"] = (df["Close"] > df["ema50"]) & (df["ema50"] > df["ema200"])

        # Breakout
        df["breakout_ok"] = df["Close"] >= df["high_20"]

        # Volume spike
        df["vol_spike"] = df["Volume"] > 1.3 * df["vol_mean20"]

        # Liquidity shock (basit versiyon)
        df["liq_shock"] = df["Volume"] > 1.5 * df["vol_mean20"]

        # Institutional momentum (placeholder ama çalışır)
        df["inst_mom_score"] = (df["Close"] / df["Close"].rolling(20).mean()) - 1
        df["inst_mom_ok"] = df["inst_mom_score"] > 0

        # =========================

        df = df.dropna()
        df = df.sort_index()

        time.sleep(0.2)

        return df


def load_price_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    dp = DataProvider()
    price_map = {}

    print(f"\nDownloading {len(tickers)} tickers...")

    for sym in tickers:
        try:
            df = dp.get(sym)
            price_map[sym] = df
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

    print(f"Downloaded OK: {len(price_map)}/{len(tickers)}")

    return price_map


def get_price_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    provider = DataProvider()
    price_map = {}

    for t in tickers:
        try:
            df = provider.get(t)
            price_map[t] = df
        except Exception as e:
            print(f"[WARN] {t} failed: {e}")

    return price_map
from __future__ import annotations

import time
from typing import Optional, Dict, List

import pandas as pd
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

        # ADV20
        df["ADV20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

        df = df.dropna()
        df = df.sort_index()

        time.sleep(0.2)

        return df


# 🔥 KRİTİK FONKSİYON (SENDE EKSİKTİ)
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
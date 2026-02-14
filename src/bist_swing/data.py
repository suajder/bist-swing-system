'''from __future__ import annotations
import pandas as pd
import yfinance as yf

class DataProvider:
    def __init__(self, period: str = "3y", interval: str = "1d"):
        self.period = period
        self.interval = interval

    def get(self, ticker: str) -> pd.DataFrame:
        df = yf.download(
            tickers=ticker,
            period=self.period,
            interval=self.interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if df is None or df.empty:
            raise RuntimeError(f"No data for {ticker}")
        # normalize columns
        need = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns {missing} for {ticker}: {list(df.columns)}")

        out = df[need].copy()
        # ADV20 (approx): Close * rolling mean volume
        out["ADV20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
        return out.dropna()'''
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests


class DataProvider:
    """
    Fetch daily OHLCV from Yahoo chart endpoint using requests (avoids yfinance/libcurl TLS issues).
    """

    def __init__(self, period_days: int = 365 * 3, session: Optional[requests.Session] = None):
        self.period_days = int(period_days)
        self.sess = session or requests.Session()
        self.sess.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )

    def get(self, ticker: str) -> pd.DataFrame:
        # Yahoo chart endpoint (public)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "range": f"{self.period_days}d",
            "interval": "1d",
            "includePrePost": "false",
            "events": "div|split",
        }

        r = self.sess.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()

        chart = js.get("chart", {})
        err = chart.get("error")
        if err:
            raise RuntimeError(f"Yahoo error: {err}")

        result = chart.get("result")
        if not result:
            raise RuntimeError("No result from Yahoo chart endpoint")

        res0 = result[0]
        ts = res0.get("timestamp", [])
        ind = res0.get("indicators", {}).get("quote", [])
        if not ts or not ind:
            raise RuntimeError("Empty OHLCV in Yahoo response")

        q = ind[0]
        df = pd.DataFrame(
            {
                "Open": q.get("open", []),
                "High": q.get("high", []),
                "Low": q.get("low", []),
                "Close": q.get("close", []),
                "Volume": q.get("volume", []),
            }
        )

        # timestamps are unix seconds
        idx = pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s").dt.tz_localize("UTC").dt.tz_convert(None)
        df.index = idx
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

        # ADV20 (Close * rolling mean volume)
        df["ADV20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
        df = df.dropna()

        # ensure numeric
        for c in ["Open", "High", "Low", "Close", "Volume", "ADV20"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna()
        df = df.sort_index()

        # polite pacing
        time.sleep(0.25)
        return df


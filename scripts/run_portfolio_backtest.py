from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

import yfinance as yf

from bist_swing.portfolio import portfolio_backtest_pro, PortfolioParams
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "configs" / "universe.txt"
OUTDIR = ROOT / "out" / "portfolio"


def read_universe(path: Path) -> List[str]:
    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)
    return tickers


def download_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance bazen kolonları MultiIndex döndürebiliyor; düzleştir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[list(need)].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def add_adv20(df: pd.DataFrame) -> pd.DataFrame:
    # ADV20 = 20g ort. işlem değeri (Close * Volume)
    out = df.copy()
    out["ADV20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    return out


def main():
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    tickers = read_universe(UNIVERSE_FILE)
    if not tickers:
        raise SystemExit("Universe is empty.")

    # makul test aralığı
    start = "2022-01-01"
    end = None  # today

    print(f"Downloading {len(tickers)} tickers...")
    price_map: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = download_ohlc(t, start=start, end=end)
        if df.empty:
            continue
        df = add_adv20(df)
        price_map[t] = df

    tickers_ok = sorted(price_map.keys())
    print(f"Downloaded OK: {len(tickers_ok)}/{len(tickers)}")

    if len(tickers_ok) < 5:
        raise SystemExit("Too few tickers downloaded; check tickers or network.")

    # Basit: hepsi için aynı param set (optimizasyon yok)
    sp = SignalParams()
    bp = BacktestParams()

    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {t: (sp, bp) for t in tickers_ok}

    # model score yoksa 0 ver (rank sadece RSI/MOM/ADV ile de çalışır)
    model_score_map = {t: 0.0 for t in tickers_ok}

    se = SignalEngine()

    pparams = PortfolioParams(
        max_open=3,
        initial_equity=50_000.0,  # senin hedef başlangıcın
        adv20_min=50_000_000.0,   # istersen sonra düşürürüz
    )

    res = portfolio_backtest_pro(
        se=se,
        tickers=tickers_ok,
        price_map=price_map,
        best_cfg_map=best_cfg_map,
        model_score_map=model_score_map,
        test_start="2023-01-01",
        test_end=None,
        pparams=pparams,
        outdir=OUTDIR,
    )

    eq = res.get("equity_curve", pd.DataFrame())
    tr = res.get("trades", pd.DataFrame())

    print("\nDone.")
    print("Equity rows:", getattr(eq, "shape", None))
    print("Trades rows:", getattr(tr, "shape", None))
    print(f"Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()

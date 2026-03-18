from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from bist_swing.portfolio import portfolio_backtest_pro, PortfolioParams
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "configs" / "universe.txt"
OUTDIR = ROOT / "out" / "portfolio"
FAILED_OUT = ROOT / "out" / "failed_tickers.txt"


def read_universe(path: Path) -> List[str]:
    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        tickers.append(s)
    return tickers


def write_universe_pruned(path: Path, keep: List[str]) -> None:
    """
    Removes failed tickers from universe.txt, with a one-time .bak backup.
    """
    bak = path.with_suffix(path.suffix + ".bak")  # universe.txt.bak
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    # rewrite as simple list (keeps it deterministic)
    out = "\n".join(keep) + "\n"
    path.write_text(out, encoding="utf-8")


def download_ohlc(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    out = df[list(need)].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def add_adv20(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ADV20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    return out


def main():
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    tickers = read_universe(UNIVERSE_FILE)
    if not tickers:
        raise SystemExit("Universe is empty.")
    
    benchmark_ticker = "XU100.IS"
    download_list = sorted(set(tickers + [benchmark_ticker]))

    # backtest window
    dl_start = "2022-01-01"
    dl_end = None

    print(f"Downloading {len(download_list)} tickers...")
    price_map: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for t in download_list:
        try:
            df = download_ohlc(t, start=dl_start, end=dl_end)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            failed.append(t)
            continue
        df = add_adv20(df)
        price_map[t] = df

    tickers_ok = sorted([t for t in tickers if t in price_map])
    print(f"Downloaded OK: {len(tickers_ok)}/{len(tickers)}")

    if failed:
        FAILED_OUT.parent.mkdir(parents=True, exist_ok=True)
        FAILED_OUT.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failed tickers: {failed}")

        # auto-prune from universe (optional but desired)
        write_universe_pruned(UNIVERSE_FILE, tickers_ok)
        print(f"Pruned universe updated: {UNIVERSE_FILE} (backup: {UNIVERSE_FILE}.bak)")

    if len(tickers_ok) < 5:
        raise SystemExit("Too few tickers downloaded; check tickers or network.")

    # same params for all tickers (no optimization)
    sp = SignalParams()
    bp = BacktestParams()
    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {t: (sp, bp) for t in tickers_ok}
    model_score_map = {t: 0.0 for t in tickers_ok}

    se = SignalEngine()

    pparams = PortfolioParams(
        max_open=3,
        initial_equity=50_000.0,
        adv20_min=50_000_000.0,
        risk_pct=0.02,
        min_atr_pct=0.0,
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
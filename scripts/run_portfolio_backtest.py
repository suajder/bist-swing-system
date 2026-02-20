from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

def prune_universe_file(universe_path: Path, failed: List[str]) -> None:
    """
    Remove failed tickers from universe file (creates a .bak backup).
    Keeps comments/blank lines; only removes exact ticker lines.
    """
    if not failed:
        return
    if not universe_path.exists():
        return

    failed_set = {s.strip() for s in failed if s and s.strip()}
    if not failed_set:
        return

    src = universe_path.read_text(encoding="utf-8").splitlines()
    kept: List[str] = []
    removed: List[str] = []

    for line in src:
        raw = line.strip()
        if (not raw) or raw.startswith("#"):
            kept.append(line)
            continue
        if raw in failed_set:
            removed.append(raw)
            continue
        kept.append(line)

    if not removed:
        return

    bak = universe_path.with_suffix(universe_path.suffix + ".bak")
    bak.write_text("\n".join(src) + "\n", encoding="utf-8")
    universe_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(f"Pruned universe: removed {len(removed)} ticker(s). Backup: {bak.name}")


def download_ohlc(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()

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
    out = df.copy()
    out["ADV20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    tickers = read_universe(UNIVERSE_FILE)
    if not tickers:
        raise SystemExit("Universe is empty.")

    start = "2022-01-01"
    end = None  # today

    print(f"Downloading {len(tickers)} tickers...")
    price_map: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for t in tickers:
        df = download_ohlc(t, start=start, end=end)
        if df.empty:
            failed.append(t)
            continue
        df = add_adv20(df)
        price_map[t] = df

    tickers_ok = sorted(price_map.keys())
    print(f"Downloaded OK: {len(tickers_ok)}/{len(tickers)}")
    if failed:
        print("Failed tickers:", failed)
        (OUTDIR / "failed_tickers.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
        prune_universe_file(UNIVERSE_FILE, failed)

    if len(tickers_ok) < 5:
        raise SystemExit("Too few tickers downloaded; check tickers or network.")

    sp = SignalParams()
    bp = BacktestParams()
    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {t: (sp, bp) for t in tickers_ok}
    model_score_map = {t: 0.0 for t in tickers_ok}

    se = SignalEngine()

    pparams = PortfolioParams(
        max_open=3,
        initial_equity=50_000.0,
        adv20_min=50_000_000.0,
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
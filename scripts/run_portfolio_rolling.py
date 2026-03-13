# scripts/run_portfolio_rolling.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from bist_swing.portfolio import portfolio_backtest_pro, PortfolioParams
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "configs" / "universe.txt"
OUT_ROOT = ROOT / "out" / "rolling"


# ----------------------------
# Data helpers (same spirit as runner)
# ----------------------------
def read_universe(path: Path) -> List[str]:
    tickers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)
    return tickers


def download_ohlc(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns; flatten
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


def month_add(ts: pd.Timestamp, n_months: int) -> pd.Timestamp:
    # Safe month add using DateOffset (keeps day where possible)
    return (ts + pd.DateOffset(months=int(n_months))).normalize()


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Rolling orchestration
# ----------------------------
def build_windows(
    *,
    start: str,
    end: str,
    window_months: int,
    step_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e <= s:
        raise ValueError("end must be after start")

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = s
    while True:
        w_end = month_add(cur, window_months)
        if w_end > e:
            break
        windows.append((cur, w_end))
        cur = month_add(cur, step_months)

    return windows


def summarize_equity(eqdf: pd.DataFrame) -> Dict[str, float]:
    """
    Basic equity stats:
      - total_return_pct
      - max_dd_pct
      - vol_daily (stdev of daily returns)
    """
    if eqdf is None or eqdf.empty or "Equity" not in eqdf.columns:
        return {}

    eq = eqdf["Equity"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(eq) < 5:
        return {}

    ret = eq.pct_change().dropna()
    total_return_pct = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    max_dd_pct = dd.min() * 100.0  # negative

    vol_daily = ret.std(ddof=0) * 100.0

    return {
        "total_return_pct": float(total_return_pct),
        "max_dd_pct": float(max_dd_pct),
        "vol_daily_pct": float(vol_daily),
    }


def main():
    # ----------------------------
    # USER KNOBS (edit here)
    # ----------------------------
    # Overall data download range (wider than test range, so indicators have warmup)
    download_start = "2022-01-01"
    download_end = None  # today

    # Rolling evaluation horizon
    rolling_start = "2023-01-01"
    rolling_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Window mechanics
    window_months = 12
    step_months = 3

    # Portfolio params (same defaults you used; tweak if you want)
    pparams = PortfolioParams(
        max_open=3,
        initial_equity=50_000.0,
        adv20_min=50_000_000.0,
    )

    # Strategy params (single set; later grid search can wrap this)
    sp = SignalParams()
    bp = BacktestParams()

    # ----------------------------
    # Setup
    # ----------------------------
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    tickers = read_universe(UNIVERSE_FILE)
    if not tickers:
        raise SystemExit("Universe is empty.")
    
    benchmark_ticker = "XU100.IS"
    download_list = sorted(set(tickers + [benchmark_ticker]))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build rolling windows
    windows = build_windows(
        start=rolling_start,
        end=rolling_end,
        window_months=window_months,
        step_months=step_months,
    )
    if not windows:
        raise SystemExit("No windows produced; check rolling_start/end and window_months.")

    print(f"Universe: {len(tickers)} tickers")
    print(f"Rolling windows: {len(windows)}  |  window={window_months}mo step={step_months}mo")
    print(f"Download range: {download_start} -> {download_end or 'today'}")
    print("Downloading price data...")

    # Download prices once, reuse
    price_map: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    for t in download_list:
        df = download_ohlc(t, start=download_start, end=download_end)
        if df.empty:
            failed.append(t)
            continue
        df = add_adv20(df)
        price_map[t] = df

    tickers_ok = sorted([t for t in tickers if t in price_map])
    print(f"Downloaded OK: {len(tickers_ok)}/{len(tickers)}")
    if failed:
        (OUT_ROOT / "failed_tickers_download.txt").write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failed tickers written to: {OUT_ROOT / 'failed_tickers_download.txt'}")

    if len(tickers_ok) < 5:
        raise SystemExit("Too few tickers downloaded; check network/tickers.")

    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {t: (sp, bp) for t in tickers_ok}
    model_score_map = {t: 0.0 for t in tickers_ok}
    se = SignalEngine()

    # Save run config snapshot (good for reproducibility)
    cfg = {
        "download_start": download_start,
        "download_end": download_end,
        "rolling_start": rolling_start,
        "rolling_end": rolling_end,
        "window_months": window_months,
        "step_months": step_months,
        "n_universe": len(tickers),
        "n_ok": len(tickers_ok),
        "portfolio_params": asdict(pparams),
    }
    (OUT_ROOT / "rolling_config.json").write_text(pd.Series(cfg).to_json(indent=2), encoding="utf-8")

    # ----------------------------
    # Run windows
    # ----------------------------
    rows: List[Dict[str, object]] = []

    for k, (w_start, w_end) in enumerate(windows, start=1):
        tag = f"W{k:02d}_{w_start.strftime('%Y%m%d')}_{w_end.strftime('%Y%m%d')}"
        outdir = OUT_ROOT / tag
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{k}/{len(windows)}] Window: {w_start.date()} -> {w_end.date()}  ({tag})")

        res = portfolio_backtest_pro(
            se=se,
            tickers=tickers_ok,
            price_map=price_map,
            best_cfg_map=best_cfg_map,
            model_score_map=model_score_map,
            test_start=w_start.strftime("%Y-%m-%d"),
            test_end=w_end.strftime("%Y-%m-%d"),
            pparams=pparams,
            outdir=outdir,
        )

        eq = res.get("equity_curve", pd.DataFrame())
        tr = res.get("trades", pd.DataFrame())

        # Load r_summary emitted by portfolio_backtest_pro (if exists)
        rsum_path = outdir / "r_summary.csv"
        rsum: Dict[str, float] = {}
        if rsum_path.exists():
            # file format: key,value (Series.to_csv(header=False))
            # robust parse:
            s = pd.read_csv(rsum_path, header=None, index_col=0).iloc[:, 0]
            rsum = {str(i): _safe_float(v) for i, v in s.items()}

        eqsum = summarize_equity(eq)

        row: Dict[str, object] = {
            "window_id": tag,
            "test_start": w_start.strftime("%Y-%m-%d"),
            "test_end": w_end.strftime("%Y-%m-%d"),
            "n_trades_rows": int(getattr(tr, "shape", (0, 0))[0]),
            "n_equity_rows": int(getattr(eq, "shape", (0, 0))[0]),
        }
        row.update(rsum)
        row.update(eqsum)
        rows.append(row)

    # ----------------------------
    # Aggregate report
    # ----------------------------
    summary = pd.DataFrame(rows)

    # stable column ordering (best effort)
    preferred = [
        "window_id", "test_start", "test_end",
        "n_exits", "total_R", "avg_R", "median_R", "win_rate",
        "avg_win_R", "avg_loss_R", "expectancy_R",
        "max_dd_R", "max_loss_streak_n", "max_consec_loss_R",
        "total_return_pct", "max_dd_pct", "vol_daily_pct",
        "n_trades_rows", "n_equity_rows",
    ]
    cols = [c for c in preferred if c in summary.columns] + [c for c in summary.columns if c not in preferred]
    summary = summary[cols]

    out_csv = OUT_ROOT / "rolling_summary.csv"
    summary.to_csv(out_csv, index=False)
    print("\n====================")
    print("ROLLING SUMMARY SAVED:", out_csv)
    print("====================")

    # Quick console diagnostics (robustness)
    if "total_R" in summary.columns:
        vals = pd.to_numeric(summary["total_R"], errors="coerce").dropna()
        if len(vals):
            pos = float((vals > 0).mean()) * 100.0
            print(f"Windows positive total_R: {pos:.1f}%  |  median total_R: {vals.median():.4f}  |  worst total_R: {vals.min():.4f}")

    if "max_dd_R" in summary.columns:
        vals = pd.to_numeric(summary["max_dd_R"], errors="coerce").dropna()
        if len(vals):
            print(f"Worst max_dd_R across windows: {vals.min():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
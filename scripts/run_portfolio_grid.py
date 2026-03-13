from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from bist_swing.portfolio import portfolio_backtest_pro, PortfolioParams
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = ROOT / "configs" / "universe.txt"
OUT_ROOT = ROOT / "out" / "grid"


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
    return (ts + pd.DateOffset(months=int(n_months))).normalize()


def build_windows(
    *,
    start: str,
    end: str,
    window_months: int,
    step_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = s
    while True:
        w_end = month_add(cur, window_months)
        if w_end > e:
            break
        windows.append((cur, w_end))
        cur = month_add(cur, step_months)
    return windows


def summarize_r_from_trades(trdf: pd.DataFrame) -> Dict[str, float]:
    if trdf.empty or "Type" not in trdf.columns or "R_PnL" not in trdf.columns:
        return {}

    ex = trdf[trdf["Type"] != "ENTRY"].copy()
    if ex.empty:
        return {}

    r = ex["R_PnL"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}

    cum = r.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    wins = r[r > 0]
    losses = r[r < 0]

    return {
        "n_exits": float(len(r)),
        "total_R": float(r.sum()),
        "avg_R": float(r.mean()),
        "median_R": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_R": float(r.mean()),
        "max_dd_R": float(dd.min()) if len(dd) else 0.0,
    }


def score_combo(
    *,
    median_total_r: float,
    worst_max_dd_r: float,
    positive_windows_pct: float,
    mean_expectancy_r: float,
    full_total_r: float,
) -> float:
    """
    Robustness-first score.
    Higher is better.
    """
    return (
        1.20 * median_total_r
        + 0.80 * mean_expectancy_r
        + 0.15 * full_total_r
        + 0.04 * positive_windows_pct
        - 0.50 * abs(worst_max_dd_r)
    )


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Date config
    # --------------------------------------------------
    download_start = "2022-01-01"
    download_end = None

    rolling_start = "2023-01-01"
    rolling_end = pd.Timestamp.today().strftime("%Y-%m-%d")
    window_months = 12
    step_months = 3

    # --------------------------------------------------
    # Base params (BASE REGIME ONLY)
    # --------------------------------------------------
    base_pp = PortfolioParams(
        max_open=3,
        initial_equity=50_000.0,
        adv20_min=50_000_000.0,
        weekly_partial_fraction=1.0 / 3.0,
        weekly_partial_min_r=1.0,
        use_market_regime=True,
        use_rs_filter=False,
        use_trend_strength_filter=False,
        use_price_gap_filter=False,
    )

    base_sp = SignalParams()
    base_bp = BacktestParams()

    # --------------------------------------------------
    # Grid space (daraltılmış, daha mantıklı)
    # --------------------------------------------------
    tp1_grid = [0.8, 1.0, 1.2]
    tp2_grid = [2.5, 3.0, 3.5]
    weekly_fraction_grid = [0.25, 1.0 / 3.0, 0.50]
    weekly_min_r_grid = [0.5, 1.0]

    grid = list(product(tp1_grid, tp2_grid, weekly_fraction_grid, weekly_min_r_grid))

    # --------------------------------------------------
    # Universe + benchmark
    # --------------------------------------------------
    if not UNIVERSE_FILE.exists():
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    tickers = read_universe(UNIVERSE_FILE)
    if not tickers:
        raise SystemExit("Universe is empty.")

    benchmark_ticker = "XU100.IS"
    download_list = sorted(set(tickers + [benchmark_ticker]))

    windows = build_windows(
        start=rolling_start,
        end=rolling_end,
        window_months=window_months,
        step_months=step_months,
    )
    if not windows:
        raise SystemExit("No rolling windows produced.")

    print(f"Universe: {len(tickers)} tickers")
    print(f"Download list: {len(download_list)} tickers (incl. benchmark)")
    print(f"Grid size: {len(grid)}")
    print(f"Rolling windows: {len(windows)}")
    print("Downloading price data once...")

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

    if len(tickers_ok) < 5:
        raise SystemExit("Too few tickers downloaded.")

    se = SignalEngine()
    results: List[Dict[str, object]] = []

    total_combos = len(grid)
    for i, (tp1_r, tp2_r, weekly_frac, weekly_min_r) in enumerate(grid, start=1):
        print(
            f"\n[{i}/{total_combos}] "
            f"tp1={tp1_r:.2f} tp2={tp2_r:.2f} "
            f"weekly_frac={weekly_frac:.3f} weekly_min_r={weekly_min_r:.2f}"
        )

        bp = replace(base_bp, tp1_R=float(tp1_r), tp2_R=float(tp2_r))

        pp_kwargs = asdict(base_pp)
        pp_kwargs["weekly_partial_fraction"] = float(weekly_frac)
        pp_kwargs["weekly_partial_min_r"] = float(weekly_min_r)
        pp = PortfolioParams(**pp_kwargs)

        best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {
            t: (base_sp, bp) for t in tickers_ok
        }
        model_score_map = {t: 0.0 for t in tickers_ok}

        window_rows: List[Dict[str, float]] = []

        for w_start, w_end in windows:
            tag = f"{w_start.strftime('%Y%m%d')}_{w_end.strftime('%Y%m%d')}"
            outdir = OUT_ROOT / "runs" / f"combo_{i:03d}" / tag
            outdir.mkdir(parents=True, exist_ok=True)

            res = portfolio_backtest_pro(
                se=se,
                tickers=tickers_ok,
                price_map=price_map,
                best_cfg_map=best_cfg_map,
                model_score_map=model_score_map,
                test_start=w_start.strftime("%Y-%m-%d"),
                test_end=w_end.strftime("%Y-%m-%d"),
                pparams=pp,
                outdir=outdir,
            )

            tr = res.get("trades", pd.DataFrame())
            rsum = summarize_r_from_trades(tr)
            if rsum:
                window_rows.append(rsum)

        if not window_rows:
            continue

        wdf = pd.DataFrame(window_rows)

        median_total_r = float(wdf["total_R"].median())
        worst_total_r = float(wdf["total_R"].min())
        worst_max_dd_r = float(wdf["max_dd_R"].min())
        mean_expectancy_r = float(wdf["expectancy_R"].mean())
        positive_windows_pct = float((wdf["total_R"] > 0).mean() * 100.0)

        full_outdir = OUT_ROOT / "runs" / f"combo_{i:03d}" / "FULL"
        full_outdir.mkdir(parents=True, exist_ok=True)

        full_res = portfolio_backtest_pro(
            se=se,
            tickers=tickers_ok,
            price_map=price_map,
            best_cfg_map=best_cfg_map,
            model_score_map=model_score_map,
            test_start=rolling_start,
            test_end=None,
            pparams=pp,
            outdir=full_outdir,
        )
        full_tr = full_res.get("trades", pd.DataFrame())
        full_rsum = summarize_r_from_trades(full_tr)
        full_total_r = float(full_rsum.get("total_R", np.nan))

        score = score_combo(
            median_total_r=median_total_r,
            worst_max_dd_r=worst_max_dd_r,
            positive_windows_pct=positive_windows_pct,
            mean_expectancy_r=mean_expectancy_r,
            full_total_r=full_total_r,
        )

        results.append(
            {
                "combo_id": i,
                "tp1_R": float(tp1_r),
                "tp2_R": float(tp2_r),
                "weekly_partial_fraction": float(weekly_frac),
                "weekly_partial_min_r": float(weekly_min_r),
                "median_total_R": median_total_r,
                "worst_total_R": worst_total_r,
                "worst_max_dd_R": worst_max_dd_r,
                "mean_expectancy_R": mean_expectancy_r,
                "positive_windows_pct": positive_windows_pct,
                "full_total_R": full_total_r,
                "score": score,
            }
        )

    if not results:
        raise SystemExit("No grid results produced.")

    resdf = pd.DataFrame(results)

    # güvenlik filtresi: çok kötü kombinasyonları alta it
    resdf["feasible"] = (
        (resdf["positive_windows_pct"] >= 55.0)
        & (resdf["worst_max_dd_R"] > -15.0)
    )

    resdf = resdf.sort_values(
        ["feasible", "score", "median_total_R", "full_total_R"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out_csv = OUT_ROOT / "grid_results.csv"
    resdf.to_csv(out_csv, index=False)

    print("\n====================")
    print("GRID RESULTS SAVED:", out_csv)
    print("====================")
    print("\nTop 15 combos:")
    print(resdf.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
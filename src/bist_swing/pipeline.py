from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .backtest import Backtester, BacktestParams
from .data import DataProvider
from .reporting import write_text
from .selection import expand_grid
from .signals import SignalEngine, SignalParams


def _score(metrics: dict) -> float:
    # Simple, robust scoring: reward CAGR, penalize drawdown
    cagr = float(metrics.get("cagr", np.nan))
    mdd = float(metrics.get("max_dd", np.nan))
    if not np.isfinite(cagr) or not np.isfinite(mdd):
        return -1e9
    return cagr + 0.5 * mdd  # mdd is negative


def run_universe_scan(
    *,
    provider: DataProvider,
    se: SignalEngine,
    bt: Backtester,
    tickers: List[str],
    train_end: str,
    base_bp: BacktestParams,
    grid: dict,
    outdir: Path,
    top_n_train: int = 8,
    top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)


    results = []
    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]] = {}
    model_score_map: Dict[str, float] = {}
    price_map: Dict[str, pd.DataFrame] = {}

    base_sp = SignalParams()

    combos = expand_grid(grid, base_sp, base_bp)

    for t in tickers:
        try:
            df = provider.get(t)
            price_map[t] = df
        except Exception as e:
            results.append({"ticker": t, "status": "data_error", "err": str(e)})
            continue

        # Train: up to train_end
        best = (-1e18, None, None, None)
        for sp, bp in combos:
            run = bt.run(daily=df, se=se, sp=sp, bp=bp, start=None, end=train_end, initial_equity=1.0)
            s = _score(run["metrics"])
            if s > best[0]:
                best = (s, sp, bp, run)

        if best[1] is None:
            results.append({"ticker": t, "status": "no_model"})
            continue

        best_score, best_sp, best_bp, best_run = best
        best_cfg_map[t] = (best_sp, best_bp)
        model_score_map[t] = float(best_score)

        results.append(
            {
                "ticker": t,
                "status": "ok",
                "train_score": float(best_score),
                "train_cagr": float(best_run["metrics"].get("cagr", np.nan)),
                "train_max_dd": float(best_run["metrics"].get("max_dd", np.nan)),
                "train_sharpe": float(best_run["metrics"].get("sharpe", np.nan)),
            }
        )

        # Write per-ticker summary
        txt = [
            f"Ticker: {t}",
            f"Train end: {train_end}",
            f"Best train score: {best_score:.6f}",
            f"Train metrics: {best_run['metrics']}",
            f"Best SignalParams: {best_sp}",
            f"Best BacktestParams: {best_bp}",
        ]
        (outdir / "per_ticker").mkdir(parents=True, exist_ok=True)
        write_text(outdir / "per_ticker" / f"{t.replace('.','_')}.txt", txt)

    dfres = pd.DataFrame(results)
    dfres.to_csv(outdir / "UNIVERSE_summary.csv", index=False)

    ok = dfres[dfres["status"] == "ok"].copy()
    ranked = ok.sort_values("train_score", ascending=False).head(int(top_n_train)).copy()

    # For Top-K, we keep same ranking (train_score). You can change to test performance later if desired.
    ranked_topk = ranked.head(int(top_k)).copy().reset_index(drop=True)
    ranked_topk.attrs["best_cfg_map"] = best_cfg_map
    ranked_topk.attrs["model_score_map"] = model_score_map
    ranked_topk.attrs["price_map"] = price_map

    ranked_topk.to_csv(outdir / "UNIVERSE_ranked_topk.csv", index=False)
    return dfres, ranked_topk

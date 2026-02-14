from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestParams
from .reporting import plot_equity
from .signals import SignalEngine, SignalParams


@dataclass(frozen=True)
class PortfolioParams:
    max_open: int = 3
    initial_equity: float = 1_000_000.0
    adv20_min: float = 50_000_000.0
    w_model: float = 1.0
    w_rsi: float = 0.15
    w_mom: float = 0.10
    w_adv: float = 0.05


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - mu) / sd


@dataclass
class _Pos:
    entry_px: float
    stop_px: float
    R: float
    shares: float
    orig_shares: float
    tp1: bool = False
    tp2: bool = False


def portfolio_backtest_pro(
    *,
    se: SignalEngine,
    tickers: List[str],
    price_map: Dict[str, pd.DataFrame],
    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]],
    model_score_map: Dict[str, float],
    test_start: str,
    test_end: Optional[str],
    pparams: PortfolioParams,
    outdir: Path,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(test_start)
    end_dt = pd.to_datetime(test_end) if test_end else None

    # build signals per ticker (full history)
    sig_map = {t: se.build(price_map[t], best_cfg_map[t][0]) for t in tickers}

    # common calendar: use first ticker
    cal = price_map[tickers[0]].index
    cal = cal[cal >= start_dt]
    if end_dt is not None:
        cal = cal[cal <= end_dt]
    if len(cal) < 10:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame()}

    cash = float(pparams.initial_equity)
    positions: Dict[str, _Pos] = {}
    trades = []
    equity_rows = []

    # Use params from first ticker for costs (or could keep per-ticker)
    any_bp = best_cfg_map[tickers[0]][1]
    slip = any_bp.slippage_bps / 10000.0
    fee = any_bp.fee_bps / 10000.0

    for i in range(len(cal) - 1):
        t = cal[i]
        nxt = cal[i + 1]

        # mark-to-market close
        eq = cash
        for sym, pos in positions.items():
            df = price_map[sym]
            if t in df.index:
                eq += pos.shares * float(df.loc[t, "Close"])
        equity_rows.append((t, eq, cash, len(positions)))

        # manage exits on nxt OHLC
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = price_map[sym]
            sig = sig_map[sym]
            _, bp = best_cfg_map[sym]

            if t not in df.index or nxt not in df.index or t not in sig.index:
                continue

            o2 = float(df.loc[nxt, "Open"])
            h2 = float(df.loc[nxt, "High"])
            l2 = float(df.loc[nxt, "Low"])

            # weekly exit
            if bool(sig.loc[t, "w_exit_regime"]):
                px = o2 * (1 - slip)
                proceeds = pos.shares * px * (1 - fee)
                cash += proceeds
                trades.append((nxt, sym, "WeeklyExit", px, pos.shares))
                positions.pop(sym, None)
                continue

            # stop
            if l2 <= pos.stop_px:
                px = pos.stop_px * (1 - slip)
                proceeds = pos.shares * px * (1 - fee)
                cash += proceeds
                trades.append((nxt, sym, "Stop", px, pos.shares))
                positions.pop(sym, None)
                continue

            # TP1/TP2
            tp1 = pos.entry_px + bp.tp1_R * pos.R
            tp2 = pos.entry_px + bp.tp2_R * pos.R

            if (not pos.tp1) and (h2 >= tp1):
                qty = min(pos.orig_shares / 3.0, pos.shares)
                px = tp1 * (1 - slip)
                proceeds = qty * px * (1 - fee)
                cash += proceeds
                pos.shares -= qty
                pos.tp1 = True
                trades.append((nxt, sym, "TP1", px, qty))

            if (not pos.tp2) and (h2 >= tp2):
                qty = min(pos.orig_shares / 3.0, pos.shares)
                px = tp2 * (1 - slip)
                proceeds = qty * px * (1 - fee)
                cash += proceeds
                pos.shares -= qty
                pos.tp2 = True
                trades.append((nxt, sym, "TP2", px, qty))

            if pos.shares <= 1e-12:
                positions.pop(sym, None)
            else:
                positions[sym] = pos

        # entries: build candidate list at t, fill next open
        cap = pparams.max_open - len(positions)
        if cap > 0:
            cands = []
            for sym in tickers:
                if sym in positions:
                    continue
                df = price_map[sym]
                sig = sig_map[sym]
                if t in sig.index and bool(sig.loc[t, "entry_signal"]):
                    if t in df.index:
                        adv20 = safe_float(df.loc[t, "ADV20"], np.nan)
                        if np.isfinite(adv20) and adv20 >= pparams.adv20_min:
                            cands.append(sym)

            if cands:
                # rank cands by composite score
                rows = []
                for sym in cands:
                    df = price_map[sym]
                    sig = sig_map[sym]
                    adv20 = safe_float(df.loc[t, "ADV20"], np.nan)
                    rsi14 = safe_float(sig.loc[t, "d_rsi14"], np.nan)
                    mom20 = safe_float(sig.loc[t, "mom20"], np.nan)
                    rows.append((sym, adv20, rsi14, mom20))

                tmp = pd.DataFrame(rows, columns=["ticker", "adv20", "rsi", "mom"]).set_index("ticker")
                tmp["log_adv"] = np.log(tmp["adv20"].clip(lower=1.0))
                tmp["z_rsi"] = zscore(tmp["rsi"])
                tmp["z_mom"] = zscore(tmp["mom"])
                tmp["z_adv"] = zscore(tmp["log_adv"])

                scores = {}
                for sym in tmp.index:
                    s_model = model_score_map.get(sym, 0.0)
                    scores[sym] = float(
                        pparams.w_model * s_model
                        + pparams.w_rsi * float(tmp.loc[sym, "z_rsi"])
                        + pparams.w_mom * float(tmp.loc[sym, "z_mom"])
                        + pparams.w_adv * float(tmp.loc[sym, "z_adv"])
                    )

                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                selected = [s for s, _ in ranked[:cap]]

                # slot sizing
                eq_now = equity_rows[-1][1]
                slot = eq_now / max(1, pparams.max_open)

                for sym in selected:
                    df = price_map[sym]
                    sig = sig_map[sym]
                    _, bp = best_cfg_map[sym]

                    if nxt not in df.index or t not in df.index or t not in sig.index:
                        continue

                    entry_px = float(df.loc[nxt, "Open"]) * (1 + slip)

                    swing_low = float(df["Low"].loc[:t].tail(bp.swing_lookback).min())
                    stop_swing = swing_low * (1 - bp.swing_buffer)
                    stop_atr = float(sig.loc[t, "d_ema20"] - (bp.atr_stop_mult * sig.loc[t, "atr14"]))
                    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

                    R = entry_px - stop_px
                    if not np.isfinite(R) or R <= 0:
                        continue

                    notional = min(slot, cash)
                    if notional <= 0:
                        continue
                    fee_cost = notional * fee
                    shares = (notional - fee_cost) / entry_px
                    if shares <= 0:
                        continue

                    cash -= notional
                    positions[sym] = _Pos(
                        entry_px=entry_px,
                        stop_px=stop_px,
                        R=R,
                        shares=shares,
                        orig_shares=shares,
                    )
                    trades.append((nxt, sym, "ENTRY", entry_px, shares))

    eqdf = pd.DataFrame(equity_rows, columns=["Date", "Equity", "Cash", "Npos"]).set_index("Date")
    trdf = pd.DataFrame(trades, columns=["Date", "Ticker", "Type", "Px", "Shares"])

    eqdf.to_csv(outdir / "equity_curve.csv")
    trdf.to_csv(outdir / "trades.csv", index=False)

    plot_equity(eqdf[["Equity"]], outdir / "equity.png", title="Portfolio Equity")

    return {"equity_curve": eqdf, "trades": trdf}

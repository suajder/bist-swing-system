from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signals import SignalEngine, SignalParams
from .live_events import resolve_intrabar_exit


@dataclass(frozen=True)
class BacktestParams:
    fee_bps: float = 8.0
    slippage_bps: float = 5.0

    use_atr_stop: bool = True
    atr_stop_mult: float = 0.2
    swing_lookback: int = 10
    swing_buffer: float = 0.01

    tp1_R: float = 2.0
    tp2_R: float = 3.0


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _equity_stats(eq: pd.Series, ann_factor: int = 252) -> Dict[str, float]:
    eq = eq.dropna()
    if len(eq) < 3:
        return {"cagr": np.nan, "max_dd": np.nan, "sharpe": np.nan}

    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = dd.min()

    mu = rets.mean() * ann_factor
    sd = rets.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = mu / sd if sd > 0 else np.nan

    return {"cagr": float(cagr), "max_dd": float(max_dd), "sharpe": float(sharpe)}


def aggregate_trades_from_legs(legs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-leg exits into per-trade summary rows.
    """
    if legs.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "ticker",
                "entry_date",
                "exit_date",
                "legs",
                "gross_pnl",
                "net_pnl",
                "avg_exit_px",
            ]
        )

    agg = (
        legs.groupby(["trade_id", "ticker", "entry_date"], as_index=False)
        .agg(
            exit_date=("date", "max"),
            legs=("reason", "count"),
            gross_pnl=("pnl", "sum"),
            avg_exit_px=("px", "mean"),
        )
        .rename(columns={"gross_pnl": "net_pnl"})
    )
    return agg


def run_backtest(
    df: pd.DataFrame,
    sp: SignalParams,
    bp: BacktestParams,
    initial_cash: float = 1_000_000.0,
) -> Dict[str, object]:
    """
    Backtest policy:
    - Entries: decision today -> fill next open (slippage applied)
    - Exits: evaluated on next day OHLC (next-open fill policy for WeeklyExit)
    - Stop/TP ordering inside a bar uses the SAME resolver as live:
      closest-to-open first, tie-break STOP (conservative)
    """
    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)

    se = SignalEngine()
    sig = se.build(df, sp)


    fee = bp.fee_bps / 10_000.0
    slip = bp.slippage_bps / 10_000.0

    cash = float(initial_cash)
    shares = 0.0

    entry_px = np.nan
    entry_date = None
    stop_px = np.nan
    R = np.nan
    tp1_done = False
    tp2_done = False
    trade_id = 0
    orig_shares = 0.0

    legs = []
    equity = []

    idx = list(df.index)
    for i in range(len(idx) - 1):  # need next bar
        t = idx[i]
        nxt = idx[i + 1]

        # mark-to-market at close
        close_t = float(df.loc[t, "Close"])
        eq_t = cash + shares * close_t
        equity.append((t, eq_t, cash, shares))

        # manage open position using NEXT day's OHLC
        if shares > 0:
            o2 = float(df.loc[nxt, "Open"])
            h2 = float(df.loc[nxt, "High"])
            l2 = float(df.loc[nxt, "Low"])

            # Weekly exit regime (decision today -> fill next open)
            if bool(sig.loc[t, "w_exit_regime"]):
                px = o2 * (1 - slip)
                pnl = (px - entry_px) * shares
                fee_cost = shares * px * fee
                cash += shares * px - fee_cost
                legs.append(
                    dict(
                        trade_id=trade_id,
                        ticker="TICKER",
                        entry_date=entry_date,
                        date=nxt,
                        entry_px=entry_px,
                        px=px,
                        shares=shares,
                        pnl=pnl - fee_cost,
                        reason="WeeklyExit",
                    )
                )
                shares = 0.0
                continue

            # Unified intrabar resolution (same as live):
            # Open'a en yakın seviye önce, tie-break STOP
            tp1 = entry_px + bp.tp1_R * R
            tp2 = entry_px + bp.tp2_R * R

            evt, lvl, _extra = resolve_intrabar_exit(
                o2=o2,
                h2=h2,
                l2=l2,
                stop_lvl=float(stop_px),
                tp1_lvl=float(tp1),
                tp2_lvl=float(tp2),
                tp1_done=bool(tp1_done),
                tp2_done=bool(tp2_done),
            )

            if evt is None:
                continue

            if evt == "STOP":
                px = float(lvl) * (1 - slip)
                pnl = (px - entry_px) * shares
                fee_cost = shares * px * fee
                cash += shares * px - fee_cost
                legs.append(
                    dict(
                        trade_id=trade_id,
                        ticker="TICKER",
                        entry_date=entry_date,
                        date=nxt,
                        entry_px=entry_px,
                        px=px,
                        shares=shares,
                        pnl=pnl - fee_cost,
                        reason="Stop",
                    )
                )
                shares = 0.0
                continue

            if evt == "TP1":
                qty = min(orig_shares / 3.0, shares)
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    pnl = (px - entry_px) * qty
                    fee_cost = qty * px * fee
                    cash += qty * px - fee_cost
                    shares -= qty
                    tp1_done = True
                    # Live ile tutarlılık: TP1 sonrası stop = entry (breakeven)
                    stop_px = entry_px
                    legs.append(
                        dict(
                            trade_id=trade_id,
                            ticker="TICKER",
                            entry_date=entry_date,
                            date=nxt,
                            entry_px=entry_px,
                            px=px,
                            shares=qty,
                            pnl=pnl - fee_cost,
                            reason="TP1",
                        )
                    )
                continue

            if evt == "TP2":
                qty = min(orig_shares / 3.0, shares)
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    pnl = (px - entry_px) * qty
                    fee_cost = qty * px * fee
                    cash += qty * px - fee_cost
                    shares -= qty
                    tp2_done = True
                    legs.append(
                        dict(
                            trade_id=trade_id,
                            ticker="TICKER",
                            entry_date=entry_date,
                            date=nxt,
                            entry_px=entry_px,
                            px=px,
                            shares=qty,
                            pnl=pnl - fee_cost,
                            reason="TP2",
                        )
                    )
                continue

            # if fully depleted by partials, close trade id
            if shares <= 1e-12:
                shares = 0.0

        # entry (decision today -> fill next open)
        if shares <= 0 and bool(sig.loc[t, "entry_signal"]):
            o2 = float(df.loc[nxt, "Open"])
            entry_px = o2 * (1 + slip)

            # stop calc uses info up to t
            swing_low = float(df["Low"].loc[:t].tail(bp.swing_lookback).min())
            stop_swing = swing_low * (1 - bp.swing_buffer)
            stop_atr = float(sig.loc[t, "d_ema20"] - (bp.atr_stop_mult * sig.loc[t, "atr14"]))
            stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

            R = entry_px - stop_px
            if not np.isfinite(R) or R <= 0:
                continue

            trade_id += 1
            entry_date = nxt
            tp1_done = False
            tp2_done = False

            # all-in for model score simplicity
            notional = cash
            fee_cost = notional * fee
            notional_net = notional - fee_cost
            if notional_net <= 0:
                continue
            shares = notional_net / entry_px
            orig_shares = shares
            cash = 0.0

    # final mtm
    t = idx[-1]
    eq_t = cash + shares * float(df.loc[t, "Close"])
    equity.append((t, eq_t, cash, shares))

    eqdf = pd.DataFrame(equity, columns=["Date", "Equity", "Cash", "Shares"]).set_index("Date")
    legs_df = pd.DataFrame(legs)
    metrics = _equity_stats(eqdf["Equity"])
    return {"legs": legs_df, "equity": eqdf, "metrics": metrics}

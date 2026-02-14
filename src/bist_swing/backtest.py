from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signals import SignalEngine, SignalParams


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
    Minimal aggregation: group by trade_id.
    """
    if legs.empty:
        return legs.copy()

    g = legs.groupby("trade_id", as_index=False)
    out = g.agg(
        ticker=("ticker", "first"),
        entry_date=("entry_date", "first"),
        exit_date=("date", "max"),
        entry_px=("entry_px", "first"),
        avg_exit_px=("px", "mean"),
        shares=("shares", "sum"),
        gross_pnl=("pnl", "sum"),
        reason=("reason", lambda x: ",".join(sorted(set(x)))),
    )
    return out


class Backtester:
    """
    Single-ticker backtest, long-only.
    - Signal: signals["entry_signal"] at day t triggers entry at next open
    - Stop/TP/WeeklyExit evaluated on next day OHLC (next-open fill policy)
    """

    def run(
        self,
        *,
        daily: pd.DataFrame,
        se: SignalEngine,
        sp: SignalParams,
        bp: BacktestParams,
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_equity: float = 1.0,
    ) -> Dict[str, object]:
        df = daily.copy()
        df = df.sort_index()
        if start:
            df = df.loc[pd.to_datetime(start) :]
        if end:
            df = df.loc[: pd.to_datetime(end)]
        if len(df) < 60:
            return {"legs": pd.DataFrame(), "equity": pd.DataFrame(), "metrics": {}}

        sig = se.build(df, sp)

        slip = bp.slippage_bps / 10000.0
        fee = bp.fee_bps / 10000.0

        cash = float(initial_equity)
        shares = 0.0
        entry_px = np.nan
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

                # Stop
                if l2 <= stop_px:
                    px = stop_px * (1 - slip)
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

                # TPs (partial 1/3 each)
                tp1 = entry_px + bp.tp1_R * R
                tp2 = entry_px + bp.tp2_R * R

                if (not tp1_done) and (h2 >= tp1):
                    qty = min(orig_shares / 3.0, shares)
                    px = tp1 * (1 - slip)
                    pnl = (px - entry_px) * qty
                    fee_cost = qty * px * fee
                    cash += qty * px - fee_cost
                    shares -= qty
                    tp1_done = True
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

                if (not tp2_done) and (h2 >= tp2):
                    qty = min(orig_shares / 3.0, shares)
                    px = tp2 * (1 - slip)
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

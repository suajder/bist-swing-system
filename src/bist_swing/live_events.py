from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .backtest import BacktestParams
from .position_state import Position


@dataclass(frozen=True)
class Event:
    ticker: str
    event: str  # ENTRY_PLANNED / STOP / TP1 / TP2 / WEEKLY_EXIT
    date: pd.Timestamp
    px: float
    info: str = ""


def compute_entry_levels(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
) -> Tuple[float, float, float, pd.Timestamp]:
    """
    Entry assumed at next bar Open (model).
    Stop from swing-low or ATR-based.
    Returns: (entry_px, stop_px, R, next_bar_date)
    """
    idx = df.index
    i = idx.get_loc(asof)
    if i + 1 >= len(idx):
        raise RuntimeError("asof has no next bar for entry")
    nxt = idx[i + 1]

    slip = bp.slippage_bps / 10000.0
    entry_px = float(df.loc[nxt, "Open"]) * (1 + slip)

    swing_low = float(df["Low"].loc[:asof].tail(bp.swing_lookback).min())
    stop_swing = swing_low * (1 - bp.swing_buffer)

    atr14 = float(sig.loc[asof, "atr14"])
    ema20 = float(sig.loc[asof, "d_ema20"])
    stop_atr = ema20 - (bp.atr_stop_mult * atr14)

    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

    r = entry_px - stop_px
    if r <= 0:
        raise RuntimeError("Invalid R (entry <= stop)")
    return entry_px, stop_px, r, nxt


def _hit(open_px: float, high_px: float, low_px: float, level: float) -> bool:
    return low_px <= level <= high_px


def _dist(open_px: float, level: float) -> float:
    return abs(level - open_px)


def evaluate_position_events(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    bp: BacktestParams,
    pos: Position,
    asof: pd.Timestamp,
) -> List[Event]:
    """
    Intrabar (OHLC) event resolution:

    - Decision time: asof close (we evaluate regime & levels)
    - Execution/trigger time: next bar OHLC (nxt)

    Priority rule:
    - Price starts at Open
    - If multiple levels are hit in the same bar, the level closest to Open is assumed hit first.
    - Ties -> STOP first (conservative)
    - If TP1 & TP2 both hit: we emit the first event and add a note that the other level was also seen.
    """
    out: List[Event] = []
    idx = df.index
    if asof not in idx:
        return out

    i = idx.get_loc(asof)
    if i + 1 >= len(idx):
        return out
    nxt = idx[i + 1]

    o2 = float(df.loc[nxt, "Open"])
    h2 = float(df.loc[nxt, "High"])
    l2 = float(df.loc[nxt, "Low"])
    slip = bp.slippage_bps / 10000.0

    # Weekly exit dominates (execute at next open)
    if bool(sig.loc[asof, "w_exit_regime"]):
        px = o2 * (1 - slip)
        out.append(Event(pos.ticker, "WEEKLY_EXIT", nxt, px, "Haftalık rejim çıkışı (trend bozuldu)"))
        return out

    stop_lvl = float(pos.stop_px)
    tp1_lvl = float(pos.entry_px + bp.tp1_R * pos.r)
    tp2_lvl = float(pos.entry_px + bp.tp2_R * pos.r)

    stop_hit = _hit(o2, h2, l2, stop_lvl)
    tp1_hit = (not pos.tp1_done) and _hit(o2, h2, l2, tp1_lvl)
    tp2_hit = (not pos.tp2_done) and _hit(o2, h2, l2, tp2_lvl)

    if not (stop_hit or tp1_hit or tp2_hit):
        return out

    candidates: List[tuple[str, float]] = []
    if stop_hit:
        candidates.append(("STOP", stop_lvl))
    if tp1_hit:
        candidates.append(("TP1", tp1_lvl))
    if tp2_hit:
        candidates.append(("TP2", tp2_lvl))

    # sort by distance to Open, tie-break STOP first
    candidates.sort(key=lambda x: (_dist(o2, x[1]), 0 if x[0] == "STOP" else 1))

    first_evt, first_lvl = candidates[0]

    # extra note: if TP1 & TP2 both hit in same bar, mention it
    extra = ""
    if tp1_hit and tp2_hit:
        if first_evt == "TP1":
            extra = f" (Aynı barda TP2 de görüldü: {tp2_lvl:.2f})"
        elif first_evt == "TP2":
            extra = f" (Aynı barda TP1 de görüldü: {tp1_lvl:.2f})"

    if first_evt == "STOP":
        px = first_lvl * (1 - slip)
        out.append(Event(pos.ticker, "STOP", nxt, px, f"Stop={first_lvl:.2f}"))
        return out

    if first_evt == "TP1":
        px = first_lvl * (1 - slip)
        out.append(Event(pos.ticker, "TP1", nxt, px, f"Hedef1={first_lvl:.2f}{extra}"))
        return out

    if first_evt == "TP2":
        px = first_lvl * (1 - slip)
        out.append(Event(pos.ticker, "TP2", nxt, px, f"Hedef2={first_lvl:.2f}{extra}"))
        return out

    return out

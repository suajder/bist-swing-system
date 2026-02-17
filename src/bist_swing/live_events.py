from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .backtest import BacktestParams

from .position_state import Position


@dataclass
class Event:
    ticker: str
    event: str  # ENTRY_PLANNED / STOP / TP1 / TP2 / WEEKLY_EXIT
    date: pd.Timestamp
    px: float
    note: str = ""


def compute_entry_and_stop(
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
) -> tuple[float, float, float, pd.Timestamp]:
    """
    Stop from swing-low or ATR-based.
    Returns: (entry_px, stop_px, R, next_bar_date)
    """
    if asof not in sig.index:
        raise KeyError("asof not in signal index")

    # Next bar for fill / exit checks
    idx = sig.index.get_loc(asof)
    if idx + 1 >= len(sig.index):
        raise RuntimeError("No next bar available for asof")
    nxt = sig.index[idx + 1]

    entry_px = float(sig.loc[nxt, "Open"])
    swing_low = float(sig.loc[asof, "d_swing_low"])
    ema20 = float(sig.loc[asof, "d_ema20"])
    atr14 = float(sig.loc[asof, "atr14"])

    stop_swing = swing_low * (1 - bp.swing_buffer)
    stop_atr = ema20 - (bp.atr_stop_mult * atr14)
    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

    r = entry_px - stop_px
    if not np.isfinite(r) or r <= 0:
        raise RuntimeError("Invalid R (entry <= stop)")

    return entry_px, stop_px, r, nxt


def _hit(open_px: float, high_px: float, low_px: float, level: float) -> bool:
    # Whether the level is inside the bar range [low, high]
    return low_px <= level <= high_px


def _dist(open_px: float, level: float) -> float:
    return abs(level - open_px)


def resolve_intrabar_exit(
    o2: float,
    h2: float,
    l2: float,
    stop_lvl: float,
    tp1_lvl: float,
    tp2_lvl: float,
    tp1_done: bool,
    tp2_done: bool,
) -> tuple[str | None, float | None, str]:
    """
    Decide which exit event happens first inside a bar, using the same rule as live:
    - Candidate levels are considered 'hit' if level is within [low, high]
    - Choose event whose level is closest to Open
    - Tie-break: STOP first (conservative)
    - If TP1 & TP2 both hit in the same bar, include a note
    Returns: (event, level, extra_note)
    """
    stop_hit = _hit(o2, h2, l2, stop_lvl)
    tp1_hit = (not tp1_done) and _hit(o2, h2, l2, tp1_lvl)
    tp2_hit = (not tp2_done) and _hit(o2, h2, l2, tp2_lvl)

    if not (stop_hit or tp1_hit or tp2_hit):
        return None, None, ""

    candidates: list[tuple[str, float]] = []
    if stop_hit:
        candidates.append(("STOP", float(stop_lvl)))
    if tp1_hit:
        candidates.append(("TP1", float(tp1_lvl)))
    if tp2_hit:
        candidates.append(("TP2", float(tp2_lvl)))

    # sort by distance to Open, tie-break STOP first
    candidates.sort(key=lambda x: (_dist(o2, x[1]), 0 if x[0] == "STOP" else 1))

    first_evt, first_lvl = candidates[0]

    extra = ""
    if tp1_hit and tp2_hit:
        if first_evt == "TP1":
            extra = f" (Aynı barda TP2 de görüldü: {tp2_lvl:.2f})"
        elif first_evt == "TP2":
            extra = f" (Aynı barda TP1 de görüldü: {tp1_lvl:.2f})"

    return first_evt, float(first_lvl), extra


def evaluate_position_events(
    pos: Position,
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
) -> List[Event]:
    """
    Evaluate STOP/TP1/TP2/WEEKLY_EXIT for an open position, using next bar OHLC.
    """
    out: List[Event] = []

    if asof not in sig.index:
        return out

    idx = sig.index.get_loc(asof)
    if idx + 1 >= len(sig.index):
        return out
    nxt = sig.index[idx + 1]

    o2 = float(sig.loc[nxt, "Open"])
    h2 = float(sig.loc[nxt, "High"])
    l2 = float(sig.loc[nxt, "Low"])

    # Weekly exit dominates (execute at next open)
    if bool(sig.loc[asof, "w_exit_regime"]):
        px = o2
        out.append(Event(pos.ticker, "WEEKLY_EXIT", nxt, px, "Haftalık rejim çıkışı (trend bozuldu)"))
        return out

    stop_lvl = float(pos.stop_px)
    tp1_lvl = float(pos.entry_px + bp.tp1_R * pos.r)
    tp2_lvl = float(pos.entry_px + bp.tp2_R * pos.r)

    first_evt, first_lvl, extra = resolve_intrabar_exit(
        o2=o2,
        h2=h2,
        l2=l2,
        stop_lvl=stop_lvl,
        tp1_lvl=tp1_lvl,
        tp2_lvl=tp2_lvl,
        tp1_done=pos.tp1_done,
        tp2_done=pos.tp2_done,
    )

    if first_evt is None:
        return out

    # keep original semantics for emitted events
    if first_evt == "STOP":
        px = float(first_lvl)
        out.append(Event(pos.ticker, "STOP", nxt, px, f"Stop={first_lvl:.2f}"))
        return out

    if first_evt == "TP1":
        px = float(first_lvl)
        out.append(Event(pos.ticker, "TP1", nxt, px, f"Hedef1={first_lvl:.2f}{extra}"))
        return out

    if first_evt == "TP2":
        px = float(first_lvl)
        out.append(Event(pos.ticker, "TP2", nxt, px, f"Hedef2={first_lvl:.2f}{extra}"))
        return out

    return out

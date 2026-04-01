from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .backtest import BacktestParams

from .position_state import Position
from bist_swing.risk_engine import update_equity, RiskState


@dataclass
class Event:
    ticker: str
    event: str  # STOP / TP1 / TP2 / WEEKLY_EXIT
    date: pd.Timestamp
    px: float
    note: str = ""


# ---------------- ENTRY ----------------

def compute_entry_and_stop(
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
) -> tuple[float, float, float, pd.Timestamp]:

    if asof not in sig.index:
        raise KeyError("asof not in signal index")

    idx = sig.index.get_loc(asof)
    if idx + 1 >= len(sig.index):
        raise RuntimeError("No next bar available")

    nxt = sig.index[idx + 1]

    entry_px = float(sig.loc[nxt, "Open"])
    swing_low = float(sig.loc[asof, "d_swing_low"])
    ema20 = float(sig.loc[asof, "d_ema20"])
    atr14 = float(sig.loc[asof, "atr14"])

    stop_swing = swing_low * (1 - bp.swing_buffer)
    stop_atr = ema20 - (bp.atr_stop_mult * atr14)

    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

    r = entry_px - stop_px
    if r <= 0:
        raise RuntimeError("Invalid R")

    return entry_px, stop_px, r, nxt


# ---------------- CORE LOGIC ----------------

def _hit(h, l, level):
    return l <= level <= h


def _dist(o, level):
    return abs(level - o)


def resolve_intrabar_exit(
    o2, h2, l2,
    stop_lvl, tp1_lvl, tp2_lvl,
    tp1_done, tp2_done
):

    stop_hit = _hit(h2, l2, stop_lvl)
    tp1_hit = (not tp1_done) and _hit(h2, l2, tp1_lvl)
    tp2_hit = (not tp2_done) and _hit(h2, l2, tp2_lvl)

    if not (stop_hit or tp1_hit or tp2_hit):
        return None, None, ""

    candidates = []
    if stop_hit:
        candidates.append(("STOP", stop_lvl))
    if tp1_hit:
        candidates.append(("TP1", tp1_lvl))
    if tp2_hit:
        candidates.append(("TP2", tp2_lvl))

    candidates.sort(key=lambda x: (_dist(o2, x[1]), 0 if x[0] == "STOP" else 1))

    evt, lvl = candidates[0]

    extra = ""
    if tp1_hit and tp2_hit:
        extra = " (aynı barda diğer TP de görüldü)"

    return evt, lvl, extra


# ---------------- MAIN EVENT ENGINE ----------------

def evaluate_position_events(
    pos: Position,
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
    risk_state: RiskState,   # 🔥 EKLENDİ
) -> List[Event]:

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

    # ---------------- WEEKLY EXIT ----------------
    if bool(sig.loc[asof, "w_exit_regime"]):
        px = o2

        pnl = (px - pos.entry_px) * pos.qty
        update_equity(risk_state, pnl)   # 🔥 KRİTİK

        out.append(Event(
            pos.ticker,
            "WEEKLY_EXIT",
            nxt,
            px,
            "Haftalık exit"
        ))
        return out

    stop_lvl = float(pos.stop_px)
    tp1_lvl = float(pos.entry_px + bp.tp1_R * pos.r)
    tp2_lvl = float(pos.entry_px + bp.tp2_R * pos.r)

    evt, lvl, extra = resolve_intrabar_exit(
        o2, h2, l2,
        stop_lvl, tp1_lvl, tp2_lvl,
        pos.tp1_done, pos.tp2_done
    )

    if evt is None:
        return out

    # ---------------- STOP ----------------
    if evt == "STOP":
        pnl = (lvl - pos.entry_px) * pos.qty
        update_equity(risk_state, pnl)   # 🔥

        out.append(Event(pos.ticker, "STOP", nxt, lvl, f"Stop={lvl:.2f}"))
        return out

    # ---------------- TP1 ----------------
    if evt == "TP1":
        qty_closed = pos.qty * 0.5   # partial exit
        pnl = (lvl - pos.entry_px) * qty_closed
        update_equity(risk_state, pnl)   # 🔥

        out.append(Event(pos.ticker, "TP1", nxt, lvl, f"TP1={lvl:.2f}{extra}"))
        return out

    # ---------------- TP2 ----------------
    if evt == "TP2":
        pnl = (lvl - pos.entry_px) * pos.qty
        update_equity(risk_state, pnl)   # 🔥

        out.append(Event(pos.ticker, "TP2", nxt, lvl, f"TP2={lvl:.2f}{extra}"))
        return out

    return out

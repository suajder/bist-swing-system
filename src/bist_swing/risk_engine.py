from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

STATE_FILE = Path("risk_state.json")


# ================================
# DATA STRUCTURE
# ================================

@dataclass
class RiskState:
    capital: float
    peak_equity: float
    current_equity: float
    current_dd: float
    base_risk_pct: float


# ================================
# LOAD / SAVE
# ================================

def load_risk_state(initial_capital: float, base_risk_pct: float) -> RiskState:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return RiskState(**data)

    state = RiskState(
        capital=initial_capital,
        peak_equity=initial_capital,
        current_equity=initial_capital,
        current_dd=0.0,
        base_risk_pct=base_risk_pct,
    )
    save_risk_state(state)
    return state


def save_risk_state(state: RiskState):
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2))


# ================================
# EQUITY UPDATE
# ================================

def update_equity(state: RiskState, pnl: float):
    state.current_equity += pnl

    if state.current_equity > state.peak_equity:
        state.peak_equity = state.current_equity

    state.current_dd = (
        (state.current_equity - state.peak_equity) / state.peak_equity
    )

    save_risk_state(state)

# ================================
# RISK CONTROL
# ================================

def risk_multiplier(dd: float) -> float:

    if dd > -0.05:
        return 1.0
    elif dd > -0.10:
        return 0.75
    elif dd > -0.15:
        return 0.50
    else:
        return 0.25


def compute_risk_pct(state: RiskState) -> float:
    return state.base_risk_pct * risk_multiplier(state.current_dd)


# ================================
# POSITION SIZE
# ================================

def calculate_position_size(state: RiskState, entry: float, stop: float) -> int:

    risk_pct = compute_risk_pct(state)
    risk_amount = state.current_equity * risk_pct

    per_share_risk = abs(entry - stop)

    if per_share_risk <= 0:
        return 0

    qty = risk_amount / per_share_risk
    return max(0, int(qty))


# ================================
# VALIDATION
# ================================

def validate_trade(entry: float, stop: float) -> bool:

    if entry <= 0 or stop <= 0:
        return False

    risk_pct = abs(entry - stop) / entry

    return risk_pct <= 0.06


# ================================
# PORTFOLIO RISK
# ================================

def compute_portfolio_risk(positions: dict) -> float:

    total = 0

    for p in positions.values():
        if p.get("is_open"):
            total += abs(p["entry_px"] - p["stop_px"]) * p["qty"]

    return total


# ================================
# DD
# ================================

def compute_drawdown(state: RiskState) -> float:
    return state.current_dd

# ================================
# KILL SWITCH (REAL + FLOATING)
# ================================

def kill_switch_triggered(
    state: RiskState,
    floating_dd: float,
    realized_limit: float = -0.20,
    floating_limit: float = -0.10,
) -> bool:
    """
    realized_limit: kapanmış zarar limiti
    floating_limit: açık pozisyon dahil max risk
    """

    if state.current_dd <= realized_limit:
        return True

    if floating_dd <= floating_limit:
        return True

    return False

# ================================
# UNREALIZED PnL
# ================================

def compute_unrealized_pnl(positions: dict, price_map: dict, asof) -> float:

    pnl = 0.0

    for sym, p in positions.items():
        if not p.get("is_open"):
            continue

        if sym not in price_map:
            continue

        try:
            df = price_map[sym]
            if asof not in df.index:
                continue

            current_px = float(df.loc[asof, "Close"])
            entry = float(p["entry_px"])
            qty = float(p["qty"])

            pnl += (current_px - entry) * qty

        except Exception:
            continue

    return pnl


# ================================
# TOTAL EQUITY (REAL + FLOATING)
# ================================

def compute_total_equity(state: RiskState, unrealized_pnl: float) -> float:
    return state.current_equity + unrealized_pnl


# ================================
# FLOATING DD
# ================================

def compute_floating_dd(state: RiskState, total_equity: float) -> float:

    if state.peak_equity <= 0:
        return 0.0

    return (total_equity - state.peak_equity) / state.peak_equity
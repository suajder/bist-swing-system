from dataclasses import dataclass

@dataclass
class RiskState:
    current_dd: float
    base_risk_pct: float


def risk_multiplier(dd):

    if dd > -5:
        return 1.0

    elif dd > -10:
        return 0.75

    elif dd > -15:
        return 0.50

    else:
        return 0.25


def compute_risk_pct(state: RiskState):

    mult = risk_multiplier(state.current_dd)

    return state.base_risk_pct * mult
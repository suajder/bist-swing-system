import math

def compute_position_size(
    equity,
    risk_pct,
    entry_price,
    stop_price
):

    risk_capital = equity * risk_pct

    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        return 0

    shares = risk_capital / risk_per_share

    return math.floor(shares)
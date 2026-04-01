import csv
from pathlib import Path
import pandas as pd

# ================================
# PATHS
# ================================

TRADES_LOG = Path("trades.csv")
EQUITY_LOG = Path("equity_log.csv")


# ================================
# TRADE LOGGER
# ================================

def log_trade(row: dict):
    exists = TRADES_LOG.exists()

    with open(TRADES_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ================================
# EQUITY LOGGER
# ================================

def log_equity(date, equity, realized_dd, floating_dd, open_positions, risk_pct):

    row = {
        "date": date,
        "equity": equity,
        "realized_dd": realized_dd,
        "floating_dd": floating_dd,
        "open_positions": open_positions,
        "risk_pct": risk_pct,
    }

    df = pd.DataFrame([row])

    if EQUITY_LOG.exists():
        df.to_csv(EQUITY_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(EQUITY_LOG, index=False)
import csv
import logging
from pathlib import Path

import pandas as pd

def setup_logger(name: str = "bist_swing") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "system.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
    return logger
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
from pathlib import Path
import sys
import pandas as pd
import yfinance as yf

from bist_swing.portfolio import (
    portfolio_backtest_pro,
    PortfolioParams,
)
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
START_DATE = "2021-01-01"
END_DATE = None

UNIVERSE_FILE = Path("configs/universe.txt")
OUTDIR = Path("out/portfolio")

# Portfolio parameters (50K growth profile)
PPARAMS = PortfolioParams(
    max_open=5,
    initial_equity=50_000.0,
    adv20_min=30_000_000,
    risk_pct=0.02,
    daily_stop_R=3.0,
    weekly_stop_R=6.0,
)

# -------------------------------------------------
# Load universe
# -------------------------------------------------
if not UNIVERSE_FILE.exists():
    print("Universe file not found:", UNIVERSE_FILE)
    sys.exit(1)

tickers = [t.strip() for t in UNIVERSE_FILE.read_text().splitlines() if t.strip()]
tickers = [t if t.endswith(".IS") else t + ".IS" for t in tickers]

print(f"Downloading {len(tickers)} tickers...")

price_map = {}
failed = []

for t in tickers:
    try:
        df = yf.download(t, start=START_DATE, progress=False)
        if df is None or df.empty:
            failed.append(t)
            continue

        df = df.rename(columns=str.title)
        df["ADV20"] = df["Close"] * df["Volume"].rolling(20).mean()
        price_map[t] = df

    except Exception:
        failed.append(t)

print(f"Downloaded OK: {len(price_map)}/{len(tickers)}")

if failed:
    print("Failed tickers:", failed)

if not price_map:
    print("No data downloaded.")
    sys.exit(1)

# -------------------------------------------------
# Build per-ticker configs
# -------------------------------------------------
se = SignalEngine()

best_cfg_map = {}
model_score_map = {}

for t in price_map.keys():
    sp = SignalParams()
    bp = BacktestParams()
    best_cfg_map[t] = (sp, bp)
    model_score_map[t] = 0.0  # baseline model weight

# -------------------------------------------------
# Run portfolio backtest
# -------------------------------------------------
res = portfolio_backtest_pro(
    se=se,
    tickers=list(price_map.keys()),
    price_map=price_map,
    best_cfg_map=best_cfg_map,
    model_score_map=model_score_map,
    test_start=START_DATE,
    test_end=END_DATE,
    pparams=PPARAMS,
    outdir=OUTDIR,
)

eq = res["equity_curve"]
tr = res["trades"]

print("\nDone.")
print("Equity rows:", eq.shape)
print("Trades rows:", tr.shape)
print("Outputs in:", OUTDIR.resolve())

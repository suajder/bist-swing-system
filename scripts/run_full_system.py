import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from bist_swing.data import load_price_data
from bist_swing.signals import SignalEngine, SignalParams

OUT = Path("out/full")
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
RISK_PCT = 0.01
MAX_HOLD_DAYS = 10


def simulate_trade(df, entry_idx, entry, stop, target):
    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD_DAYS, len(df))):
        high = df.iloc[i]["High"]
        low = df.iloc[i]["Low"]

        if low <= stop:
            return -1
        if high >= target:
            return 2

    return 0


def run():

    # universe
    with open("configs/universe.txt") as f:
        tickers = [x.strip() for x in f]

    price_map = load_price_data(tickers)

    se = SignalEngine()
    sp = SignalParams()

    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []

    for sym in tickers:

        if sym not in price_map:
            continue

        df = price_map[sym]

        if len(df) < 200:
            continue

        sig = se.build(df, sp)

        for t in sig.index:

            if not sig.loc[t, "entry_signal"]:
                continue

            # 🔥 FILTER STACK
            if not df.loc[t, "trend_ok"]:
                continue
            if not df.loc[t, "breakout_ok"]:
                continue
            if not df.loc[t, "vol_spike"]:
                continue

            entry = df.loc[t, "Close"]
            atr = df.loc[t, "atr14"]

            if atr <= 0:
                continue

            stop = entry - 2 * atr
            target = entry + 2 * (entry - stop)

            try:
                entry_idx = df.index.get_loc(t)
            except:
                continue

            R = entry - stop
            if R <= 0:
                continue

            risk_amount = equity * RISK_PCT

            result_R = simulate_trade(df, entry_idx, entry, stop, target)

            pnl = result_R * risk_amount
            equity += pnl

            trades.append({
                "ticker": sym,
                "date": t,
                "R": result_R,
                "pnl": pnl,
                "equity": equity
            })

            equity_curve.append(equity)

    df_trades = pd.DataFrame(trades)

    if df_trades.empty:
        print("No trades")
        return

    # ===== METRICS =====
    winrate = (df_trades["R"] > 0).mean()
    avg_R = df_trades["R"].mean()
    total_return = (equity / INITIAL_CAPITAL - 1)

    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    max_dd = dd.min()

    print("\n=== SYSTEM PERFORMANCE ===")
    print(f"Trades: {len(df_trades)}")
    print(f"Winrate: {winrate:.2%}")
    print(f"Avg R: {avg_R:.2f}")
    print(f"Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # ===== SAVE =====
    df_trades.to_csv(OUT / "trades.csv", index=False)
    equity_series.to_csv(OUT / "equity.csv", index=False)

    # ===== PLOT =====
    plt.figure()
    plt.plot(equity_series)
    plt.title("Equity Curve")
    plt.grid()
    plt.savefig(OUT / "equity_curve.png")

    print("\nSaved to /out/full")


if __name__ == "__main__":
    run()
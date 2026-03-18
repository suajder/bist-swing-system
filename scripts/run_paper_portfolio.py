import pandas as pd
from pathlib import Path
import random

OUT = Path("out/paper")
OUT.mkdir(parents=True, exist_ok=True)


INITIAL_CAPITAL = 100_000
RISK_PCT = 0.01
MAX_POSITIONS = 5


def run():

    df = pd.read_csv("out/live/live_signals.csv")

    if df.empty:
        print("No signals - nothing to simulate")

        # boş output üret (pipeline kırılmasın)
        pd.DataFrame().to_csv(OUT / "paper_trades.csv", index=False)
        return

    equity = INITIAL_CAPITAL

    trades = []
    equity_curve = []

    open_positions = 0

    for _, row in df.iterrows():

        if open_positions >= MAX_POSITIONS:
            break

        entry = row["entry"]
        stop = row["stop"]

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            continue

        risk_amount = equity * RISK_PCT
        shares = risk_amount / risk_per_share

        # 🎯 SIMULATION (2R win / 1R loss)
        outcome = random.choice(["win", "loss"])

        if outcome == "win":
            pnl = risk_amount * 2
        else:
            pnl = -risk_amount

        equity += pnl

        trades.append({
            "ticker": row["ticker"],
            "entry": entry,
            "stop": stop,
            "shares": shares,
            "risk": risk_amount,
            "outcome": outcome,
            "pnl": pnl,
            "equity_after": equity,
        })

        equity_curve.append(equity)
        open_positions += 1

    out = pd.DataFrame(trades)
    out.to_csv(OUT / "paper_trades.csv", index=False)

    # 📊 PERFORMANCE
    total_return = (equity / INITIAL_CAPITAL - 1) * 100

    print("\n=== PAPER TRADES ===")
    print(out)

    print("\n=== PERFORMANCE ===")
    print(f"Final Equity: {equity:.2f}")
    print(f"Return: {total_return:.2f}%")
    print(f"Trades: {len(out)}")

    if len(out) > 0:
        winrate = (out["outcome"] == "win").mean() * 100
        print(f"Winrate: {winrate:.2f}%")

    # equity curve kaydet
    pd.DataFrame({"equity": equity_curve}).to_csv(
        OUT / "equity_curve.csv", index=False
    )


if __name__ == "__main__":
    run()
import pandas as pd
from pathlib import Path

OUT = Path("out/paper")
OUT.mkdir(parents=True, exist_ok=True)


def run():

    df = pd.read_csv("out/live/live_signals.csv")

    if df.empty:
        print("No signals")
        return

    equity = 100000
    risk_pct = 0.02

    trades = []

    for _, row in df.iterrows():

        entry = row["entry"]
        stop = row["stop"]

        risk = entry - stop
        if risk <= 0:
            continue

        risk_amount = equity * risk_pct

        shares = risk_amount / risk

        # simulate TP = 2R
        target = entry + 2 * risk

        trades.append({
            "ticker": row["ticker"],
            "entry": entry,
            "stop": stop,
            "target": target,
            "shares": shares,
        })

    out = pd.DataFrame(trades)
    out.to_csv(OUT / "paper_trades.csv", index=False)

    print("\n=== PAPER TRADES ===")
    print(out.head())


if __name__ == "__main__":
    run()
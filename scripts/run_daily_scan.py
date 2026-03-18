import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from bist_swing.data import get_price_data
from bist_swing.signals import SignalEngine
from bist_swing.utils import safe_float
from bist_swing.signals import SignalParams

OUT = Path("out/live")
OUT.mkdir(parents=True, exist_ok=True)


def run():

    with open("configs/universe.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"\nDownloading {len(tickers)} tickers...")

    price_map = get_price_data(tickers)

    se = SignalEngine()
    sp = SignalParams()

    signals = []

    for sym in tickers:

        if sym not in price_map:
            continue

        df = price_map[sym]

        if df.empty:
            continue

        sig = se.build(df, sp)

        if sig.empty:
            continue

        t = df.index[-1]

        if t not in sig.index:
            continue

        if not bool(sig.loc[t, "entry_signal"]):
            continue

        adv20 = safe_float(df.loc[t, "ADV20"], 0)

        if adv20 < 10_000_000:
            continue

        inst_ok = bool(df.loc[t, "inst_mom_ok"]) if "inst_mom_ok" in df.columns else True
        liq_ok = bool(df.loc[t, "liq_shock"]) if "liq_shock" in df.columns else True

        if not (inst_ok and liq_ok):
            continue

        entry = float(df.loc[t, "Close"])

        if "atr14" not in df.columns:
            continue

        atr = float(df.loc[t, "atr14"])

        if atr <= 0:
            continue

        stop = entry - 2 * atr
        risk = entry - stop

        if risk <= 0:
            continue

        signals.append({
            "date": t,
            "ticker": sym,
            "entry": entry,
            "stop": stop,
            "risk": risk,
        })

    out = pd.DataFrame(signals)

    if out.empty:
        print("\nNo signals today.")

    out = pd.DataFrame(columns=["date", "ticker", "entry", "stop", "risk"])
    out.to_csv(OUT / "live_signals.csv", index=False)

    return

if __name__ == "__main__":
    run()
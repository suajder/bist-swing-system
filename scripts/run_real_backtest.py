import pandas as pd
from pathlib import Path

from bist_swing.data import load_price_data

OUT = Path("out/backtest")
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
RISK_PCT = 0.01
MAX_HOLD_DAYS = 10

USE_FAKE_SIGNALS = False  # ⚠️ TEST MODU


def simulate_trade(df, entry_idx, entry_price, stop, target):
    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLD_DAYS, len(df))):

        high = df.iloc[i]["High"]
        low = df.iloc[i]["Low"]

        if low <= stop:
            return -1

        if high >= target:
            return +2

    return 0


def run():

    # =========================
    # SIGNAL SOURCE
    # =========================
    if USE_FAKE_SIGNALS:
        print("⚠️ TEST MODE: Fake signals oluşturuluyor")

        tickers = ["ASELS.IS", "THYAO.IS", "BIMAS.IS"]
        price_map = load_price_data(tickers)

        signals = []

        for sym in tickers:
            df = price_map[sym]

            if len(df) < 50:
                continue

            t = df.index[-20]

            entry = df.loc[t, "Close"]
            stop = entry * 0.95

            signals.append({
                "date": t,
                "ticker": sym,
                "entry": entry,
                "stop": stop,
            })

        signals = pd.DataFrame(signals)

    else:
        signals = pd.read_csv("out/live/live_signals.csv")

    print("Total signals loaded:", len(signals))

    if signals.empty:
        print("No signals → backtest skipped")
        return

    tickers = signals["ticker"].unique().tolist()
    price_map = load_price_data(tickers)

    equity = INITIAL_CAPITAL
    trades = []
    equity_curve = []

    # =========================
    # DEBUG: veri kontrolü
    # =========================
    for sym in tickers:
        df = price_map[sym]
        print(f"{sym} rows:", len(df))

    # =========================
    # TRADE LOOP
    # =========================
    for _, row in signals.iterrows():

        sym = row["ticker"]
        df = price_map[sym]

        entry_price = row["entry"]
        stop = row["stop"]
        target = entry_price + 2 * (entry_price - stop)

        try:
            entry_idx = df.index.get_loc(pd.to_datetime(row["date"]))
        except Exception as e:
            print(f"Index bulunamadı: {sym}", e)
            continue

        # =========================
        # 🔥 EDGE FILTER BLOĞU (KRİTİK)
        # =========================

        if entry_idx < 200:
            continue  # yeterli veri yok

        row_df = df.iloc[entry_idx]

        # 1. EMA hesapla
        ema50 = df["Close"].ewm(span=50).mean().iloc[entry_idx]
        ema200 = df["Close"].ewm(span=200).mean().iloc[entry_idx]

        trend_ok = (
            row_df["Close"] > ema50 and
            ema50 > ema200
        )

        # 2. Breakout (20 gün)
        high_20 = df["High"].rolling(20).max().iloc[entry_idx]
        breakout_ok = row_df["Close"] >= high_20 * 0.98 #1

        # 3. Volume spike
        vol_mean = df["Volume"].rolling(20).mean().iloc[entry_idx]
        vol_ok = row_df["Volume"] > 1.2 * vol_mean #1.5

        # 4. ATR filter (opsiyonel ama güçlü)
        if "atr14" in df.columns:
            atr_ok = df["atr14"].iloc[entry_idx] / row_df["Close"] > 0.005 #0.01
        else:
            atr_ok = True

        # 🚨 FINAL FILTER
        if not (trend_ok and breakout_ok and vol_ok and atr_ok):
            continue
        
        inst_ok = True
        liq_ok = True

        if "inst_mom_ok" in df.columns:
            inst_ok = bool(df["inst_mom_ok"].iloc[entry_idx])

        if "liq_shock" in df.columns:
            liq_ok = bool(df["liq_shock"].iloc[entry_idx])

        if not (inst_ok and liq_ok):
            continue

        # =========================
        # TRADE EXECUTION
        # =========================

        R = entry_price - stop
        if R <= 0:
            continue

        risk_amount = equity * RISK_PCT

        result_R = simulate_trade(df, entry_idx, entry_price, stop, target)

        pnl = result_R * risk_amount
        equity += pnl

        trades.append({
            "ticker": sym,
            "entry": entry_price,
            "stop": stop,
            "target": target,
            "R_result": result_R,
            "pnl": pnl,
            "equity": equity,
        })

        equity_curve.append(equity)

    out = pd.DataFrame(trades)
    out.to_csv(OUT / "real_trades.csv", index=False)

    # =========================
    # METRİKLER
    # =========================
    if not out.empty:
        winrate = (out["R_result"] > 0).mean() * 100
        avg_R = out["R_result"].mean()
        total_return = (equity / INITIAL_CAPITAL - 1) * 100

        print("\n=== REAL BACKTEST ===")
        print(f"Trades: {len(out)}")
        print(f"Winrate: {winrate:.2f}%")
        print(f"Avg R: {avg_R:.2f}")
        print(f"Return: {total_return:.2f}%")
        print(f"Final Equity: {equity:.2f}")

    else:
        print("\n⚠️ Hiç trade oluşmadı")

    pd.DataFrame({"equity": equity_curve}).to_csv(
        OUT / "equity_curve.csv", index=False
    )


if __name__ == "__main__":
    run()
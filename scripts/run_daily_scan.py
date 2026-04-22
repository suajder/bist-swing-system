import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from bist_swing.data import get_price_data
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.utils import safe_float
from bist_swing.telegram_notifier import TelegramNotifier
from bist_swing.logger import setup_logger

logger = setup_logger("run_daily_scan")

OUT = Path("out/live")
OUT.mkdir(parents=True, exist_ok=True)


def run():

    with open("configs/live.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    adv20_limit = config.get("adv20_min", 10_000_000)
    top_k = config.get("top_k", 5)

    with open("configs/universe.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    logger.info(f"Downloading {len(tickers)} tickers...")

    price_map = get_price_data(tickers)

    se = SignalEngine()
    sp = SignalParams()

    signals = []

    for sym in tickers:

        if sym not in price_map:
            continue

        df = price_map[sym]
        logger.debug(f"{sym} columns: {df.columns}")

        if df.empty or len(df) < 50:
            continue

        sig = se.build(df, sp)

        if sig.empty:
            continue

        t = df.index[-1]

        if t not in sig.index:
            continue

        if not bool(sig.loc[t, "entry_signal"]):
            continue

        # =========================
        # CORE FILTERS
        # =========================
        adv20 = safe_float(df.loc[t, "ADV20"], 0)
        if adv20 < adv20_limit:
            continue

        inst_ok = bool(df.loc[t, "inst_mom_ok"]) if "inst_mom_ok" in df.columns else True
        liq_ok = bool(df.loc[t, "liq_shock"]) if "liq_shock" in df.columns else True

        if not (inst_ok and liq_ok):
            continue

        if "atr14" not in df.columns:
            continue

        entry = float(df.loc[t, "Close"])
        atr = float(df.loc[t, "atr14"])

        if atr <= 0:
            continue

        stop = entry - 2 * atr
        risk = entry - stop

        if risk <= 0:
            continue

        # =========================
        # 🔥 SCORE SYSTEM (SIMPLIFIED)
        # =========================
        score = 0

        trend_ok = df.loc[t, "trend_ok"]
        breakout_ok = df.loc[t, "breakout_ok"]
        vol_ok = df.loc[t, "vol_spike"]

        if trend_ok:
            score += 1

        if breakout_ok:
            score += 1

        if vol_ok:
            score += 1

        # INST MOMENTUM
        if "inst_mom_score" in df.columns:
            if df.loc[t, "inst_mom_score"] > 0:
                score += 1

        # LIQUIDITY BONUS
        if df.loc[t, "ADV20"] > 20_000_000:
            score += 1
        
        logger.info(f"{sym} -> SCORE: {score}")

        # MIN SCORE FILTER
        if score < 3:
            continue

        signals.append({
            "date": t,
            "ticker": sym,
            "entry": entry,
            "stop": stop,
            "risk": risk,
            "score": score,
        })

    out = pd.DataFrame(signals)

    if out.empty:
        logger.info("No signals today.")
        out.to_csv(OUT / "live_signals.csv", index=False)
        return

    # =========================
    # 🔥 TOP SELECTION
    # =========================
    out = out.sort_values("score", ascending=False).head(top_k)

    out.to_csv(OUT / "live_signals.csv", index=False)

    logger.info("=== LIVE SIGNALS ===")
    logger.info(f"\n{out}")

# TELEGRAM GÖNDERİMİ
    try:
        tg = TelegramNotifier()

        if out.empty:
            tg.send("📭 Bugün sinyal yok.")
            return

        msg = "🚀 LIVE SIGNALS\n\n"

        for _, row in out.iterrows():
            msg += (
                f"{row['ticker']}\n"
                f"Entry: {row['entry']:.2f}\n"
                f"Stop: {row['stop']:.2f}\n"
                f"Risk: {row['risk']:.2f}\n\n"
            )

        tg.send(msg)

    except Exception as e:
        logger.error(f"Telegram error: {e}")
        
if __name__ == "__main__":
    run()
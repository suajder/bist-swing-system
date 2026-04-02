import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
portfolio_dir = ROOT / "out" / "portfolio"

trades = pd.read_csv(portfolio_dir / "trades.csv")

exits = trades[trades["Type"] != "ENTRY"].copy()
exits["Date"] = pd.to_datetime(exits["Date"])

today = exits["Date"].max()

# -------------------------
# Günlük PnL
# -------------------------

daily = exits.groupby(exits["Date"].dt.date)["R_PnL"].sum()

today_R = daily.iloc[-1]

# -------------------------
# Haftalık PnL
# -------------------------

exits["week"] = exits["Date"].dt.to_period("W")

weekly = exits.groupby("week")["R_PnL"].sum()

this_week_R = weekly.iloc[-1]

# -------------------------
# Drawdown
# -------------------------

cum = exits["R_PnL"].cumsum()
dd = cum - cum.cummax()

max_dd = dd.min()
current_dd = dd.iloc[-1]

# -------------------------
# Loss streak
# -------------------------

loss_streak = 0
for r in reversed(exits["R_PnL"].tolist()):
    if r < 0:
        loss_streak += 1
    else:
        break


from bist_swing.logger import setup_logger
logger = setup_logger("risk_monitor")

logger.info("\n==============================")
logger.info("CANLI RİSK MONİTÖRÜ")
logger.info("==============================\n")

logger.info(f"Bugün R sonucu      : {today_R:.2f}")
logger.info(f"Bu hafta R sonucu   : {this_week_R:.2f}")
logger.info(f"Aktif Drawdown      : {current_dd:.2f}")
logger.info(f"Max Drawdown        : {max_dd:.2f}")
logger.info(f"Kayıp Serisi        : {loss_streak}")

logger.info("\nRİSK DURUMU")
logger.info("-------------------------")

if today_R <= -3:
    logger.warning("Günlük risk limiti aşıldı → BUGÜN TRADE YOK")

elif this_week_R <= -6:
    logger.warning("Haftalık risk limiti aşıldı → HAFTA SONUNA KADAR TRADE YOK")

elif current_dd <= -15:
    logger.warning("Drawdown yüksek → RİSK %50 DÜŞÜR")

elif loss_streak >= 5:
    logger.warning("Kayıp serisi → POZİSYON SAYISINI AZALT")

else:
    logger.info("✓ Risk normal → TRADE OK")

logger.info("\n==============================")
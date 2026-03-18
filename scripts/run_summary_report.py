import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

portfolio_dir = ROOT / "out" / "portfolio"
rolling_dir = ROOT / "out" / "rolling"

trades = pd.read_csv(portfolio_dir / "trades.csv")
rolling = pd.read_csv(rolling_dir / "rolling_summary.csv")


exits = trades[trades["Type"] != "ENTRY"]

total_R = exits["R_PnL"].sum()
avg_R = exits["R_PnL"].mean()
median_R = exits["R_PnL"].median()
win_rate = (exits["R_PnL"] > 0).mean() * 100

avg_win = exits[exits["R_PnL"] > 0]["R_PnL"].mean()
avg_loss = exits[exits["R_PnL"] < 0]["R_PnL"].mean()

cum = exits["R_PnL"].cumsum()
dd = cum - cum.cummax()

max_dd = dd.min()

positive_windows = (rolling["total_R"] > 0).mean() * 100
median_window = rolling["total_R"].median()
worst_window = rolling["total_R"].min()
worst_dd = rolling["max_dd_R"].min()


print("\n==============================")
print("PORTFÖY SWING SİSTEM RAPORU")
print("==============================\n")

print("PERFORMANS")
print("--------------------------------")
print(f"Toplam R            : {total_R:.2f}")
print(f"Ortalama R          : {avg_R:.2f}")
print(f"Medyan R            : {median_R:.2f}")
print(f"Kazanma Oranı       : %{win_rate:.2f}")
print(f"Ortalama Kazanç     : {avg_win:.2f}R")
print(f"Ortalama Kayıp      : {avg_loss:.2f}R")

print("\nRİSK")
print("--------------------------------")
print(f"Max Drawdown        : {max_dd:.2f}R")

print("\nROLLING ANALİZ")
print("--------------------------------")
print(f"Pozitif Pencere     : %{positive_windows:.1f}")
print(f"Medyan Pencere R    : {median_window:.2f}")
print(f"En Kötü Pencere R   : {worst_window:.2f}")
print(f"En Kötü DD          : {worst_dd:.2f}")

print("\n==============================")
print("RAPOR SONU")
print("==============================")
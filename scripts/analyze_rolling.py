from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLL_CSV = ROOT / "out" / "rolling" / "rolling_summary.csv"

def main():
    df = pd.read_csv(ROLL_CSV)

    # Beklenen kolonlar: window_id, start, end, total_R, max_dd_R, win_rate, n_exits, ...
    # (senin rolling script nasıl yazdıysa isimler hafif farklı olabilir)
    print("\n=== Rolling summary head ===")
    print(df.head(10).to_string(index=False))

    # Toplam R ve maxDD dağılımı
    print("\n=== Key stats ===")
    for col in ["total_R", "max_dd_R", "win_rate", "n_exits"]:
        if col in df.columns:
            print(f"{col:10s}: median={df[col].median(): .4f}  mean={df[col].mean(): .4f}  min={df[col].min(): .4f}  max={df[col].max(): .4f}")

    # En kötü 3 window (total_R)
    if "total_R" in df.columns:
        print("\n=== Worst windows by total_R ===")
        print(df.sort_values("total_R").head(3).to_string(index=False))

    # En kötü 3 window (max_dd_R: en negatif)
    if "max_dd_R" in df.columns:
        print("\n=== Worst windows by max_dd_R ===")
        print(df.sort_values("max_dd_R").head(3).to_string(index=False))

    # Pozitif window oranı
    if "total_R" in df.columns:
        pos = (df["total_R"] > 0).mean() * 100
        print(f"\nPositive windows: {pos:.1f}%")

if __name__ == "__main__":
    main()
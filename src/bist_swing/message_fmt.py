from __future__ import annotations

import pandas as pd

def fmt_entry_tr(
    ticker: str,
    asof: pd.Timestamp,
    next_open_est: float,
    stop_px: float,
    r: float,
    tp1: float,
    tp2: float,
    tp1_R: float,
    tp2_R: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    mom20: float,
    adv20: float,
    score: float,
) -> str:
    trend = "↑" if ema20 > ema50 else "↓"
    adv_mr = adv20 / 1_000_000_000
    mom_pct = mom20 * 100.0
    return (
        f"📈 {ticker} | GİRİŞ SİNYALİ (Swing)\n"
        f"AsOf: {asof.date()} | Next Açılış≈{next_open_est:.2f} | Stop={stop_px:.2f} | 1R={r:.2f}\n\n"
        f"EMA20/50: {ema20:.2f}/{ema50:.2f} ({trend}) | RSI14: {rsi14:.1f} | Mom20: %{mom_pct:.2f}\n"
        f"TP1({tp1_R:.1f}R): {tp1:.2f} | TP2({tp2_R:.1f}R): {tp2:.2f}\n"
        f"ADV20: {adv_mr:.2f} Mr TL | Skor: {score:.4f}"
    )

def fmt_event_tr(ticker: str, event: str, date: pd.Timestamp, px: float, info: str = "") -> str:
    tr = {
        "STOP": "ZARAR DURDUR",
        "TP1": "1. HEDEF",
        "TP2": "2. HEDEF",
        "WEEKLY_EXIT": "HAFTALIK ÇIKIŞ",
        "DATA_ERROR": "VERİ HATASI",
    }.get(event, event)
    note = f"\nNot: {info}" if info else ""
    return f"⚠️ {ticker} | {tr}\nTarih: {date.date()} | Fiyat: {px:.2f}{note}"

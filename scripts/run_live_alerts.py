from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

from bist_swing.data import DataProvider
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams
from bist_swing.telegram_notifier import TelegramNotifier
from bist_swing.state_store import StateStore

# Load .env early (TG_BOT_TOKEN, TG_CHAT_ID)
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "signals_state.json"


def safe_send(notifier: TelegramNotifier, text: str) -> None:
    try:
        notifier.send(text)
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")


def bist_session_closed(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:
        return True
    return now_ist.time() >= time(18, 10)


def pick_safe_asof(any_df: pd.DataFrame, user_asof: str | None) -> pd.Timestamp:
    idx = list(any_df.index)
    if len(idx) < 3:
        raise RuntimeError("Not enough bars for safe asof.")
    if user_asof:
        a = pd.to_datetime(user_asof)
        if a not in any_df.index:
            candidates = [d for d in idx if d <= a]
            if not candidates:
                raise RuntimeError("No suitable date <= --asof")
            a = candidates[-1]
        loc = any_df.index.get_loc(a)
        if isinstance(loc, slice) or loc + 1 >= len(any_df.index):
            raise RuntimeError("--asof has no next bar.")
        return pd.to_datetime(a)

    # safe default: use previous completed bar
    safe = pd.to_datetime(any_df.index[-2])
    loc = any_df.index.get_loc(safe)
    if isinstance(loc, slice) or loc + 1 >= len(any_df.index):
        safe = pd.to_datetime(any_df.index[-3])
    return safe


def load_universe(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def fmt_entry_tr(
    ticker: str,
    asof: pd.Timestamp,
    score: float,
    close_px: float,
    next_open_est: float,
    stop_px: float,
    r: float,
    tp1: float,
    tp2: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    mom20: float,
    adv20: float,
    tp1_R: float,
    tp2_R: float,
) -> str:
    trend = "↑" if ema20 > ema50 else "↓"
    adv_mr = adv20 / 1_000_000_000
    mom_pct = mom20 * 100.0
    return (
        f"📈 {ticker} | GİRİŞ SİNYALİ (Swing)\n"
        f"AsOf: {asof.date()} | Kapanış={close_px:.2f} | Skor={score:.4f}\n"
        f"Next Açılış≈{next_open_est:.2f} | Stop={stop_px:.2f} | 1R={r:.2f}\n\n"
        f"EMA20/50: {ema20:.2f}/{ema50:.2f} ({trend}) | RSI14: {rsi14:.1f} | Mom20: %{mom_pct:.2f}\n"
        f"TP1({tp1_R:.1f}R): {tp1:.2f} | TP2({tp2_R:.1f}R): {tp2:.2f}\n"
        f"ADV20: {adv_mr:.2f} Mr TL"
    )


def compute_entry_levels(
    df: pd.DataFrame,
    sig: pd.DataFrame,
    asof: pd.Timestamp,
    bp: BacktestParams,
) -> tuple[float, float, float, pd.Timestamp]:
    """
    Entry assumed at next bar Open (model). Stop from swing-low or ATR-based.
    Returns: (entry_px, stop_px, R, next_bar_date)
    """
    idx = df.index
    i = idx.get_loc(asof)
    if i + 1 >= len(idx):
        raise RuntimeError("asof has no next bar for entry")
    nxt = idx[i + 1]

    slip = bp.slippage_bps / 10000.0
    entry_px = float(df.loc[nxt, "Open"]) * (1 + slip)

    # swing stop
    swing_low = float(df["Low"].loc[:asof].tail(bp.swing_lookback).min())
    stop_swing = swing_low * (1 - bp.swing_buffer)

    # atr stop around ema20
    atr14 = float(sig.loc[asof, "atr14"])
    ema20 = float(sig.loc[asof, "d_ema20"])
    stop_atr = ema20 - (bp.atr_stop_mult * atr14)

    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

    r = entry_px - stop_px
    if r <= 0:
        raise RuntimeError("Invalid R (entry <= stop)")
    return entry_px, stop_px, r, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "live.yaml"))
    ap.add_argument("--universe", default=str(ROOT / "configs" / "universe.txt"))
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = load_universe(Path(args.universe))

    # DataProvider uses requests+Yahoo chart endpoint (no yfinance/curl)
    period_days = int(cfg.get("period_days", 365 * 3))
    provider = DataProvider(period_days=period_days)

    se = SignalEngine()
    sp = SignalParams(**cfg.get("signal_params", {}))
    bp = BacktestParams(**cfg.get("backtest_params", {}))

    notifier = TelegramNotifier()
    store = StateStore(STATE_PATH)
    state = store.load()

    # Fetch
    price_map: dict[str, pd.DataFrame] = {}
    for t in universe:
        try:
            price_map[t] = provider.get(t)
        except Exception as e:
            key = f"DATAERR::{t}::{datetime.now().date()}"
            if not store.seen(state, key):
                safe_send(notifier, f"⚠️ {t} | VERİ HATASI | {str(e)[:180]}")
                store.mark(state, key)
            continue

    if not price_map:
        safe_send(notisfier := notifier, "⚠️ Live run: hiç veri çekilemedi.")
        store.save(state)
        return

    any_df = next(iter(price_map.values()))
    asof = pick_safe_asof(any_df, args.asof)

    now_ist = datetime.now(ZoneInfo("Europe/Istanbul"))
    if not bist_session_closed(now_ist) and args.asof is None:
        safe_send(notifier, f"ℹ️ BIST açık olabilir; güvenli asof={asof.date()} (son tamamlanan bar).")

    # Scan
    adv_min = float(cfg.get("adv20_min", 50_000_000.0))
    top_k = int(cfg.get("top_k", 5))

    candidates: list[tuple[str, float, float]] = []  # (ticker, score, close)
    sig_cache: dict[str, pd.DataFrame] = {}

    for t, df in price_map.items():
        if asof not in df.index:
            continue
        if float(df.loc[asof, "ADV20"]) < adv_min:
            continue

        sig = se.build(df, sp)
        sig_cache[t] = sig

        if bool(sig.loc[asof, "entry_signal"]):
            # simple score: momentum + tiny trend strength
            score = float(sig.loc[asof, "mom20"]) + 0.001 * float(sig.loc[asof, "d_ema20"] / sig.loc[asof, "d_ema50"] - 1)
            candidates.append((t, score, float(df.loc[asof, "Close"])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    picks = candidates[:top_k]

    if picks:
        header_key = f"ENTRY_HDR::{asof.date()}::top{top_k}::adv{int(adv_min)}"

        if not store.seen(state, header_key):
            safe_send(notifier, f"✅ ENTRY_CANDIDATES asof={asof.date()} | top{top_k} | adv_min={adv_min:,.0f}")
            store.mark(state, header_key)

        for t, score, close_px in picks:
            key = f"ENTRY::{t}::{asof.date()}"
            if store.seen(state, key):
                continue

            df = price_map[t]
            sig = sig_cache.get(t) 
            if sig is None:
                sig = se.build(df, sp)

            try:
                entry_px, stop_px, r, nxt = compute_entry_levels(df, sig, asof, bp)
                tp1 = entry_px + bp.tp1_R * r
                tp2 = entry_px + bp.tp2_R * r

                ema20 = float(sig.loc[asof, "d_ema20"])
                ema50 = float(sig.loc[asof, "d_ema50"])
                rsi14 = float(sig.loc[asof, "d_rsi14"])
                mom20 = float(sig.loc[asof, "mom20"])
                adv20 = float(df.loc[asof, "ADV20"])

                msg = fmt_entry_tr(
                    ticker=t,
                    asof=asof,
                    score=float(score),
                    close_px=float(close_px),
                    next_open_est=float(entry_px),
                    stop_px=float(stop_px),
                    r=float(r),
                    tp1=float(tp1),
                    tp2=float(tp2),
                    ema20=ema20,
                    ema50=ema50,
                    rsi14=rsi14,
                    mom20=mom20,
                    adv20=adv20,
                    tp1_R=float(bp.tp1_R),
                    tp2_R=float(bp.tp2_R),
                )
                safe_send(notifier, msg)
                store.mark(state, key)

            except Exception as e:
                safe_send(notifier, f"⚠️ {t} | SİNYAL HESAP HATASI | {str(e)[:180]}")
                store.mark(state, key)

    else:
        key = f"NOENTRY::{asof.date()}"
        if not store.seen(state, key):
            safe_send(notifier, f"⛔ NO ENTRY asof={asof.date()} | adv_min={adv_min:,.0f}")
            store.mark(state, key)

    store.save(state)


if __name__ == "__main__":
    main()

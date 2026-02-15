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

from bist_swing.position_state import Position
from bist_swing.live_events import compute_entry_levels, evaluate_position_events

# .env (TG_BOT_TOKEN, TG_CHAT_ID, vs.)
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "signals_state.json"


def safe_send(notifier: TelegramNotifier, text: str) -> None:
    try:
        notifier.send(text)
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")


def bist_session_closed(now_ist: datetime) -> bool:
    # Basit yaklaşım: Hafta sonu kapalı, hafta içi 18:10 sonrası kapalı varsay.
    if now_ist.weekday() >= 5:
        return True
    return now_ist.time() >= time(18, 10)


def load_universe(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def pick_safe_asof(any_df: pd.DataFrame, user_asof: str | None) -> pd.Timestamp:
    idx = any_df.index
    if len(idx) < 3:
        raise RuntimeError("Not enough bars for safe asof.")

    if user_asof:
        a = pd.to_datetime(user_asof)
        if a not in idx:
            candidates = [d for d in idx if d <= a]
            if not candidates:
                raise RuntimeError("No suitable date <= --asof")
            a = candidates[-1]
        loc = idx.get_loc(a)
        if isinstance(loc, slice) or loc + 1 >= len(idx):
            raise RuntimeError("--asof has no next bar.")
        return pd.to_datetime(a)

    # default: son tamamlanmış bar = -2
    safe = pd.to_datetime(idx[-2])
    loc = idx.get_loc(safe)
    if isinstance(loc, slice) or loc + 1 >= len(idx):
        safe = pd.to_datetime(idx[-3])
    return safe


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "live.yaml"))
    ap.add_argument("--universe", default=str(ROOT / "configs" / "universe.txt"))
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = load_universe(Path(args.universe))

    notifier = TelegramNotifier()
    store = StateStore(STATE_PATH)
    state = store.load()
    state.setdefault("positions", {})  # <- pozisyonlar burada tutulur

    # Parametreler
    period_days = int(cfg.get("period_days", 1095))  # ~3y
    adv_min = float(cfg.get("adv20_min", 50_000_000.0))
    top_k = int(cfg.get("top_k", 5))

    sp = SignalParams(**cfg.get("signal_params", {}))
    bp = BacktestParams(**cfg.get("backtest_params", {}))

    provider = DataProvider(period_days=period_days)
    se = SignalEngine()

    # Veri çek
    price_map: dict[str, pd.DataFrame] = {}
    bad_tickers: list[str] = []

    for t in universe:
        try:
            df = provider.get(t)
            price_map[t] = df
        except Exception as e:
            bad_tickers.append(t)
            key = f"DATAERR::{t}::{datetime.now().date()}"
            if not store.seen(state, key):
                msg = str(e)
                short = "404/Not Found (Yahoo)" if "404" in msg or "Not Found" in msg else msg[:140]
                safe_send(notifier, f"⚠️ {t} | VERİ HATASI | {short}")
                store.mark(state, key)

    if not price_map:
        safe_send(notifier, "⚠️ Live run: hiç veri çekilemedi.")
        store.save(state)
        return

    any_df = next(iter(price_map.values()))
    asof = pick_safe_asof(any_df, args.asof)

    ok_n = len(price_map)
    bad_n = len(bad_tickers)
    sum_key = f"RUN_SUMMARY::{asof.date()}"
    if not store.seen(state, sum_key):
        safe_send(
            notifier,
            f"ℹ️ Run özeti | asof={asof.date()} | Veri OK: {ok_n} | Hatalı: {bad_n}"
            + (f"\nHatalı semboller (ilk 10): {', '.join(bad_tickers[:10])}" if bad_n else "")
        )
        store.mark(state, sum_key)


    now_ist = datetime.now(ZoneInfo("Europe/Istanbul"))
    if args.asof is None and not bist_session_closed(now_ist):
        safe_send(notifier, f"ℹ️ BIST açık olabilir; güvenli asof={asof.date()} (son tamamlanan bar).")

    # ---------------------------------------------------------------------
    # 1) AÇIK POZİSYONLARI YÖNET (STOP/TP1/TP2/WEEKLY_EXIT) + TEKNİK MESAJ
    # ---------------------------------------------------------------------
    for sym, pdict in list(state["positions"].items()):
        try:
            pos = Position.from_dict(pdict)
            if not pos.is_open:
                continue
            if sym not in price_map:
                continue

            dfp = price_map[sym]
            if asof not in dfp.index:
                continue

            sigp = se.build(dfp, sp)
            events = evaluate_position_events(dfp, sigp, bp, pos, asof)

            # Teknik metrikler (asof barından)
            ema20 = float(sigp.loc[asof, "d_ema20"])
            ema50 = float(sigp.loc[asof, "d_ema50"])
            rsi14 = float(sigp.loc[asof, "d_rsi14"])
            mom20 = float(sigp.loc[asof, "mom20"])
            trend = "↑" if ema20 > ema50 else "↓"
            mom_pct = mom20 * 100.0

            for ev in events:
                ev_key = f"EV::{ev.event}::{ev.ticker}::{ev.date.date()}"
                if store.seen(state, ev_key):
                    continue

                tp1_lvl = pos.entry_px + bp.tp1_R * pos.r
                tp2_lvl = pos.entry_px + bp.tp2_R * pos.r

                tr = {
                    "STOP": "ZARAR DURDUR",
                    "TP1": "1. HEDEF",
                    "TP2": "2. HEDEF",
                    "WEEKLY_EXIT": "HAFTALIK ÇIKIŞ",
                }.get(ev.event, ev.event)

                msg = (
                    f"⚠️ {ev.ticker} | {tr}\n"
                    f"Tarih: {ev.date.date()} | Fiyat: {ev.px:.2f}\n"
                    f"Entry={pos.entry_px:.2f} | Stop={pos.stop_px:.2f} | TP1={tp1_lvl:.2f} | TP2={tp2_lvl:.2f}\n"
                    f"EMA20/50: {ema20:.2f}/{ema50:.2f} ({trend}) | RSI14: {rsi14:.1f} | Mom20: %{mom_pct:.2f}\n"
                    f"Not: {ev.info}"
                )

                safe_send(notifier, msg)
                store.mark(state, ev_key)

                # A kuralı: TP1 -> %50 realize + stop=entry (breakeven)
                if ev.event == "TP1":
                    pos.tp1_done = True
                    pos.qty = max(0.5, pos.qty * 0.5)
                    pos.stop_px = pos.entry_px
                elif ev.event == "TP2":
                    pos.tp2_done = True
                    pos.is_open = False
                elif ev.event in ("STOP", "WEEKLY_EXIT"):
                    pos.is_open = False

            state["positions"][sym] = pos.to_dict()

        except Exception as e:
            print(f"[WARN] position manage failed for {sym}: {e}")

    # ---------------------------------------------------------------------
    # 2) ENTRY ADAYLARI (top_k) + ENTRY MESAJI + POSITION AÇ
    # ---------------------------------------------------------------------
    candidates: list[tuple[str, float, float]] = []  # (ticker, score, close)
    sig_cache: dict[str, pd.DataFrame] = {}

    for t, df in price_map.items():
        try:
            if asof not in df.index:
                continue
            if "ADV20" in df.columns and float(df.loc[asof, "ADV20"]) < adv_min:
                continue

            sig = se.build(df, sp)
            sig_cache[t] = sig

            # entry_signal bekleniyor (signals.py içinde)
            if bool(sig.loc[asof, "entry_signal"]):
                # skor: momentum + çok küçük trend farkı
                score = float(sig.loc[asof, "mom20"]) + 0.001 * float(sig.loc[asof, "d_ema20"] / sig.loc[asof, "d_ema50"] - 1)
                candidates.append((t, score, float(df.loc[asof, "Close"])))
        except Exception as e:
            print(f"[WARN] scan failed for {t}: {e}")

    candidates.sort(key=lambda x: x[1], reverse=True)
    picks = candidates[:top_k]

    header_key = f"ENTRY_HDR::{asof.date()}::top{top_k}::adv{int(adv_min)}"
    if picks:
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
                adv20 = float(df.loc[asof, "ADV20"]) if "ADV20" in df.columns else 0.0

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

                # Position aç (planned entry -> nxt open)
                # Aynı sembolde açık position yoksa aç
                prev = state["positions"].get(t)
                if not prev or not prev.get("is_open", False):
                    state["positions"][t] = Position(
                        ticker=t,
                        entry_date=str(pd.to_datetime(nxt).date()),
                        entry_px=float(entry_px),
                        stop_px=float(stop_px),
                        r=float(r),
                        qty=1.0,
                        tp1_done=False,
                        tp2_done=False,
                        is_open=True,
                    ).to_dict()

            except Exception as e:
                safe_send(notifier, f"⚠️ {t} | SİNYAL HESAP HATASI | {str(e)[:180]}")
                store.mark(state, key)
    else:
        if not store.seen(state, header_key):
            safe_send(notifier, f"⛔ NO ENTRY asof={asof.date()} | adv_min={adv_min:,.0f}")
            store.mark(state, header_key)

    store.save(state)


if __name__ == "__main__":
    main()

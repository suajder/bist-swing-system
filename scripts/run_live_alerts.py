from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from bist_swing.data import DataProvider
from bist_swing.signals import SignalEngine, SignalParams
from bist_swing.backtest import BacktestParams
from bist_swing.telegram_notifier import TelegramNotifier
from bist_swing.state_store import StateStore

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "signals_state.json"

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
    safe = pd.to_datetime(any_df.index[-2])
    loc = any_df.index.get_loc(safe)
    if isinstance(loc, slice) or loc + 1 >= len(any_df.index):
        safe = pd.to_datetime(any_df.index[-3])
    return safe

def load_universe(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]

def fmt_event(ticker: str, event: str, asof: pd.Timestamp, px: float | None = None, extra: str = "") -> str:
    p = f" @ {px:.2f}" if px is not None else ""
    return f"{ticker} | {event} | asof={asof.date()}{p} {extra}".strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "live.yaml"))
    ap.add_argument("--universe", default=str(ROOT / "configs" / "universe.txt"))
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = load_universe(Path(args.universe))

    provider = DataProvider(period_days=int(cfg.get("period_days", 365*3)))

    se = SignalEngine()
    sp = SignalParams(**cfg.get("signal_params", {}))
    bp = BacktestParams(**cfg.get("backtest_params", {}))

    notifier = TelegramNotifier()
    store = StateStore(STATE_PATH)
    state = store.load()

    price_map = {}
    for t in universe:
        try:
            price_map[t] = provider.get(t)
        except Exception as e:
            key = f"DATAERR::{t}::{datetime.now().date()}"
            if not store.seen(state, key):
                notifier.send(fmt_event(t, "DATA_ERROR", pd.Timestamp.today(), extra=str(e)[:180]))
                store.mark(state, key)
            continue

    if not price_map:
        notifier.send("Live run: no tickers fetched.")
        return

    any_df = next(iter(price_map.values()))
    asof = pick_safe_asof(any_df, args.asof)

    now_ist = datetime.now(ZoneInfo("Europe/Istanbul"))
    if not bist_session_closed(now_ist) and args.asof is None:
        notifier.send(f"INFO: BIST may be open; using safe asof={asof.date()} (index[-2]).")

    # universe scan for today's entry signals (likidite filtresi)
    adv_min = float(cfg.get("adv20_min", 50_000_000.0))
    top_k = int(cfg.get("top_k", 5))

    candidates = []
    for t, df in price_map.items():
        if asof not in df.index:
            continue
        if float(df.loc[asof, "ADV20"]) < adv_min:
            continue
        sig = se.build(df, sp)
        if bool(sig.loc[asof, "entry_signal"]):
            # simple score = mom20 + trend strength
            score = float(sig.loc[asof, "mom20"]) + 0.001 * float(sig.loc[asof, "d_ema20"] / sig.loc[asof, "d_ema50"] - 1)
            candidates.append((t, score, float(df.loc[asof, "Close"])))

    candidates.sort(key=lambda x: x[1], reverse=True)
    picks = candidates[:top_k]

    # event messaging (dedup)
    if picks:
        notifier.send(f"ENTRY_CANDIDATES asof={asof.date()} | top{top_k}")
        for t, score, close_px in picks:
            key = f"ENTRY::{t}::{asof.date()}"
            if store.seen(state, key):
                continue
            notifier.send(fmt_event(t, "ENTRY_SIGNAL", asof, px=close_px, extra=f"score={score:.4f}"))
            store.mark(state, key)
    else:
        key = f"NOENTRY::{asof.date()}"
        if not store.seen(state, key):
            notifier.send(f"NO ENTRY signals asof={asof.date()} (adv_min={adv_min:,.0f})")
            store.mark(state, key)

    store.save(state)

if __name__ == "__main__":
    main()

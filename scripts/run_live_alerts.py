from __future__ import annotations

import argparse
from pathlib import Path

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
from bist_swing.logger import log_trade, log_equity, setup_logger

logger = setup_logger("run_live_alerts")

from bist_swing.risk_engine import (
    load_risk_state,
    calculate_position_size,
    validate_trade,
    compute_portfolio_risk,
    compute_drawdown,
    kill_switch_triggered,
    compute_unrealized_pnl,
    compute_total_equity,
    compute_floating_dd,
    update_equity,
)

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(dotenv_path=ROOT / ".env", override=True)

STATE_PATH = ROOT / "signals_state.json"


# ================================
# UTIL
# ================================

def safe_send(notifier, text):
    try:
        notifier.send(text)
    except Exception as e:
        logger.error(str(e))


def pick_safe_asof(df: pd.DataFrame):
    return pd.to_datetime(df.index[-2])


def market_filter(df: pd.DataFrame, asof):
    try:
        close = df.loc[asof, "Close"]
        ema50 = df["Close"].ewm(span=50).mean().loc[asof]
        ema200 = df["Close"].ewm(span=200).mean().loc[asof]
        return close > ema50 > ema200
    except Exception:
        return True


# ================================
# MAIN
# ================================

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/live.yaml"))
    ap.add_argument("--universe", default=str(ROOT / "configs/universe.txt"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    notifier = TelegramNotifier()
    store = StateStore(STATE_PATH)
    state = store.load()
    state.setdefault("positions", {})

    # RISK STATE LOAD
    risk_state = load_risk_state(
        cfg.get("capital", 100000),
        cfg.get("risk_per_trade", 0.01),
    )

    capital = cfg.get("capital", 100000)
    max_risk_pct = cfg.get("max_portfolio_risk_pct", 0.05)
    max_pos = cfg.get("max_open_positions", 5)

    sp = SignalParams(**cfg.get("signal_params", {}))
    bp = BacktestParams(**cfg.get("backtest_params", {}))

    provider = DataProvider()
    se = SignalEngine()

    universe = [line.strip() for line in Path(args.universe).read_text().splitlines() if line.strip()]

    price_map = {}
    for t in universe:
        try:
            price_map[t] = provider.get(t)
        except Exception:
            continue

    if not price_map:
        return

    asof = pick_safe_asof(next(iter(price_map.values())))

    # MARKET FILTER
    if "XU100.IS" in price_map:
        if not market_filter(price_map["XU100.IS"], asof):
            safe_send(notifier, "❌ MARKET BAD")
            return
    
    # ---------------- FLOATING DD HESAPLA ----------------
    unrealized = compute_unrealized_pnl(state["positions"], price_map, asof)
    total_equity = compute_total_equity(risk_state, unrealized)
    floating_dd = compute_floating_dd(risk_state, total_equity)

    if floating_dd < -0.08:
        safe_send(notifier, "⚠️ DD %8 geçti - risk azaltılıyor")

    # ---------------- KILL SWITCH ----------------
    if kill_switch_triggered(
        risk_state,
        floating_dd,
        realized_limit=cfg.get("kill_switch_dd", -0.20),
        floating_limit=cfg.get("kill_switch_floating_dd", -0.10),
    ):
        safe_send(
            notifier,
            f"🛑 KILL SWITCH AKTİF\n"
            f"Realized DD: %{risk_state.current_dd*100:.2f}\n"
            f"Floating DD: %{floating_dd*100:.2f}"
        )
        return

    total_risk = compute_portfolio_risk(state["positions"])
    open_positions = sum(1 for p in state["positions"].values() if p["is_open"])

    # ================================
    # POSITION MANAGEMENT (EXIT ENGINE)
    # ================================

    for sym, pdict in list(state["positions"].items()):

        try:
            pos = Position.from_dict(pdict)

            if not pos.is_open:
                continue

            if sym not in price_map:
                continue

            df = price_map[sym]
            sig = se.build(df, sp)

            events = evaluate_position_events(
                pos,
                sig,
                asof,
                bp,
                risk_state   # 🔥 KRİTİK
            )

            for ev in events:

                safe_send(
                    notifier,
                    f"⚠️ {ev.ticker} | {ev.event}\nFiyat: {ev.px:.2f}\n{ev.note}"
                )

                # STATE UPDATE
                if ev.event == "TP1":
                    pos.tp1_done = True
                    pos.qty *= 0.5
                    pos.stop_px = pos.entry_px

                elif ev.event in ["TP2", "STOP", "WEEKLY_EXIT"]:
                    pos.is_open = False

            state["positions"][sym] = pos.to_dict()

        except Exception as e:
            logger.error(f"[EXIT ERROR] {sym}: {e}")

    # SCAN
    candidates = []

    for t, df in price_map.items():
        try:
            sig = se.build(df, sp)

            if not bool(sig.loc[asof, "entry_signal"]):
                continue

            entry_px, stop_px, r, nxt = compute_entry_levels(df, sig, asof, bp)

            if (r / entry_px) > 0.05:
                continue

            score = float(sig.loc[asof, "mom20"])
            candidates.append((t, score, entry_px, stop_px, r, sig))

        except Exception:
            continue

    candidates.sort(key=lambda x: x[1], reverse=True)

    # ================================
    # EXIT ENGINE (CRITICAL)
    # ================================

    for ticker, p_dict in state["positions"].items():

        if not p_dict["is_open"]:
            continue

        if ticker not in price_map:
            continue

        df = price_map[ticker]
        sig = se.build(df, sp)

        pos = Position(**p_dict)

        events = evaluate_position_events(pos, sig, asof, bp)

        for ev in events:

            exit_price = ev.px

            # ---------------- PNL ----------------
            pnl = (exit_price - pos.entry_px) * pos.qty

            # ---------------- EQUITY UPDATE ----------------
            update_equity(risk_state, pnl)

            # ---------------- POSITION UPDATE ----------------
            if ev.event == "STOP":
                pos.is_open = False

            elif ev.event == "TP1":
                pos.tp1_done = True

            elif ev.event == "TP2":
                pos.tp2_done = True
                pos.is_open = False

            elif ev.event == "WEEKLY_EXIT":
                pos.is_open = False

            # ---------------- SAVE BACK ----------------
            state["positions"][ticker] = pos.to_dict()

            # ---------------- TELEGRAM ----------------
            safe_send(
                notifier,
                f"📉 EXIT {ticker}\n"
                f"{ev.event} @ {exit_price:.2f}\n"
                f"PnL: {pnl:.0f} TL"
            )

    # ENTRY
    for t, score, entry_px, stop_px, r, sig in candidates:

        if open_positions >= max_pos:
            break

        if not validate_trade(entry_px, stop_px):
            continue

        qty = calculate_position_size(risk_state, entry_px, stop_px)
        if qty <= 0:
            continue

        potential_risk = total_risk + (r * qty)
        if (potential_risk / capital) > max_risk_pct:
            continue

        tp1 = entry_px + bp.tp1_R * r
        tp2 = entry_px + bp.tp2_R * r

        msg = (
            f"🚀 {t}\n"
            f"Entry: {entry_px:.2f}\n"
            f"Stop: {stop_px:.2f}\n"
            f"TP1: {tp1:.2f} | TP2: {tp2:.2f}\n"
            f"Lot: {qty}"
        )

        safe_send(notifier, msg)

        state["positions"][t] = Position(
            ticker=t,
            entry_date=str(asof.date()),
            entry_px=entry_px,
            stop_px=stop_px,
            r=r,
            qty=qty,
            tp1_done=False,
            tp2_done=False,
            is_open=True,
        ).to_dict()

        log_trade({
            "date": str(asof.date()),
            "ticker": t,
            "entry": entry_px,
            "stop": stop_px,
            "qty": qty,
        })

        _realized_dd = compute_drawdown(risk_state)
        _risk_pct = total_risk / capital if capital else 0
        log_equity(
            date=str(asof.date()),
            equity=total_equity,
            realized_dd=_realized_dd,
            floating_dd=floating_dd,
            open_positions=open_positions,
            risk_pct=_risk_pct
        )

        total_risk += r * qty
        open_positions += 1

    # EOD REPORT
    total_risk = compute_portfolio_risk(state["positions"])
    total_qty = sum(p["qty"] for p in state["positions"].values() if p["is_open"])
    open_positions = sum(1 for p in state["positions"].values() if p["is_open"])

    # ---------------- UNREALIZED ----------------
    unrealized = compute_unrealized_pnl(state["positions"], price_map, asof)

    # ---------------- TOTAL EQUITY ----------------
    total_equity = compute_total_equity(risk_state, unrealized)

    # ---------------- DD ----------------
    realized_dd = compute_drawdown(risk_state)
    floating_dd = compute_floating_dd(risk_state, total_equity)

    risk_pct = total_risk / capital if capital else 0

    if risk_pct < 0.03:
        pass
    elif risk_pct < max_risk_pct:
        pass
    else:
        pass

    msg = (
        f"📊 EOD RAPOR\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Pozisyon: {open_positions}/{max_pos}\n"
        f"Risk: {total_risk:,.0f} TL (%{risk_pct*100:.2f})\n"
        f"Max: %{max_risk_pct*100:.0f}\n\n"

        f"💰 Equity: {total_equity:,.0f} TL\n"
        f"📈 Unrealized: {unrealized:,.0f} TL\n\n"

        f"DD (Realized): %{realized_dd*100:.2f}\n"
        f"DD (Floating): %{floating_dd*100:.2f}\n\n"

        f"📦 Lot: {total_qty}"
    )

    safe_send(notifier, msg)

    store.save(state)


if __name__ == "__main__":
    main()
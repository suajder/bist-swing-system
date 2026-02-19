from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestParams
from .live_events import resolve_intrabar_exit
from .reporting import plot_equity
from .signals import SignalEngine, SignalParams


@dataclass(frozen=True)
class PortfolioParams:
    # portfolio constraints
    max_open: int = 5

    # starting capital
    initial_equity: float = 50_000.0

    # liquidity filter
    adv20_min: float = 50_000_000.0

    # risk-based position sizing  
    risk_pct: float = 0.02 # %2 risk per trade (balanced growth)

    daily_stop_R: float = 3.0
    weekly_stop_R: float = 6.0

    max_notional_pct: float = 0.35       # equity'nin max %35'i tek pozisyona
    min_cash_buffer_pct: float = 0.05    # equity'nin %5'i nakitte kalsın

    # ranking weights
    w_model: float = 1.0
    w_rsi: float = 0.15
    w_mom: float = 0.10
    w_adv: float = 0.05


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - mu) / sd

def r_summary(trdf: pd.DataFrame) -> Dict[str, float]:
    """
    Compute R-based performance summary from trade log.
    Expects columns: Type, R_PnL
    We summarize EXIT trades (non-ENTRY) by default.
    """
    if trdf.empty or "R_PnL" not in trdf.columns or "Type" not in trdf.columns:
        return {}

    # exits only (ENTRY has 0 by design)
    ex = trdf[trdf["Type"] != "ENTRY"].copy()
    if ex.empty:
        return {}

    r = ex["R_PnL"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}

    n = float(len(r))
    total_r = float(r.sum())
    avg_r = float(r.mean())
    med_r = float(r.median())

    wins = r[r > 0]
    losses = r[r < 0]

    win_rate = float((r > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
    expectancy = float(win_rate * avg_win + (1.0 - win_rate) * avg_loss)

    # cumulative R & max drawdown in R
    cum = r.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd_r = float(dd.min())  # negative

    # max consecutive loss streak (in R and count)
    loss_mask = (r < 0).to_numpy()
    max_streak = 0
    cur = 0
    for v in loss_mask:
        if v:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    # max consecutive loss sum (most negative contiguous sum)
    # (Kadane variant on losses only)
    min_ending = 0.0
    min_so_far = 0.0
    for x in r.to_numpy():
        min_ending = min(0.0, min_ending + float(x))
        min_so_far = min(min_so_far, min_ending)
    max_consec_loss_r = float(min_so_far)  # negative

    return {
        "n_exits": n,
        "total_R": total_r,
        "avg_R": avg_r,
        "median_R": med_r,
        "win_rate": win_rate,
        "avg_win_R": avg_win,
        "avg_loss_R": avg_loss,
        "expectancy_R": expectancy,
        "max_dd_R": max_dd_r,
        "max_loss_streak_n": float(max_streak),
        "max_consec_loss_R": max_consec_loss_r,
    }

@dataclass
class _Pos:
    entry_px: float
    stop_px: float
    R: float
    shares: float
    orig_shares: float
    tp1: bool = False
    tp2: bool = False


def portfolio_backtest_pro(
    *,
    se: SignalEngine,
    tickers: List[str],
    price_map: Dict[str, pd.DataFrame],
    best_cfg_map: Dict[str, Tuple[SignalParams, BacktestParams]],
    model_score_map: Dict[str, float],
    test_start: str,
    test_end: Optional[str],
    pparams: PortfolioParams,
    outdir: Path,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    start_dt = pd.to_datetime(test_start)
    end_dt = pd.to_datetime(test_end) if test_end else None

    # build signals per ticker (full history)
    sig_map = {t: se.build(price_map[t], best_cfg_map[t][0]) for t in tickers}

    # common calendar: use first ticker
    cal = price_map[tickers[0]].index
    cal = cal[cal >= start_dt]
    if end_dt is not None:
        cal = cal[cal <= end_dt]
    if len(cal) < 10:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame()}

    cash = float(pparams.initial_equity)
    positions: Dict[str, _Pos] = {}
    trades: List[tuple] = []

    def log_trade(dt, ticker: str, typ: str, px: float, shares: float, r_pnl: float = 0.0) -> None:
        notional = float(shares) * float(px) if np.isfinite(px) else np.nan
        trades.append((dt, ticker, typ, float(px), float(shares), float(notional), float(r_pnl)))

    day_r = 0.0
    week_r = 0.0
    cur_day = None
    cur_week = None
    equity_rows: List[tuple] = []

    # Use params from first ticker for costs (or could keep per-ticker)
    any_bp = best_cfg_map[tickers[0]][1]
    slip = any_bp.slippage_bps / 10000.0
    fee = any_bp.fee_bps / 10000.0

    for i in range(len(cal) - 1):
        t = cal[i]

        d = pd.Timestamp(t).date()
        w = pd.Timestamp(t).isocalendar().week

        if cur_day is None or d != cur_day:
            cur_day = d
            day_r = 0.0

        if cur_week is None or w != cur_week:
            cur_week = w
            week_r = 0.0

        nxt = cal[i + 1]

        # mark-to-market close
        eq = cash
        for sym, pos in positions.items():
            df = price_map[sym]
            if t in df.index:
                eq += pos.shares * float(df.loc[t, "Close"])
        equity_rows.append((t, eq, cash, len(positions)))

        # manage exits on nxt OHLC
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = price_map[sym]
            sig = sig_map[sym]
            _, bp = best_cfg_map[sym]

            if t not in df.index or nxt not in df.index or t not in sig.index:
                continue

            o2 = float(df.loc[nxt, "Open"])
            h2 = float(df.loc[nxt, "High"])
            l2 = float(df.loc[nxt, "Low"])

            # weekly exit
            if bool(sig.loc[t, "w_exit_regime"]):
                qty = float(pos.shares)
                px = o2 * (1 - slip)
                proceeds = pos.shares * px * (1 - fee)
                cash += proceeds
                risk_ref = float(pos.orig_shares) * float(pos.R)  # TL cinsinden trade risk bütçesi
                cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)  # TL pnl (leg)
                r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                

                day_r += r_pnl
                week_r += r_pnl
                log_trade(nxt, sym, "WeeklyExit", px, pos.shares, r_pnl)
                positions.pop(sym, None)
                continue

            # Unified intrabar resolution (same as live/backtest):
            # closest-to-open first, tie-break STOP (conservative)
            tp1 = pos.entry_px + bp.tp1_R * pos.R
            tp2 = pos.entry_px + bp.tp2_R * pos.R

            evt, lvl, _extra = resolve_intrabar_exit(
                o2=o2,
                h2=h2,
                l2=l2,
                stop_lvl=float(pos.stop_px),
                tp1_lvl=float(tp1),
                tp2_lvl=float(tp2),
                tp1_done=bool(pos.tp1),
                tp2_done=bool(pos.tp2),
            )

            if evt == "STOP":
                qty = float(pos.shares)
                px = float(lvl) * (1 - slip)
                proceeds = pos.shares * px * (1 - fee)
                cash += proceeds
                risk_ref = float(pos.orig_shares) * float(pos.R)  # TL cinsinden trade risk bütçesi
                cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)  # TL pnl (leg)
                r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                
                day_r += r_pnl
                week_r += r_pnl
                log_trade(nxt, sym, "Stop", px, pos.shares, r_pnl)
                positions.pop(sym, None)
                continue

            if evt == "TP1":
                qty = min(pos.orig_shares / 3.0, pos.shares)
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    proceeds = qty * px * (1 - fee)
                    cash += proceeds
                    risk_ref = float(pos.orig_shares) * float(pos.R)  # TL cinsinden trade risk bütçesi
                    cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)  # TL pnl (leg)
                    r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0

                    day_r += r_pnl
                    week_r += r_pnl
                    pos.shares -= qty
                    pos.tp1 = True
                    pos.stop_px = pos.entry_px
                    log_trade(nxt, sym, "TP1", px, qty, r_pnl)

                if pos.shares <= 1e-12:
                    positions.pop(sym, None)
                else:
                    positions[sym] = pos
                continue

            if evt == "TP2":
                qty = min(pos.orig_shares / 3.0, pos.shares)
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    proceeds = qty * px * (1 - fee)
                    cash += proceeds
                    risk_ref = float(pos.orig_shares) * float(pos.R)  # TL cinsinden trade risk bütçesi
                    cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)  # TL pnl (leg)
                    r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0

                    day_r += r_pnl
                    week_r += r_pnl
                    pos.shares -= qty
                    pos.tp2 = True
                    log_trade(nxt, sym, "TP2", px, qty, r_pnl)

                if pos.shares <= 1e-12:
                    positions.pop(sym, None)
                else:
                    positions[sym] = pos
                continue

            # no exit on this bar
            if pos.shares <= 1e-12:
                positions.pop(sym, None)
            else:
                positions[sym] = pos

        # entries: build candidate list at t, fill next open
        killsw = (day_r <= -pparams.daily_stop_R) or (week_r <= -pparams.weekly_stop_R)

        cap = pparams.max_open - len(positions)
        if (cap > 0) and (not killsw):

            cands: List[str] = []
            for sym in tickers:
                if sym in positions:
                    continue
                df = price_map[sym]
                sig = sig_map[sym]
                if t in sig.index and bool(sig.loc[t, "entry_signal"]):
                    if t in df.index:
                        adv20 = safe_float(df.loc[t, "ADV20"], np.nan)
                        if np.isfinite(adv20) and adv20 >= pparams.adv20_min:
                            cands.append(sym)

            if cands:
                # rank cands by composite score
                rows = []
                for sym in cands:
                    df = price_map[sym]
                    sig = sig_map[sym]
                    adv20 = safe_float(df.loc[t, "ADV20"], np.nan)
                    rsi14 = safe_float(sig.loc[t, "d_rsi14"], np.nan)
                    mom20 = safe_float(sig.loc[t, "mom20"], np.nan)
                    rows.append((sym, adv20, rsi14, mom20))

                tmp = pd.DataFrame(rows, columns=["ticker", "adv20", "rsi", "mom"]).set_index("ticker")
                tmp["log_adv"] = np.log(tmp["adv20"].clip(lower=1.0))
                tmp["z_rsi"] = zscore(tmp["rsi"])
                tmp["z_mom"] = zscore(tmp["mom"])
                tmp["z_adv"] = zscore(tmp["log_adv"])

                scores: Dict[str, float] = {}
                for sym in tmp.index:
                    s_model = model_score_map.get(sym, 0.0)
                    scores[sym] = float(
                        pparams.w_model * s_model
                        + pparams.w_rsi * float(tmp.loc[sym, "z_rsi"])
                        + pparams.w_mom * float(tmp.loc[sym, "z_mom"])
                        + pparams.w_adv * float(tmp.loc[sym, "z_adv"])
                    )

                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                selected = [s for s, _ in ranked[:cap]]

                # diversification slot sizing (cap per position)
                eq_now = float(equity_rows[-1][1])
                slot = eq_now / max(1, pparams.max_open)

                for sym in selected:
                    df = price_map[sym]
                    sig = sig_map[sym]
                    _, bp = best_cfg_map[sym]

                    if nxt not in df.index or t not in df.index or t not in sig.index:
                        continue

                    entry_px = float(df.loc[nxt, "Open"]) * (1 + slip)

                    swing_low = float(df["Low"].loc[:t].tail(bp.swing_lookback).min())
                    stop_swing = swing_low * (1 - bp.swing_buffer)
                    stop_atr = float(sig.loc[t, "d_ema20"] - (bp.atr_stop_mult * sig.loc[t, "atr14"]))
                    stop_px = float(stop_atr if bp.use_atr_stop else stop_swing)

                    R = entry_px - stop_px
                    if not np.isfinite(R) or R <= 0:
                        continue

                    # --- Risk-based position sizing (% risk of equity) ---
                    risk_amt = eq_now * float(pparams.risk_pct)
                    shares_risk = np.floor(risk_amt / R)

                    if (not np.isfinite(shares_risk)) or shares_risk < 1:
                        continue

                    # cap by diversification slot (avoid concentration)
                    shares_slot = np.floor(slot / entry_px)
                    if (not np.isfinite(shares_slot)) or shares_slot < 1:
                        continue

                    max_notional = eq_now * pparams.max_notional_pct
                    shares_notional = np.floor(max_notional / entry_px)
                    if (not np.isfinite(shares_notional)) or shares_notional < 1:
                        continue

                    # cap by cash affordability (include fee)
                    cash_buffer = eq_now * pparams.min_cash_buffer_pct
                    available_cash = max(0.0, cash - cash_buffer)
                    shares_cash = np.floor(available_cash / (entry_px * (1.0 + fee)))

                    if (not np.isfinite(shares_cash)) or shares_cash < 1:
                        continue

                    shares = float(min(shares_risk, shares_slot, shares_notional, shares_cash))
                    if shares < 1:
                        continue

                    notional = shares * entry_px
                    fee_cost = notional * fee
                    total_cost = notional + fee_cost
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    positions[sym] = _Pos(
                        entry_px=entry_px,
                        stop_px=stop_px,
                        R=R,
                        shares=shares,
                        orig_shares=shares,
                    )
                    log_trade(nxt, sym, "ENTRY", entry_px, shares, 0.0)

    eqdf = pd.DataFrame(equity_rows, columns=["Date", "Equity", "Cash", "Npos"]).set_index("Date")
    trdf = pd.DataFrame(
    trades,
    columns=["Date", "Ticker", "Type", "Px", "Shares", "Notional", "R_PnL"]
)
        # --- R summary ---
    rsum = r_summary(trdf)

    if rsum:
        pd.Series(rsum).to_csv(outdir / "r_summary.csv", header=False)
        # küçük bir “gözle kontrol” için
        print("\n=== R SUMMARY (exits) ===")
        for k, v in rsum.items():
            if "rate" in k:
                print(f"{k:18s}: {v*100:6.2f}%")
            else:
                print(f"{k:18s}: {v: .4f}")

    eqdf.to_csv(outdir / "equity_curve.csv")
    trdf.to_csv(outdir / "trades.csv", index=False)

    plot_equity(eqdf[["Equity"]], outdir / "equity.png", title="Portfolio Equity")

    return {"equity_curve": eqdf, "trades": trdf}

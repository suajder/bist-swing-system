from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bist_swing.liquidity_shock import add_liquidity_shock_cols

from .backtest import BacktestParams
from .live_events import resolve_intrabar_exit
from .reporting import plot_equity
from .logger import setup_logger

logger = setup_logger("portfolio")
from bist_swing.institutional_momentum import add_institutional_momentum_cols
from .signals import SignalEngine, SignalParams
from .indicators import zscore, ema, atr

# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class PortfolioParams:
    # portfolio constraints
    max_open: int = 5

    # starting capital
    initial_equity: float = 50_000.0

    # liquidity filter (20d avg traded value)
    adv20_min: float = 50_000_000.0

    # risk-based position sizing
    risk_pct: float = 0.02  # base risk (balanced)

    # hard kill-switch (portfolio-level)
    daily_stop_R: float = 3.0
    weekly_stop_R: float = 6.0

    # weekly partial logic
    weekly_partial_fraction: float = 0.50
    weekly_partial_min_r: float = 1.0
    # soft throttling
    throttle_w1_R: float = 2.0
    throttle_w2_R: float = 4.0
    throttle_risk1: float = 0.015
    throttle_risk2: float = 0.010
    throttle_block_R: float = 5.0

    # concentration + cash buffer
    max_notional_pct: float = 0.35
    min_cash_buffer_pct: float = 0.05

    # ranking weights
    w_model: float = 1.0
    w_rsi: float = 0.15
    w_mom: float = 0.10
    w_adv: float = 0.05

    # trend filter
    trend_fast: int = 20
    trend_slow: int = 50
    min_trend_spread_pct: float = 0.01

    # ATR filter
    atr_n: int = 14
    min_atr_pct: float = 0.010
    max_atr_pct: float = 0.060

    # market regime
    max_index_vol20: float = 0.03

    # filter switches 
    use_market_regime: bool = True
    use_liquidity_shock = True
    use_rs_filter: bool = False
    use_trend_strength_filter: bool = False
    use_price_gap_filter: bool = False

    # relative strength filter
    rs_lookback: int = 60
    min_rs_excess: float = 0.00

    # sanity filters
    min_price: float = 10.0
    max_gap_pct: float = 0.05

    # re-entry cooldown after STOP (days)
    stop_cooldown_days: int = 10

    use_vol_contraction_filter: bool = True
    vol_contract_lookback: int = 10
    max_atr_ratio: float = 0.85 

    capital_filter_lookback = 50
    max_loss_streak = 5.0
    max_dd_limit_R = 20.0

    use_institutional_momentum = True
    min_inst_mom_score = 0.0

    # =========================
    # CAPITAL CURVE FILTER
    # =========================
    capital_filter_lookback: int = 20

    # =========================
    # LOSS STREAK THROTTLE
    # =========================
    max_loss_streak: int = 3

    # =========================
    # ADAPTIVE DD GUARD
    # =========================
    max_dd_limit_R: float = 3.0

    w_inst = 0.25

# ============================================================
# Utils
# ============================================================

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# zscore removed from here, imported from indicators


def add_trend_cols(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["Close"], int(fast))
    out["ema_slow"] = ema(out["Close"], int(slow))
    out["trend_up"] = out["ema_fast"] > out["ema_slow"]
    out["trend_spread_pct"] = out["ema_fast"] / out["ema_slow"] - 1.0
    out["EMA20"] = ema(out["Close"], 20)
    return out


def add_atr_cols(df: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df.copy()
    out["ATR"] = atr(out, int(n))
    out["atr_pct"] = out["ATR"] / out["Close"]
    return out

def add_vol_contraction_cols(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    out = df.copy()

    atr_mean = out["ATR"].rolling(int(lookback)).mean()
    out["atr_ratio"] = out["ATR"] / atr_mean
    out["vol_contract_ok"] = out["atr_ratio"] <= 1.0

    return out

def add_rs_filter_cols(df: pd.DataFrame, bench_close: pd.Series, lookback: int) -> pd.DataFrame:
    out = df.copy()

    stock_ret = out["Close"] / out["Close"].shift(int(lookback)) - 1.0
    bench_ret = bench_close / bench_close.shift(int(lookback)) - 1.0
    bench_ret = bench_ret.reindex(out.index)

    out["rs_stock_ret"] = stock_ret
    out["rs_bench_ret"] = bench_ret
    out["rs_excess"] = out["rs_stock_ret"] - out["rs_bench_ret"]
    return out


def r_summary(trdf: pd.DataFrame) -> Dict[str, float]:
    """
    Compute R-based performance summary from trade log.
    Summarizes EXIT legs (Type != ENTRY).
    """
    if trdf.empty or "R_PnL" not in trdf.columns or "Type" not in trdf.columns:
        return {}

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

    cum = r.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd_r = float(dd.min())

    loss_mask = (r < 0).to_numpy()
    max_streak = 0
    cur = 0
    for v in loss_mask:
        if v:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    min_ending = 0.0
    min_so_far = 0.0
    for x in r.to_numpy():
        min_ending = min(0.0, min_ending + float(x))
        min_so_far = min(min_so_far, min_ending)
    max_consec_loss_r = float(min_so_far)

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


def effective_risk_pct(
    pp: PortfolioParams,
    week_r: float,
    day_r: float,
    dd_r: float,
) -> float:
    rp = float(pp.risk_pct)

    # weekly throttle
    if week_r <= -1.0:
        rp = min(rp, 0.015)
    if week_r <= -2.0:
        rp = min(rp, 0.010)
    if week_r <= -3.0:
        return 0.0

    # daily throttle
    if day_r <= -1.0:
        rp = min(rp, 0.0125)
    if day_r <= -2.0:
        rp = min(rp, 0.0100)
    if day_r <= -2.5:
        return 0.0

    # drawdown throttle
    if dd_r <= -2.0:
        rp = min(rp, 0.015)
    if dd_r <= -4.0:
        rp = min(rp, 0.0100)
    if dd_r <= -6.0:
        rp = min(rp, 0.0075)
    if dd_r <= -8.0:
        return 0.0

    return float(max(0.0, rp))

# ============================================================
# Position
# ============================================================

@dataclass
class _Pos:
    trade_id: int
    entry_px: float
    stop_px: float
    R: float  # TL/share
    shares: float
    orig_shares: float
    tp1: bool = False
    tp2: bool = False
    weekly_partial_done: bool = False

# ============================================================
# Portfolio Backtest
# ============================================================

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

    # benchmark for RS filter
    bench_close = None
    if "XU100.IS" in price_map:
        bench_close = price_map["XU100.IS"]["Close"].astype(float)

    # enrich price frames
    for sym in tickers:
        df = price_map[sym].copy()
        if "ADV20" not in df.columns:
            df["ADV20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

        df = add_trend_cols(df, pparams.trend_fast, pparams.trend_slow)
        df = add_atr_cols(df, pparams.atr_n)
        df = add_vol_contraction_cols(df, pparams.vol_contract_lookback)
        df = add_institutional_momentum_cols(df)
        df["atr_ok"] = (df["atr_pct"] >= float(pparams.min_atr_pct)) & (df["atr_pct"] <= float(pparams.max_atr_pct))
        df = add_liquidity_shock_cols(df)

        if bench_close is not None and sym != "XU100.IS":
            df = add_rs_filter_cols(df, bench_close, pparams.rs_lookback)
        else:
            df["rs_excess"] = np.nan

        price_map[sym] = df

    # market regime filter (XU100)
    market_regime_series = None
    if "XU100.IS" in price_map:
        idx_df = price_map["XU100.IS"].copy()
        idx_close = idx_df["Close"].astype(float)
        idx_ema50 = ema(idx_close, 50)
        idx_ma200 = idx_close.rolling(200).mean()
        idx_ret = idx_close.pct_change()
        idx_vol20 = idx_ret.rolling(20).std()
        
        market_regime_series = idx_close > idx_ema50
    else:
        idx_close = None
        idx_ema50 = None
        idx_ma200 = None
        idx_vol20 = None

    # build signals per ticker
    sig_map = {t: se.build(price_map[t], best_cfg_map[t][0], market_regime_series) for t in tickers}

    # common calendar
    cal = price_map[tickers[0]].index
    cal = cal[cal >= start_dt]
    if end_dt is not None:
        cal = cal[cal <= end_dt]
    if len(cal) < 10:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame()}

    cash = float(pparams.initial_equity)
    positions: Dict[str, _Pos] = {}
    next_trade_id = 1
    peak_eq = float(pparams.initial_equity)

    trades: List[tuple] = []

    def log_trade(
        dt,
        ticker: str,
        trade_id: int,
        typ: str,
        px: float,
        shares: float,
        r_pnl: float = 0.0,
        r_leg: float = 0.0,
    ) -> None:
        notional = float(shares) * float(px) if np.isfinite(px) else np.nan
        trades.append(
            (
                dt,
                ticker,
                int(trade_id),
                typ,
                float(px),
                float(shares),
                float(notional),
                float(r_pnl),
                float(r_leg),
            )
        )

    day_r = 0.0
    week_r = 0.0
    cur_day = None
    cur_week = None
    equity_rows: List[tuple] = []
    last_stop_date: Dict[str, pd.Timestamp] = {}

    any_bp = best_cfg_map[tickers[0]][1]
    slip = float(any_bp.slippage_bps) / 10000.0
    fee = float(any_bp.fee_bps) / 10000.0

    for i in range(len(cal) - 1):
        t = cal[i]
        nxt = cal[i + 1]

        # market regime state at t
        market_trend_ok = True
        market_vol_ok = True

        if idx_close is not None and t in idx_close.index:
            ema50_t = safe_float(idx_ema50.loc[t], np.nan)
            close_t = safe_float(idx_close.loc[t], np.nan)

            if np.isfinite(ema50_t) and np.isfinite(close_t):
                market_trend_ok = close_t > ema50_t

            if idx_vol20 is not None:
                vol20_t = safe_float(idx_vol20.loc[t], np.nan)
                if np.isfinite(vol20_t):
                    market_vol_ok = vol20_t <= float(pparams.max_index_vol20)

        d = pd.Timestamp(t).date()
        w = pd.Timestamp(t).isocalendar().week

        if cur_day is None or d != cur_day:
            cur_day = d
            day_r = 0.0

        if cur_week is None or w != cur_week:
            cur_week = w
            week_r = 0.0

        # mark-to-market equity at close(t)
        eq = cash
        for sym, pos in positions.items():
            df = price_map[sym]
            if t in df.index:
                eq += pos.shares * float(df.loc[t, "Close"])
        equity_rows.append((t, eq, cash, len(positions)))
        peak_eq = max(peak_eq, float(eq))

        # =========================
        # Exits (executed on nxt bar)
        # =========================
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = price_map[sym]
            sig = sig_map[sym]
            _, bp = best_cfg_map[sym]

            if (t not in df.index) or (nxt not in df.index) or (t not in sig.index):
                continue

            o2 = float(df.loc[nxt, "Open"])
            h2 = float(df.loc[nxt, "High"])
            l2 = float(df.loc[nxt, "Low"])

            # weekly exit -> profit-gated partial + stop tighten
            if bool(sig.loc[t, "w_exit_regime"]) and (not pos.weekly_partial_done):
                current_r_leg = (o2 - pos.entry_px) / pos.R if (np.isfinite(pos.R) and pos.R > 0) else 0.0

                # tighten stop regardless
                ema20 = safe_float(sig.loc[t, "d_ema20"], np.nan)
                if np.isfinite(ema20):
                    pos.stop_px = max(float(pos.stop_px), float(ema20))

                # only partial if in profit enough
                if current_r_leg >= float(pparams.weekly_partial_min_r):
                    qty = float(pos.shares) * float(pparams.weekly_partial_fraction)
                    if qty > 0:
                        px = o2 * (1 - slip)
                        proceeds = qty * px * (1 - fee)
                        cash += proceeds

                        risk_ref = float(pos.orig_shares) * float(pos.R)
                        cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)
                        r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                        r_leg = (float(px) - float(pos.entry_px)) / float(pos.R) if (np.isfinite(pos.R) and pos.R > 0) else 0.0

                        day_r += r_pnl
                        week_r += r_pnl

                        pos.shares -= qty
                        pos.weekly_partial_done = True

                        log_trade(nxt, sym, pos.trade_id, "WeeklyPartial", px, qty, r_pnl, r_leg)

                if pos.shares <= 1e-12:
                    positions.pop(sym, None)
                else:
                    positions[sym] = pos
                continue

            tp1 = float(pos.entry_px) + float(bp.tp1_R) * float(pos.R)
            tp2 = float(pos.entry_px) + float(bp.tp2_R) * float(pos.R)

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
                proceeds = qty * px * (1 - fee)
                cash += proceeds

                risk_ref = float(pos.orig_shares) * float(pos.R)
                cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)
                r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                r_leg = (float(px) - float(pos.entry_px)) / float(pos.R) if (np.isfinite(pos.R) and pos.R > 0) else 0.0

                day_r += r_pnl
                week_r += r_pnl
                log_trade(nxt, sym, pos.trade_id, "Stop", px, qty, r_pnl, r_leg)

                last_stop_date[sym] = pd.Timestamp(nxt)
                positions.pop(sym, None)
                continue

            if evt == "TP1":
                qty = min(float(pos.orig_shares) / 3.0, float(pos.shares))
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    proceeds = qty * px * (1 - fee)
                    cash += proceeds

                    risk_ref = float(pos.orig_shares) * float(pos.R)
                    cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)
                    r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                    r_leg = (float(px) - float(pos.entry_px)) / float(pos.R) if (np.isfinite(pos.R) and pos.R > 0) else 0.0

                    day_r += r_pnl
                    week_r += r_pnl

                    pos.shares -= qty
                    pos.tp1 = True
                    pos.stop_px = pos.entry_px

                    log_trade(nxt, sym, pos.trade_id, "TP1", px, qty, r_pnl, r_leg)

                if pos.shares <= 1e-12:
                    positions.pop(sym, None)
                else:
                    positions[sym] = pos
                continue

            if evt == "TP2":
                qty = min(float(pos.orig_shares) / 3.0, float(pos.shares))
                if qty > 0:
                    px = float(lvl) * (1 - slip)
                    proceeds = qty * px * (1 - fee)
                    cash += proceeds

                    risk_ref = float(pos.orig_shares) * float(pos.R)
                    cash_pnl = (float(px) - float(pos.entry_px)) * float(qty)
                    r_pnl = (cash_pnl / risk_ref) if (np.isfinite(risk_ref) and risk_ref > 0) else 0.0
                    r_leg = (float(px) - float(pos.entry_px)) / float(pos.R) if (np.isfinite(pos.R) and pos.R > 0) else 0.0

                    day_r += r_pnl
                    week_r += r_pnl

                    pos.shares -= qty
                    pos.tp2 = True

                    log_trade(nxt, sym, pos.trade_id, "TP2", px, qty, r_pnl, r_leg)

                if pos.shares <= 1e-12:
                    positions.pop(sym, None)
                else:
                    positions[sym] = pos
                continue

            if pos.shares <= 1e-12:
                positions.pop(sym, None)
            else:
                positions[sym] = pos

        # =========================
        # Entries (signals at t -> fill at nxt open)
        # =========================
        hard_kill = (day_r <= -pparams.daily_stop_R) or (week_r <= -pparams.weekly_stop_R)
        soft_block = week_r <= -float(pparams.throttle_block_R)
        cap = int(pparams.max_open) - len(positions)

        # =========================
        # REGİME FILTER
        # =========================

        regime_ok = True
        if bool(pparams.use_market_regime):
                regime_ok = market_trend_ok and market_vol_ok

        # =========================
        # CAPITAL CURVE FILTER
        # =========================

        capital_ok = True
        if len(equity_rows) > int(pparams.capital_filter_lookback):
            eq_series = pd.Series([x[1] for x in equity_rows])
            ma = eq_series.rolling(int(pparams.capital_filter_lookback)).mean()

            if np.isfinite(ma.iloc[-1]):
                capital_ok = eq_series.iloc[-1] >= ma.iloc[-1]

        # =========================
        # LOSS STREAK THROTTLE
        # =========================

        loss_streak = 0
        for tr in reversed(trades):
            if tr[3] == "Stop":
                loss_streak += 1
            else:
                break

        streak_ok = loss_streak < int(pparams.max_loss_streak)

        # =========================
        # ADAPTIVE DD GUARD
        # =========================

        dd_cash = eq - peak_eq
        risk_unit = max(1e-9, eq * float(pparams.risk_pct))
        dd_r = dd_cash / risk_unit

        dd_guard_ok = True
        if dd_r <= -float(pparams.max_dd_limit_R):
            dd_guard_ok = False

        if (
            (cap > 0)
            and (not hard_kill)
            and (not soft_block)
            and regime_ok
            and capital_ok
            and streak_ok
            and dd_guard_ok
        ):
            cands: List[str] = []
            for sym in tickers:
                if sym in positions:
                    continue

                df = price_map[sym]
                sig = sig_map[sym]

                # cooldown after stop
                if sym in last_stop_date:
                    if (pd.Timestamp(t) - last_stop_date[sym]).days < int(pparams.stop_cooldown_days):
                        continue

                if (t in sig.index) and bool(sig.loc[t, "entry_signal"]):
                    if t in df.index:
                        adv20 = safe_float(df.loc[t, "ADV20"], np.nan)

                        # ATR band filter
                        atr_ok = bool(df.loc[t, "atr_ok"]) if "atr_ok" in df.columns else True
                        if "atr_ok" in df.columns and not bool(df.loc[t, "atr_ok"]):
                            continue

                        # basic trend filter
                        trend_ok = bool(df.loc[t, "trend_up"]) if "trend_up" in df.columns else True

                        inst_mom_ok = True
                        if bool(pparams.use_institutional_momentum):
                            inst_mom_ok = bool(df.loc[t, "inst_mom_ok"])

                        # trend-strength filter (optional)
                        trend_spread = safe_float(df.loc[t, "trend_spread_pct"], np.nan)
                        trend_strength_ok = True
                        if bool(pparams.use_trend_strength_filter) and np.isfinite(trend_spread):
                            trend_strength_ok = trend_spread >= float(pparams.min_trend_spread_pct)

                        # relative strength filter (optional)
                        rs_excess = safe_float(df.loc[t, "rs_excess"], np.nan)
                        rs_ok = True
                        if bool(pparams.use_rs_filter) and ("rs_excess" in df.columns) and np.isfinite(rs_excess):
                            rs_ok = rs_excess >= float(pparams.min_rs_excess)

                        # price + gap sanity filter (optional)
                        close_t = safe_float(df.loc[t, "Close"], np.nan)

                        liq_ok = True
                        if pparams.use_liquidity_shock and "liq_shock" in df.columns:
                            liq_ok = bool(df.loc[t, "liq_shock"])

                        price_ok = True
                        gap_ok = True
                        if bool(pparams.use_price_gap_filter):
                            price_ok = np.isfinite(close_t) and (close_t >= float(pparams.min_price))

                            prev_close = safe_float(df.loc[t, "Close"], np.nan)
                            next_open = safe_float(df.loc[nxt, "Open"], np.nan) if nxt in df.index else np.nan
                            if np.isfinite(prev_close) and prev_close > 0 and np.isfinite(next_open):
                                gap_pct = abs(next_open / prev_close - 1.0)
                                gap_ok = gap_pct <= float(pparams.max_gap_pct)

                        # volatility contraction filter (optional)
                        vol_contract_ok = True
                        if bool(pparams.use_vol_contraction_filter):
                            atr_ratio = safe_float(df.loc[t, "atr_ratio"], np.nan)
                            if np.isfinite(atr_ratio):
                                vol_contract_ok = atr_ratio <= float(pparams.max_atr_ratio)

                        if (
                            np.isfinite(adv20)
                            and (adv20 >= pparams.adv20_min)
                            and atr_ok
                            and trend_ok
                            and trend_strength_ok
                            and rs_ok
                            and price_ok
                            and gap_ok
                            and vol_contract_ok
                            and inst_mom_ok
                            and liq_ok
                        ):
                            cands.append(sym)

            if cands:
                rows = []
                for sym in cands:
                    df_sym = price_map[sym]
                    sig = sig_map[sym]
                    adv20 = safe_float(df_sym.loc[t, "ADV20"], np.nan)
                    rsi14 = safe_float(sig.loc[t, "d_rsi14"], np.nan)
                    mom20 = safe_float(sig.loc[t, "mom20"], np.nan)
                    inst_val = safe_float(df_sym.loc[t, "inst_mom_score"], 0.0) if "inst_mom_score" in df_sym.columns else 0.0
                    rows.append((sym, adv20, rsi14, mom20, inst_val))

                tmp = pd.DataFrame(rows, columns=["ticker", "adv20", "rsi", "mom", "inst"]).set_index("ticker")
                tmp["log_adv"] = np.log(tmp["adv20"].clip(lower=1.0))
                tmp["z_rsi"] = zscore(tmp["rsi"])
                tmp["z_mom"] = zscore(tmp["mom"])
                tmp["z_adv"] = zscore(tmp["log_adv"])

                scores: Dict[str, float] = {}

                for sym in tmp.index:
                    s_model = float(model_score_map.get(sym, 0.0))
                    inst = float(tmp.loc[sym, "inst"])   

                    scores[sym] = float(
                        pparams.w_model * s_model
                        + pparams.w_rsi * float(tmp.loc[sym, "z_rsi"])
                        + pparams.w_mom * float(tmp.loc[sym, "z_mom"])
                        + pparams.w_adv * float(tmp.loc[sym, "z_adv"])
                        + pparams.w_inst * inst
                    )
                    
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                selected = [s for s, _ in ranked[:cap]]

                eq_now = float(equity_rows[-1][1])
                slot = eq_now / max(1, int(pparams.max_open))

                # drawdown calculation
                dd_cash = eq_now - peak_eq
                risk_unit = max(1e-9, eq_now * float(pparams.risk_pct))
                dd_r = dd_cash / risk_unit

                dyn_risk_pct = effective_risk_pct(pparams, week_r, day_r, dd_r)
                if dyn_risk_pct <= 0:
                    continue

                for sym in selected:
                    df = price_map[sym]
                    sig = sig_map[sym]
                    _, bp = best_cfg_map[sym]

                    if (nxt not in df.index) or (t not in df.index) or (t not in sig.index):
                        continue

                    entry_px = float(df.loc[nxt, "Open"]) * (1 + slip)

                    swing_low = float(df["Low"].loc[:t].tail(int(bp.swing_lookback)).min())
                    stop_swing = swing_low * (1 - float(bp.swing_buffer))

                    stop_atr = float(sig.loc[t, "d_ema20"] - (float(bp.atr_stop_mult) * float(sig.loc[t, "atr14"])))
                    stop_px = float(stop_atr if bool(bp.use_atr_stop) else stop_swing)

                    R = entry_px - stop_px
                    if (not np.isfinite(R)) or (R <= 0):
                        continue

                    risk_amt = eq_now * float(dyn_risk_pct)
                    shares_risk = np.floor(risk_amt / R)
                    if (not np.isfinite(shares_risk)) or shares_risk < 1:
                        continue

                    shares_slot = np.floor(slot / entry_px)
                    if (not np.isfinite(shares_slot)) or shares_slot < 1:
                        continue

                    max_notional = eq_now * float(pparams.max_notional_pct)
                    shares_notional = np.floor(max_notional / entry_px)
                    if (not np.isfinite(shares_notional)) or shares_notional < 1:
                        continue

                    cash_buffer = eq_now * float(pparams.min_cash_buffer_pct)
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

                    trade_id = next_trade_id
                    next_trade_id += 1

                    cash -= total_cost
                    positions[sym] = _Pos(
                        trade_id=trade_id,
                        entry_px=entry_px,
                        stop_px=stop_px,
                        R=R,
                        shares=shares,
                        orig_shares=shares,
                    )
                    log_trade(nxt, sym, trade_id, "ENTRY", entry_px, shares, 0.0, 0.0)

    eqdf = pd.DataFrame(equity_rows, columns=["Date", "Equity", "Cash", "Npos"]).set_index("Date")
    trdf = pd.DataFrame(
        trades,
        columns=["Date", "Ticker", "TradeID", "Type", "Px", "Shares", "Notional", "R_PnL", "R_Leg"],
    )

    rsum = r_summary(trdf)
    if rsum:
        pd.Series(rsum).to_csv(outdir / "r_summary.csv", header=False)
        logger.info("\n=== R SUMMARY (exits) ===")
        for k, v in rsum.items():
            if "rate" in k:
                logger.info(f"{k:18s}: {v*100:6.2f}%")
            else:
                logger.info(f"{k:18s}: {v: .4f}")

    eqdf.to_csv(outdir / "equity_curve.csv")
    trdf.to_csv(outdir / "trades.csv", index=False)

    plot_equity(eqdf[["Equity"]], outdir / "equity.png", title="Portfolio Equity")
    return {"equity_curve": eqdf, "trades": trdf}
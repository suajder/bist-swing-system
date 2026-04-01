import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

st.title("📊 BIST Swing Dashboard")

# =========================
# LOAD DATA
# =========================
trades_path = Path("out/backtest/real_trades.csv")
equity_path = Path("out/backtest/equity_curve.csv")

if not trades_path.exists():
    st.warning("No backtest data found")
    st.stop()

trades = pd.read_csv(trades_path)

equity = pd.read_csv(equity_path) if equity_path.exists() else None

# =========================
# METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

if not trades.empty:
    total_trades = len(trades)
    winrate = (trades["R_result"] > 0).mean() * 100
    avg_R = trades["R_result"].mean()
    total_return = (trades["equity"].iloc[-1] / 100000 - 1) * 100

    col1.metric("Trades", total_trades)
    col2.metric("Winrate", f"{winrate:.2f}%")
    col3.metric("Avg R", f"{avg_R:.2f}")
    col4.metric("Return", f"{total_return:.2f}%")

# =========================
# EQUITY CURVE
# =========================
if equity is not None:
    st.subheader("Equity Curve")
    st.line_chart(equity["equity"])

# =========================
# TRADE TABLE
# =========================
st.subheader("Trades")
st.dataframe(trades)

# =========================
# DRAWdown
# =========================
if equity is not None:
    eq = equity["equity"]
    dd = (eq / eq.cummax() - 1) * 100

    st.subheader("Drawdown %")
    st.line_chart(dd)

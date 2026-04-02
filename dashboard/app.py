import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# STYLING & CONFIG
st.set_page_config(page_title="BIST Swing System", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #4CAF50;
}
.metric-label {
    font-size: 14px;
    color: #A0A0A0;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 BIST Swing Dashboard")

# PATHS
portfolio_dir = Path("../out/portfolio")
if not portfolio_dir.exists():
    portfolio_dir = Path("out/portfolio")

trades_path = portfolio_dir / "trades.csv"
equity_path = portfolio_dir / "equity_curve.csv"

if not trades_path.exists():
    st.warning(f"No trades data found at {trades_path}. Please run backtests first.")
    st.stop()

@st.cache_data
def load_data():
    trades = pd.read_csv(trades_path)
    # Parse dates if they exist
    if "Date" in trades.columns:
        trades["Date"] = pd.to_datetime(trades["Date"])
    
    equity = pd.read_csv(equity_path) if equity_path.exists() else None
    if equity is not None and "Date" in equity.columns:
        equity["Date"] = pd.to_datetime(equity["Date"])
    return trades, equity

trades, equity = load_data()

# Only keep exits for R metrics
exits = trades[trades["Type"] != "ENTRY"].copy()

# SIDEBAR FILTERS
st.sidebar.header("Filters")
if not exits.empty and "Date" in exits.columns:
    min_date = exits["Date"].min().date()
    max_date = exits["Date"].max().date()
    
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        exits = exits[(exits["Date"] >= start_ts) & (exits["Date"] <= end_ts)]
        if equity is not None:
            equity = equity[(equity["Date"] >= start_ts) & (equity["Date"] <= end_ts)]

tickers = st.sidebar.multiselect("Select Ticker", options=sorted(trades["Ticker"].unique()))
if tickers:
    exits = exits[exits["Ticker"].isin(tickers)]
    trades = trades[trades["Ticker"].isin(tickers)]

# METRICS CALCULATION
total_trades = len(exits)
if total_trades > 0:
    winrate = (exits["R_PnL"] > 0).mean() * 100
    avg_R = exits["R_PnL"].mean()
    total_R = exits["R_PnL"].sum()
else:
    winrate = 0.0
    avg_R = 0.0
    total_R = 0.0

total_return = 0.0
if equity is not None and not equity.empty:
    eq_series = equity["Equity"]
    start_eq = eq_series.iloc[0]
    end_eq = eq_series.iloc[-1]
    total_return = (end_eq / start_eq - 1) * 100

st.markdown("### Performance Overview")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Closed Trades", total_trades)
col2.metric("Winrate", f"{winrate:.1f}%")
col3.metric("Avg R", f"{avg_R:.2f}")
col4.metric("Total R", f"{total_R:.2f}")
col5.metric("Total Return", f"{total_return:.2f}%")

st.markdown("---")

# CHARTS
if equity is not None and not equity.empty:
    st.subheader("Equity Curve & Drawdown")
    
    # Calculate Drawdown
    eq_series = equity["Equity"]
    roll_max = eq_series.cummax()
    dd_series = (eq_series / roll_max - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["Date"], y=equity["Equity"], mode="lines", name="Equity", line=dict(color="#4CAF50", width=2)))
    
    fig.update_layout(title="Portfolio Equity", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=equity["Date"], y=dd_series, mode="lines", fill="tozeroy", name="Drawdown", line=dict(color="#F44336", width=2)))
    fig_dd.update_layout(title="Drawdown (%)", template="plotly_dark", height=250, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_dd, use_container_width=True)

col_charts1, col_charts2 = st.columns(2)

with col_charts1:
    if not exits.empty:
        st.subheader("R PnL Distribution")
        fig_hist = px.histogram(exits, x="R_PnL", nbins=30, color_discrete_sequence=["#2196F3"], template="plotly_dark")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

with col_charts2:
    if not exits.empty:
        st.subheader("Cumulative R PnL")
        exits = exits.sort_values("Date")
        exits["Cum_R"] = exits["R_PnL"].cumsum()
        fig_cum_r = px.line(exits, x="Date", y="Cum_R", template="plotly_dark", color_discrete_sequence=["#FF9800"])
        st.plotly_chart(fig_cum_r, use_container_width=True)

st.markdown("---")
st.subheader("All Trades")
st.dataframe(trades.sort_values(by="Date", ascending=False), use_container_width=True)

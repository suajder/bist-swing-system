import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

# Add root to sys.path so we can import from bist_swing
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.live_portfolio import PortfolioManager
from bist_swing.data import get_price_data

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

def render_backtest():
    st.title("📈 BIST Swing Dashboard - Backtest")

    # PATHS
    portfolio_dir = Path("../out/portfolio")
    if not portfolio_dir.exists():
        portfolio_dir = Path("out/portfolio")

    trades_path = portfolio_dir / "trades.csv"
    equity_path = portfolio_dir / "equity_curve.csv"

    if not trades_path.exists():
        st.warning(f"No trades data found at {trades_path}. Please run backtests first.")
        return

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


def render_live_portfolio():
    st.title("💼 Canlı Portföy Yönetimi")
    pm = PortfolioManager()

    # --- 1. AYARLAR ---
    with st.expander("⚙️ Sermaye ve Risk Ayarları", expanded=False):
        with st.form("settings_form"):
            c1, c2, c3, c4 = st.columns(4)
            capital = c1.number_input("Güncel Kasa (TL)", value=pm.state.capital, step=1000.0)
            risk = c2.number_input("İşlem Başı Risk (%)", value=pm.state.risk_pct, step=0.1)
            fric = c3.number_input("Komisyon/Kayma (%)", value=pm.state.friction_pct * 100, step=0.01)
            m_open = c4.number_input("Max Açık İşlem", value=pm.state.max_open, step=1)
            
            submitted = st.form_submit_button("Ayarları Kaydet")
            if submitted:
                pm.update_settings(capital, risk, m_open, fric / 100.0)
                st.success("Ayarlar kaydedildi.")
                st.rerun()

    current_capital_display = pm.state.capital
    
    st.markdown(f"**Güncel Kasa**: {current_capital_display:,.2f} TL  |  **Açık İşlem Limiti**: {len(pm.state.positions)} / {pm.state.max_open}")
    st.markdown("---")

    # --- 2. GÜNÜN SİNYALLERİ ---
    st.subheader("🔔 Bugünün Sinyalleri (Aksiyon Bekleyen)")
    signals_file = Path("out/live/live_signals.csv")
    if signals_file.exists():
        sig_df = pd.read_csv(signals_file)
        if not sig_df.empty:
            st.write("Aşağıdaki sinyaller, risk kurallarınıza göre filtrelenmiştir.")
            open_slots = pm.state.max_open - len(pm.state.positions)
            st.info(f"Boş Kapasite: {open_slots} hisse")
            
            for _, row in sig_df.iterrows():
                ticker = row["ticker"]
                entry = row["entry"]
                stop = row["stop"]
                
                # Zaten portföyde mi?
                already = any(p.ticker == ticker for p in pm.state.positions)
                
                with st.container():
                    colA, colB = st.columns([3, 1])
                    with colA:
                        # Hesaplamalar (görsel bilgi için)
                        risk_amount = pm.state.capital * (pm.state.risk_pct / 100.0)
                        rps = entry - stop
                        suggested_qty = max(1, int(risk_amount / rps)) if rps > 0 else 0
                        
                        # Kullanıcının manuel müdahale edebileceği lot input'u (streamlit widget olduğu için değer değiştikçe sayfa yenilenir ve güncel değer custom_qty'ye atanır)
                        custom_qty = st.number_input(f"Alınacak Lot (Önerilen: {suggested_qty})", min_value=1, value=suggested_qty, step=1, key=f"qty_{ticker}")
                        
                        # Seçilen lot sayısına göre dinamik hesaplanan değerler
                        custom_notional = custom_qty * entry
                        custom_risk = custom_qty * rps
                        
                        st.markdown(f"**{ticker}** | Giriş: `{entry:.2f}` | Stop: `{stop:.2f}` | Risk/Hisse: `{rps:.2f}`")
                        st.caption(f"İşlem Büyüklüğü: **{custom_notional:,.2f} TL** | Riske Edilen Tutar: **{custom_risk:,.2f} TL**")
                    
                    with colB:
                        if already:
                            st.button("Zaten Portföyde", disabled=True, key=f"ds_{ticker}")
                        elif open_slots <= 0:
                            st.button("Kapasite Dolu", disabled=True, key=f"ds_{ticker}")
                        else:
                            st.write("") # Dikey hizalama için boşluk
                            if st.button("➕ Portföye Ekle", key=f"add_{ticker}", use_container_width=True):
                                try:
                                    pm.add_position(ticker, entry, stop, custom_qty=custom_qty)
                                    st.success(f"{ticker} portföye eklendi.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                st.divider()
        else:
            st.info("Bugün için sinyal bulunamadı.")
    else:
        st.info("Sinyal dosyası bulunamadı. Lütfen günlük taramayı çalıştırın.")

    # --- 3. AÇIK POZİSYONLAR ---
    st.subheader("💼 Açık Pozisyonlar ve Canlı PnL")
    if not pm.state.positions:
        st.write("Şu an açık pozisyonunuz bulunmamaktadır.")
    else:
        # PnL hesaplaması için anlık fiyatları çekelim
        open_tickers = [p.ticker for p in pm.state.positions]
        with st.spinner("Anlık fiyatlar çekiliyor..."):
            price_map = get_price_data(open_tickers)
            
        unrealized_total = 0.0
        
        for pos in pm.state.positions:
            df = price_map.get(pos.ticker)
            current_px = pos.entry_price # default
            prev_px = current_px
            if df is not None and not df.empty:
                current_px = float(df.iloc[-1]["Close"])
                if len(df) > 1:
                    prev_px = float(df.iloc[-2]["Close"])
                else:
                    prev_px = current_px
                    
            notional = pos.qty * current_px
            fric_cost = notional * pm.state.friction_pct
            
            pnl_tl = (current_px - pos.entry_price) * pos.qty - fric_cost
            pnl_pct = (current_px / pos.entry_price - 1.0) * 100
            
            # Günlük PnL Değişimi
            daily_pnl_pct = (current_px / prev_px - 1.0) * 100 if prev_px > 0 else 0.0
            
            unrealized_total += pnl_tl
            
            # Alert Mantığı
            alert_msg = ""
            rps = pos.entry_price - pos.stop_price
            tp1_target = pos.entry_price + rps
            
            if current_px <= pos.stop_price * 1.02:
                alert_msg = "🚨 <span style='color:#F44336; font-size:16px;'>**Stoba Yakın!**</span>"
            elif not pos.tp1_done and current_px >= tp1_target * 0.98:
                alert_msg = "🎯 <span style='color:#4CAF50; font-size:16px;'>**TP1 Hedefinde!**</span>"
            
            with st.container():
                st.markdown(f"### {pos.ticker}  {alert_msg}", unsafe_allow_html=True)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Maliyet", f"{pos.entry_price:.2f}")
                c2.metric("Güncel Fiyat", f"{current_px:.2f}", f"{pnl_pct:.2f}%")
                c3.metric("Günlük Değişim", f"{current_px:.2f}", f"{daily_pnl_pct:.2f}%")
                c4.metric("Lot", f"{pos.qty}")
                c5.metric("Kâr/Zarar (TL)", f"{pnl_tl:,.2f}")
                
                bc1, bc2, bc3, bc4 = st.columns(4)
                
                # TP1 Button
                if pos.tp1_done or pos.qty < 3:
                     bc1.button("TP1 (Kâr Al)", disabled=True, key=f"tp1_{pos.ticker}")
                else:
                    if bc1.button("🟢 TP1 (1/3 Sat)", key=f"tp1_{pos.ticker}"):
                        realized = pm.close_position(pos.ticker, current_px, fraction=0.333, mark_tp1=True)
                        st.success(f"{pos.ticker} TP1 alındı. Gerçekleşen Kâr/Zarar: {realized:,.2f} TL")
                        st.rerun()

                # TP2 Button
                if pos.tp2_done or pos.qty < 2:
                     bc2.button("TP2 (Kâr Al)", disabled=True, key=f"tp2_{pos.ticker}")
                else:
                    if bc2.button("🔵 TP2 (Kalanın Yarısını Sat)", key=f"tp2_{pos.ticker}"):
                        realized = pm.close_position(pos.ticker, current_px, fraction=0.5, mark_tp2=True)
                        st.success(f"{pos.ticker} TP2 alındı. Gerçekleşen Kâr/Zarar: {realized:,.2f} TL")
                        st.rerun()
                        
                # Tümüyle Kapat
                if bc4.button("🔴 Tümüyle Kapat", key=f"close_{pos.ticker}"):
                    realized = pm.close_position(pos.ticker, current_px, fraction=1.0)
                    st.success(f"{pos.ticker} pozisyonu kapatıldı. Gerçekleşen K/Z: {realized:,.2f} TL")
                    st.rerun()
                
                # Ek Alım (Maliyet Düşürme/Artırma) Expandersı
                with st.expander("➕ Pozisyona Ekleme Yap (Maliyet Güncelle)"):
                    with st.form(f"add_more_{pos.ticker}"):
                        a1, a2, a3 = st.columns(3)
                        add_qty = a1.number_input("Eklenecek Lot", min_value=1, value=1, step=1, key=f"am_q_{pos.ticker}")
                        new_px = a2.number_input("Alış Fiyatı", min_value=0.01, value=current_px, step=0.01, key=f"am_p_{pos.ticker}")
                        new_stop = a3.number_input("Yeni Stop Fiyatı", min_value=0.01, value=pos.stop_price, step=0.01, key=f"am_s_{pos.ticker}")
                        
                        if st.form_submit_button("Lot Ekle"):
                            try:
                                pm.add_to_existing_position(pos.ticker, add_qty, new_px, new_stop)
                                st.success("Pozisyon başarıyla güncellendi.")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                                
                st.divider()
                
        color = "green" if unrealized_total >= 0 else "red"
        st.markdown(f"### Toplam (Açık) Kâr/Zarar: <span style='color:{color};'>**{unrealized_total:,.2f} TL**</span>", unsafe_allow_html=True)


# ==========================================
# PAGE NAVIGATION
# ==========================================
page = st.sidebar.radio("Navigasyon", ["💼 Canlı Portföy", "📊 Backtest Raporu"])

if page == "📊 Backtest Raporu":
    render_backtest()
else:
    render_live_portfolio()

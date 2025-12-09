import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Stablecoin Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for Professional Look ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333d;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-card * { color: white !important; }
    
    .risk-stable { color: #21c354 !important; font-weight: bold; }     /* Green */
    .risk-technical { color: #ffe600 !important; font-weight: bold; } /* Yellow */
    .risk-soft { color: #ffa421 !important; font-weight: bold; }      /* Orange */
    .risk-hard { color: #ff4b4b !important; font-weight: bold; }      /* Red */
    
    .history-card {
        background-color: #262730;
        border: 1px solid #41444e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .history-card h4, .history-card p, .history-card span, .history-card div {
        color: white !important;
    }
    .history-card a { color: #4DA6FF !important; }
    
    .edu-section {
        background-color: #1e2127;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #4DA6FF;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants: Historical Context ---
DEPEG_EVENTS = [
    {
        "date": "2023-03-11",
        "event": "Silicon Valley Bank Collapse",
        "price_impact": "USDC $0.870",
        "panic_score": 5,
        "vibe": "Banking Sector Contagion",
        "description": "Circle (USDC) revealed $3.3B was stuck in failed SVB. USDC crashed to 87 cents (Hard Depeg).",
        "link": "https://www.coindesk.com/markets/2023/03/11/usdc-stablecoin-depegs-falls-to-087-amid-svb-panic/"
    },
    {
        "date": "2023-08-07",
        "event": "Curve 3Pool Imbalance",
        "price_impact": "USDT $0.997",
        "panic_score": 2,
        "vibe": "Liquidity Pool Imbalance",
        "description": "Heavy selling in DeFi's main pool (Curve) caused a wobble. Technical Depeg only.",
        "link": "https://www.coindesk.com/markets/2023/08/07/tether-cto-says-usdt-redemptions-proceeding-normally-amid-curve-3pool-imbalance/"
    },
    {
        "date": "2022-11-10",
        "event": "FTX Collapse Contagion",
        "price_impact": "USDT $0.970",
        "panic_score": 4,
        "vibe": "Systemic Market Failure",
        "description": "FTX imploded. Massive fear of contagion caused USDT to suffer a Soft Depeg.",
        "link": "https://www.cnbc.com/2022/11/10/tether-usdt-stablecoin-falls-below-1-dollar-peg-amid-crypto-market-panic.html"
    },
    {
        "date": "2022-05-12",
        "event": "Terra (LUNA) Collapse",
        "price_impact": "UST $0.000",
        "panic_score": 5,
        "vibe": "Algorithmic Failure",
        "description": "Algorithmic stablecoin UST went to zero. Trust in ALL stablecoins evaporated overnight.",
        "link": "https://www.bloomberg.com/news/articles/2022-05-12/tether-drops-below-1-in-startling-sign-of-crypto-market-strain"
    }
]

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_crypto_data(coin='USDT', days=90):
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    all_data = []
    current_time = int(datetime.now().timestamp())
    hours_needed = days * 24
    limit = 2000
    num_requests = (hours_needed // limit) + 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for batch in range(num_requests):
            status_text.text(f"Fetching {coin} data batch {batch+1}/{num_requests}...")
            params = {'fsym': coin, 'tsym': 'USD', 'limit': min(limit, hours_needed - len(all_data)), 'toTs': current_time}
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                batch_data = data['Data']['Data']
                all_data = batch_data + all_data 
                if batch_data: current_time = batch_data[0]['time']
                progress_bar.progress((batch + 1) / num_requests)
            else: break
            time.sleep(0.2)
            
        status_text.empty()
        progress_bar.empty()
        
        if not all_data: return None
        df = pd.DataFrame(all_data)
        df['Datetime'] = pd.to_datetime(df['time'], unit='s')
        df['Close'] = df['close'] 
        df = df[['Datetime', 'Close']].sort_values('Datetime')
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['Datetime'] >= cutoff_date]
        return df
    except: return None

def categorize_risk_row(row):
    """Classifies a single row into a Risk Category based on Deviation %."""
    # Convert deviation to percentage (absolute value)
    # Deviation is (Price - 1.0), so 0.999 is -0.001. Abs is 0.001. *100 is 0.1%.
    dev_pct = abs(row['Deviation']) * 100
    
    if dev_pct <= 0.1:
        return "STABLE"
    elif dev_pct <= 0.5:
        return "TECHNICAL DEPEG"
    elif dev_pct <= 1.0:
        return "SOFT DEPEG"
    else:
        return "HARD DEPEG"

def calculate_metrics(df, contamination=0.01):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility_24h'] = data['Returns'].rolling(window=24).std() * 100 
    data['Deviation'] = (data['Close'] - 1.0)
    data = data.dropna()
    
    features = ['Close', 'Volatility_24h']
    model = IsolationForest(contamination=contamination, random_state=42)
    data['Anomaly'] = model.fit_predict(data[features])
    data['Is_Anomaly'] = data['Anomaly'] == -1
    data['Volatility_7d'] = data['Returns'].rolling(window=24*7).std() * 100
    
    # --- NEW: Apply Risk Categories to the DataFrame ---
    data['Risk_Category'] = data.apply(categorize_risk_row, axis=1)
    
    return data

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Dashboard", "Education: What is Depegging?"])

# ==============================================================================
# PAGE 1: LIVE DASHBOARD
# ==============================================================================
if page == "Live Dashboard":
    st.sidebar.header("Configuration")
    selected_coin = st.sidebar.selectbox("Select Stablecoin", ["USDT", "USDC"])
    st.sidebar.caption(f"Fetching real hourly data for {selected_coin} via CryptoCompare.")
    days_to_fetch = st.sidebar.slider("Days to Analyze", 7, 365, 90)
    contamination = st.sidebar.slider("ML Sensitivity", 0.001, 0.05, 0.01, format="%.3f")
    
    if st.sidebar.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.title(f"🛡️ {selected_coin} Stability Risk Monitor")

    # Load & Process
    raw_df = fetch_crypto_data(coin=selected_coin, days=days_to_fetch)
    
    if raw_df is not None:
        st.toast(f"Data Up-to-Date: {len(raw_df)} hours loaded")
        processed_df = calculate_metrics(raw_df, contamination)
        
        latest = processed_df.iloc[-1]
        last_price = latest['Close']
        risk_status = latest['Risk_Category']
        
        # Color logic for the top card
        if risk_status == "STABLE":
            risk_color = "risk-stable"
            risk_desc = "Within normal bounds (<0.1%)"
        elif risk_status == "TECHNICAL DEPEG":
            risk_color = "risk-technical"
            risk_desc = "Exceeds redemption fee (>0.1%)"
        elif risk_status == "SOFT DEPEG":
            risk_color = "risk-soft"
            risk_desc = "Liquidity stress (>0.5%)"
        else:
            risk_color = "risk-hard"
            risk_desc = "CRITICAL: Peg broken (>1.0%)"

        # Top Cards
        col1, col2, col3, col4 = st.columns(4)
        latest_dev_pct = (last_price - 1.0) * 100
        with col1: st.metric("Current Price", f"${last_price:.4f}", f"{latest_dev_pct:.3f}% (Peg)")
        with col2: st.metric("24h Volatility", f"{latest['Volatility_24h']:.4f}%")
        with col3: st.metric("Anomalies (Period)", f"{processed_df['Is_Anomaly'].sum()}")
        with col4: 
            st.markdown(f"""
            <div class='metric-card'>
                <span style='color:#888; font-size: 0.8em'>Risk Level</span><br>
                <span class='{risk_color}' style='font-size:1.4em'>{risk_status}</span><br>
                <span style='font-size: 0.7em; color: #ccc'>{risk_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Interactive Dashboard", "History of Volatility", "Raw Data"])
        
        with tab1:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=(f"{selected_coin} Price vs Peg", "Deviation (%)", "7-Day Volatility"),
                                row_heights=[0.5, 0.25, 0.25])
            
            fig.add_trace(go.Scatter(x=processed_df['Datetime'], y=processed_df['Close'], mode='lines', name='Price', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
            
            anom = processed_df[processed_df['Is_Anomaly']]
            fig.add_trace(go.Scatter(x=anom['Datetime'], y=anom['Close'], mode='markers', name='ML Anomaly', marker=dict(color='red', size=6, symbol='x')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=processed_df['Datetime'], y=processed_df['Close']-1, mode='lines', name='Deviation', fill='tozeroy', line=dict(color='#ff7f0e')), row=2, col=1)
            fig.add_trace(go.Scatter(x=processed_df['Datetime'], y=processed_df['Volatility_7d'], mode='lines', name='Vol (7d)', line=dict(color='#9467bd')), row=3, col=1)
            
            fig.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### History of Volatility")
            for event in DEPEG_EVENTS:
                border_color = "#ff4b4b" if event['panic_score'] >= 4 else "#ffa421" if event['panic_score'] >= 3 else "#21c354"
                st.markdown(f"""
                <div class="history-card" style="border-left: 5px solid {border_color}">
                    <h4>{event['panic_icon']} {event['event']} <span style="font-size:0.8em; color:#bbb; float:right">{event['date']}</span></h4>
                    <p><i>"{event['vibe']}"</i></p>
                    <p>{event['description']}</p>
                    <hr style="margin: 5px 0; border-color: #444;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span><b>Panic Meter:</b> {'🔥' * event['panic_score']} ({event['panic_score']}/5)</span>
                        <span><b>Impact:</b> {event['price_impact']}</span>
                        <a href="{event['link']}" target="_blank" style="text-decoration:none; color: #4DA6FF;">Read More ↗</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            # Highlight the Risk Category column for clarity
            st.dataframe(
                processed_df[['Datetime', 'Close', 'Risk_Category', 'Is_Anomaly', 'Deviation', 'Volatility_24h']], 
                use_container_width=True
            )
            st.download_button(f"Download {selected_coin} CSV", processed_df.to_csv(index=False).encode('utf-8'), f"{selected_coin.lower()}_data.csv", "text/csv")
    else:
        st.error("Failed to fetch data. Please check your internet connection.")

# ==============================================================================
# PAGE 2: EDUCATION PAGE
# ==============================================================================
elif page == "Education: What is Depegging?":
    st.title("📚 Understanding Stablecoin Risk")
    
    st.markdown("""
    ### What is this project about?
    This tool is a real-time risk monitor for **Stablecoins** (specifically USDT and USDC). 
    It uses live market data and Machine Learning (Isolation Forest) to detect when these coins drift away from their promised value of $1.00.
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1. What is a Stablecoin?
        A stablecoin is a cryptocurrency designed to have a relatively stable price, typically by being pegged to a commodity or currency or having its supply regulated by an algorithm.
        
        **USDT (Tether) & USDC (Circle):** These are **Fiat-Collateralized**. For every 1 digital coin, there is supposed to be $1 USD (cash or bonds) sitting in a bank vault backing it up.
        """)
    
    with col2:
        st.markdown("""
        #### 2. What is Depegging?
        Depegging is when a stablecoin deviates from its tied value ($1.00).
        
        While small fluctuations are normal, significant drops indicate that the market has lost confidence in the coin's backing or liquidity.
        """)
        
    st.divider()
    
    st.header("📉 The 3 Types of Depegging")
    st.markdown("Not all price drops are the same. This dashboard categorizes risk into three institutional tiers:")
    
    with st.container():
        st.markdown("""
        <div class="edu-section" style="border-left-color: #ffe600;">
            <h3>⚠️ Type 1: Technical Depeg (0.1% - 0.5%)</h3>
            <p><strong>Price Range:</strong> $0.995 - $0.999</p>
            <p><strong>Cause:</strong> Normal market friction, high gas fees, or temporary liquidity imbalances in DeFi pools (like Curve).</p>
            <p><strong>Why it matters:</strong> Most issuers charge a ~0.1% fee to redeem fiat. If the price drops below $0.999, arbitrageurs usually step in to buy it back up. If they don't, it's a warning sign.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-section" style="border-left-color: #ffa421;">
            <h3>🟠 Type 2: Soft Depeg (0.5% - 1.0%)</h3>
            <p><strong>Price Range:</strong> $0.990 - $0.995</p>
            <p><strong>Cause:</strong> Significant selling pressure or "FUD" (Fear, Uncertainty, Doubt). Market makers are hesitant to step in.</p>
            <p><strong>Real Life Example:</strong> During the FTX collapse contagion (Nov 2022), USDT briefly traded in this range as traders fled to safety.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="edu-section" style="border-left-color: #ff4b4b;">
            <h3>🚨 Type 3: Hard Depeg (> 1.0%)</h3>
            <p><strong>Price Range:</strong> Below $0.990</p>
            <p><strong>Cause:</strong> Systemic failure, bank runs, or frozen reserves.</p>
            <p><strong>Real Life Example:</strong> <strong>USDC in March 2023 ($0.87)</strong>. Circle revealed $3.3B was stuck in Silicon Valley Bank, causing a massive panic sell-off.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("""
    **💡 Why this project matters:** For companies handling millions in cross-border payments (like WeWire), a "Soft Depeg" of even 0.5% can wipe out profit margins on a transaction. 
    This tool acts as a "Smoke Detector" to flag these risks before they become disasters.
    """)

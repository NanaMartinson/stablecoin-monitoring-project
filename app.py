import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest 

# --- Page Configuration ---
st.set_page_config(
    page_title="Stablecoin Monitor",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_crypto_data(coin='USDT', days=365): 
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    all_data = []
    current_time = int(datetime.now().timestamp())
    hours_needed = days * 24
    limit = 2000
    num_requests = (hours_needed // limit) + 1
    
    # --- API KEY HANDLING (SECURE) ---
    api_key = st.secrets.get("CRYPTO_API_KEY", None)
    headers = {}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    # ---------------------------------

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for batch in range(num_requests):
            status_text.text(f"Fetching {coin} data batch {batch+1}/{num_requests}...")
            
            params = {
                'fsym': coin, 
                'tsym': 'USD', 
                'limit': min(limit, hours_needed - len(all_data)), 
                'toTs': current_time
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                batch_data = data['Data']['Data']
                if not batch_data:
                    break
                
                all_data = batch_data + all_data
                current_time = batch_data[0]['time']
                
                progress_bar.progress((batch + 1) / num_requests)
            else:
                st.error(f"API Error: {data.get('Message', 'Unknown error')}")
                break
                
            time.sleep(0.2) 
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
        
    status_text.empty()
    progress_bar.empty()
    
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_peg_status_and_color(price):
    """
    Determines the visual status of the peg based on the current price.
    Returns: (Label, ColorHex)
    """
    diff = abs(price - 1.0)
    
    if diff < 0.002: # 0.998 - 1.002
        return "Stable", "#008000" # Yahoo Green
    elif diff < 0.005: # 0.995 - 0.998 OR 1.002 - 1.005
        return "Technical Depeg", "#FFD700" # Gold/Yellow
    elif diff < 0.02: # 0.98 - 0.995
        return "Soft Depeg", "#FFA500" # Orange
    else: # < 0.98 or > 1.02
        return "Hard Depeg", "#FF0000" # Red

def classify_peg_status(price):
    """Classifies for dataframe usage (text only)."""
    diff = abs(price - 1.0)
    if diff < 0.002: return "Stable"
    elif diff < 0.005: return "Technical Depeg" 
    elif diff < 0.02: return "Soft Depeg"
    else: return "Hard Depeg"

def detect_anomalies_ml(df):
    """Isolation Forest for Anomaly Detection"""
    X = df[['close']].values
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    preds = iso_forest.fit_predict(X)
    df['Is Anomaly'] = preds == -1
    return df

def process_advanced_metrics(df):
    """Adds volatility metrics and classifications."""
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. Volatility Metrics
    df['Hourly Change %'] = df['close'].pct_change() * 100
    df['Daily Volatility (24h)'] = df['close'].rolling(window=24).std()
    
    # 2. Stability Classification
    df['Peg Status'] = df['close'].apply(classify_peg_status)
    
    # 3. Anomaly Detection
    df = detect_anomalies_ml(df)
    
    return df

# --- Main App Layout ---

st.title("Stablecoin Peg Monitor")
st.markdown("Monitor the stability of major stablecoins against the USD.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    selected_coin = st.selectbox("Select Stablecoin", ["USDT", "USDC"])
    selected_days = st.slider("Timeframe (Days)", min_value=7, max_value=2000, value=30)
    st.caption(f"Showing data for the last {selected_days} days.")
    
    st.markdown("---")
    st.markdown("**About this Project**")
    st.caption("A portfolio project demonstrating API integration, data visualization, and risk analysis for fintech applications.")

# Fetch Data
with st.spinner(f"Loading data for {selected_coin}..."):
    raw_df = fetch_crypto_data(selected_coin, selected_days)

if not raw_df.empty:
    df = process_advanced_metrics(raw_df)
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Hall of Pain (History)", "Raw Data & Analysis", "Stablecoin Mechanics"])

    # --- TAB 1: Dashboard ---
    with tab1:
        # Metrics Row
        current_price = df['close'].iloc[-1]
        
        # Determine Status Label and Color
        status_label, status_color = get_peg_status_and_color(current_price)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.4f}")
        with col2:
            st.metric("Period High", f"${df['high'].max():.4f}")
        with col3:
            st.metric("Period Low", f"${df['low'].min():.4f}")
        with col4:
            # Custom HTML to match st.metric size (36px) and styling
            st.markdown(f"""
            <div>
                <div style="font-size: 14px; opacity: 0.6; margin-bottom: 0px;">Peg Risk Level</div>
                <div style="font-size: 36px; font-weight: 600; color: {status_color}; line-height: 1.2;">
                    {status_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chart 1: Price Action
        st.subheader(f"{selected_coin} / USD Price Action")
        
        fig = go.Figure()
        
        # Main Price Line (Yahoo Style: Thin line with light fill)
        fig.add_trace(go.Scatter(
            x=df['time'], 
            y=df['close'], 
            mode='lines', 
            name='Price', 
            line=dict(color='#008000', width=1.5), # Standard Finance Green
            fill='tozeroy', 
            fillcolor='rgba(0, 128, 0, 0.05)' # Very subtle green tint
        ))
        
        # Peg Line
        fig.add_trace(go.Scatter(
            x=[df['time'].iloc[0], df['time'].iloc[-1]], 
            y=[1.0, 1.0], 
            mode='lines', 
            name='Peg ($1.00)', 
            line=dict(color='gray', width=1, dash='dash')
        ))

        # ML Anomalies
        anomalies = df[df['Is Anomaly'] == True]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['time'],
                y=anomalies['close'],
                mode='markers',
                name='ML Detected Anomaly',
                marker=dict(color='#FFA500', size=5, symbol='circle-open', line=dict(width=2)),
                hovertemplate="<b>ML Anomaly</b><br>Price: $%{y:.4f}<br>Date: %{x}<extra></extra>"
            ))
        
        # Scaling
        y_min = np.percentile(df['close'], 1)  
        y_max = np.percentile(df['close'], 99) 
        y_range_min = min(0.998, y_min - 0.001)
        y_range_max = max(1.002, y_max + 0.001)

        # Yahoo Finance Style Layout
        fig.update_layout(
            height=550,
            template="plotly_white", # Clean white background
            hovermode="x unified",   # Shared hover
            
            # X-Axis: Time buttons and Crosshairs
            xaxis=dict(
                type="date",
                showgrid=True, 
                gridcolor='#f0f0f0', # Subtle grid
                showspikes=True, spikethickness=1, spikecolor='#999999', spikemode='across', # Crosshairs
                rangeslider=dict(visible=False), # Hide the bulky slider, use buttons instead
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor="#f8f9fa",
                    activecolor="#e2e6ea",
                    font=dict(color="black", size=11)
                )
            ),
            
            # Y-Axis: Right-aligned, formatted currency
            yaxis=dict(
                tickformat=".4f", 
                range=[y_range_min, y_range_max], 
                fixedrange=False,
                side="right", # Yahoo style: Axis on right
                showgrid=True, 
                gridcolor='#f0f0f0',
                showspikes=True, spikethickness=1, spikecolor='#999999', spikemode='across', # Crosshairs
            ),
            
            margin=dict(l=10, r=50, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Volatility
        st.subheader("Volatility (Rolling 24h Standard Deviation)")
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df['time'], 
            y=df['Daily Volatility (24h)'],
            mode='lines',
            name='24h Volatility',
            line=dict(color='#7209b7', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(114, 9, 183, 0.1)'
        ))
        
        fig_vol.update_layout(
            height=300,
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(
                type="date", 
                showgrid=True, gridcolor='#f0f0f0',
                showspikes=True, spikethickness=1, spikecolor='#999999', spikemode='across'
            ),
            yaxis=dict(
                tickformat=".5f",
                side="right", # Axis on right
                showgrid=True, gridcolor='#f0f0f0',
            ),
            margin=dict(l=10, r=50, t=30, b=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- TAB 2: Hall of Pain (History) ---
    with tab2:
        st.header("The Hall of Pain: Historic De-pegs")
        st.markdown("A museum of the most significant stablecoin failures and crises in history.")

        # Event 1: Terra
        st.subheader("1. The Terra (UST) Collapse")
        st.markdown("**Date:** May 2022 | **Low:** $0.00")
        col_hist1, col_hist2 = st.columns([2, 1])
        with col_hist1:
            st.markdown("""
            * **The Cause:** UST was an *algorithmic* stablecoin with no real reserves. It relied on a code-based relationship with its sister token, LUNA. When large withdrawals drained the liquidity pool, the algorithm entered a "death spiral," printing trillions of LUNA tokens to try and save the peg, causing both assets to go to zero.
            * **Who was affected:** Retail investors lost life savings ($40B+ wiped out). It triggered the bankruptcy of major crypto funds like **Three Arrows Capital**, **Celsius**, and **Voyager**.
            """)
        with col_hist2:
            st.info("📉 **Impact:** Total ecosystem collapse. Triggered 'Crypto Winter' of 2022.")
        st.markdown("[🔗 Read the post-mortem (Coindesk)](https://www.coindesk.com/learn/the-fall-of-terra-a-timeline-of-the-meteoric-rise-and-crash-of-ust-and-luna/)")
        st.divider()

        # Event 2: USDC / SVB
        st.subheader("2. The Silicon Valley Bank Crisis (USDC)")
        st.markdown("**Date:** March 2023 | **Low:** $0.87")
        col_hist3, col_hist4 = st.columns([2, 1])
        with col_hist3:
            st.markdown("""
            * **The Cause:** Circle (issuer of USDC) revealed that **$3.3 Billion** of its cash reserves were stuck in Silicon Valley Bank (SVB) when regulators seized the failed bank.
            * **Who was affected:** Panic sellers who dumped USDC at $0.88-$0.90 lost 10%+ instantly. DeFi protocols like **DAI** (which is backed by USDC) also de-pegged.
            * **The Fix:** The US Federal Reserve stepped in to guarantee all SVB deposits, restoring confidence.
            """)
        with col_hist4:
            st.warning("⚠️ **Lesson:** Even 'safe' centralized coins have banking counterparty risk.")
        st.markdown("[🔗 Read the story (The Guardian)](https://www.theguardian.com/technology/2023/mar/11/usd-coin-depeg-silicon-valley-bank-collapse)")
        st.divider()

        # Event 3: Iron Finance
        st.subheader("3. Iron Finance (The 'Mark Cuban' Rug)")
        st.markdown("**Date:** June 2021 | **Low:** $0.00")
        col_hist5, col_hist6 = st.columns([2, 1])
        with col_hist5:
            st.markdown("""
            * **The Cause:** A classic "bank run" on a partially-collateralized stablecoin. As the price of the collateral token (TITAN) fell, users panicked and redeemed IRON, creating a negative feedback loop.
            * **Who was affected:** Billionaire **Mark Cuban** famously lost money in this trade, calling for regulation afterwards. The token fell from $65 to near zero in hours.
            """)
        with col_hist6:
            st.error("📉 **Impact:** First major high-profile failure of the 'partial-collateral' model.")
        st.markdown("[🔗 Read Mark Cuban's reaction (Decrypt)](https://decrypt.co/73810/mark-cuban-hit-apparent-defi-rug-pull)")

    # --- TAB 3: Raw Data & Analysis ---
    with tab3:
        st.header("Advanced Data Analysis")
        st.markdown("Filter, analyze, and download full hourly datasets including volatility metrics and anomaly detection.")
        
        # --- Filters ---
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            status_filter = st.multiselect(
                "Filter by Stability Status",
                options=df['Peg Status'].unique(),
                default=df['Peg Status'].unique()
            )
            
        with col_filter2:
            anomaly_filter = st.checkbox("Show Anomalies Only (Isolation Forest ML)", value=False)

        # Apply Filters
        filtered_df = df.copy()
        if status_filter:
            filtered_df = filtered_df[filtered_df['Peg Status'].isin(status_filter)]
        
        if anomaly_filter:
            filtered_df = filtered_df[filtered_df['Is Anomaly'] == True]

        # Display Summary Stats of Filtered Data
        st.markdown(f"**Showing {len(filtered_df)} rows** out of {len(df)} total records.")

        # --- Data Table ---
        display_df = filtered_df[[
            'time', 'close', 'high', 'low', 'volumeto', 
            'Hourly Change %', 'Daily Volatility (24h)', 'Peg Status', 'Is Anomaly'
        ]].sort_values('time', ascending=False)

        st.dataframe(
            display_df,
            column_config={
                "time": st.column_config.DatetimeColumn("Date & Time", format="D MMM YYYY, h:mm a"),
                "close": st.column_config.NumberColumn("Close Price", format="$%.4f"),
                "high": st.column_config.NumberColumn("High", format="$%.4f"),
                "low": st.column_config.NumberColumn("Low", format="$%.4f"),
                "volumeto": st.column_config.NumberColumn("Volume ($)", format="$%.0f"),
                "Hourly Change %": st.column_config.NumberColumn("Change", format="%.4f%%"),
                "Daily Volatility (24h)": st.column_config.NumberColumn("Volatility (24h)", format="%.5f"),
                "Is Anomaly": st.column_config.CheckboxColumn("ML Anomaly", help="Detected by Isolation Forest"),
            },
            use_container_width=True,
            height=600
        )

        # --- Download Button ---
        csv_full = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv_full,
            file_name=f'{selected_coin}_full_analysis.csv',
            mime='text/csv',
        )

        st.info("""
        **Methodology Notes:**
        * **Volatility:** Calculated as the rolling standard deviation of the price over the specified window (24h, 7d, etc).
        * **Peg Status:** 'Stable' (±0.2%), 'Technical Depeg' (±0.5%), 'Soft Depeg' (±2.0%), 'Hard Depeg' (>2.0%).
        * **Anomaly (ML):** Detected using the **Isolation Forest** algorithm (Unsupervised Machine Learning). This model isolates anomalies by randomly partitioning the data; anomalies are isolated faster (fewer partitions) than normal data points.
        """)

    # --- TAB 4: Mechanics ---
    with tab4:
        st.header("Stablecoin Mechanics 101")
        
        with st.expander("1. What are Stablecoins?", expanded=True):
            st.markdown("""
            **Stablecoins** are cryptocurrencies designed to minimize volatility by pegging their value to a stable asset, typically a fiat currency like the US Dollar (USD) or a commodity like gold.
            
            They solve the primary hurdle of crypto adoption: **volatility**. While Bitcoin can swing 10% in a day, stablecoins aim to stay at exactly $1.00, making them useful for payments, settlement, and preserving wealth.
            
            **The Three Main Types:**
            * **Fiat-Collateralized:** Backed 1:1 by reserves of cash and cash equivalents (Treasury bills) held in a bank. (e.g., USDT, USDC).
            * **Crypto-Collateralized:** Backed by other cryptocurrencies, often over-collateralized to account for volatility. (e.g., DAI).
            * **Algorithmic:** No reserves. Uses code and market incentives to mint/burn tokens to maintain the peg. (e.g., TerraUSD - *Failed*).
            """)

        with st.expander("2. The Titans: USDT vs. USDC", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### USDT (Tether)")
                st.markdown("""
                * **Launched:** 2014 by Tether Limited.
                * **Status:** The "First Mover." It is the most dominant stablecoin by market cap and volume.
                * **Reputation:** Historically controversial regarding the transparency of its reserves, though it has improved reporting recently.
                * **Use Case:** Deepest liquidity. Used primarily for trading and arbitrage on offshore exchanges.
                """)
                
            with col_b:
                st.markdown("### USDC (USD Coin)")
                st.markdown("""
                * **Launched:** 2018 by Centre (Circle & Coinbase).
                * **Status:** The "Compliance King." Designed for regulated markets.
                * **Reputation:** High transparency. Monthly audits are published, and reserves are held in US-regulated financial institutions.
                * **Use Case:** DeFi collateral, corporate treasury, and institutional settlement.
                """)

        with st.expander("3. What is De-pegging?", expanded=False):
            st.markdown("""
            **De-pegging** occurs when a stablecoin's price deviates from its target value (usually $1.00).
            
            ### Types of De-pegging:
            
            1.  **Soft De-peg (Temporary):**
                * *Example:* Price hits $0.995 or $1.005.
                * *Cause:* Normal market fluctuations. Someone sells a large amount (dumping), temporarily exhausting the buy-side liquidity. Arbitrage bots usually fix this in seconds.
                
            2.  **Structural De-peg (Crisis):**
                * *Example:* Price drops to $0.88 (USDC in March 2023).
                * *Cause:* Fundamental fear that the backing reserves are missing or inaccessible. For USDC, this happened when Silicon Valley Bank collapsed, freezing $3.3B of Circle's reserves.
                
            3.  **Algorithmic Death Spiral (Collapse):**
                * *Example:* Price drops to $0.00 (TerraUSD in May 2022).
                * *Cause:* Loss of confidence in the algorithm. Once the peg breaks, the mechanism to fix it actually creates hyperinflation of the sister token (LUNA), driving value to zero.
            """)

        with st.expander("4. How De-pegging Affects the Market", expanded=False):
            st.markdown("""
            When a major stablecoin wobbles, the ripple effects are massive:
            
            * **Arbitrage:** Traders rush to buy the coin at a discount (e.g., $0.98) hoping to redeem it for $1.00 later. This buying pressure often helps restore the peg.
            * **Liquidity Crunch:** If traders fear a total collapse, they flee to "safety" (Bitcoin or Fiat), draining liquidity from DeFi pools.
            * **Contagion:** Many crypto loans use stablecoins as collateral. If the price drops, these loans get liquidated, causing cascading sell-offs across the entire market (ETH, BTC).
            * **Flight to Quality:** During the USDC de-peg, funds flowed massively into USDT. During USDT FUD (fear, uncertainty, doubt), funds flow to USDC.
            """)

        with st.expander("5. Who Cares? (Institutional Relevance)", expanded=False):
            st.markdown("""
            Why do fintechs and institutions monitor this data?
            
            * **High-Frequency Traders (HFT):** They profit from the millisecond discrepancies between $0.9999 and $1.0001.
            * **Market Makers:** They provide liquidity to exchanges. If a stablecoin de-pegs, their inventory loses value, so they need real-time alerts to pull liquidity.
            * **Payment Processors:** Companies settling cross-border payments need assurance that $1 million sent is actually worth $1 million upon receipt.
            * **Regulators:** They view stablecoins as "Systemically Important." A collapse of a major stablecoin could technically impact the real-world bond market, as stablecoin issuers are massive buyers of US Treasury bills.
            """)

else:
    st.warning("No data found. Please check your API key or try again later.")

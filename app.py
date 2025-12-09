import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go
import numpy as np

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

def calculate_risk_level(std_dev):
    if std_dev < 0.001:
        return "Low", "green", "Price is tightly pegged. Standard Deviation < 0.1%"
    elif std_dev < 0.005:
        return "Moderate", "orange", "Minor volatility detected. Standard Deviation < 0.5%"
    else:
        return "High", "red", "Significant de-pegging events detected. Standard Deviation > 0.5%"

# --- Main App Layout ---

st.title("Stablecoin Peg Monitor")
st.markdown("Monitor the stability of major stablecoins against the USD.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    selected_coin = st.selectbox("Select Stablecoin", ["USDT", "USDC"])
    selected_days = st.slider("Timeframe (Days)", min_value=7, max_value=730, value=30)
    st.caption(f"Showing data for the last {selected_days} days.")
    
    st.markdown("---")
    st.markdown("**About this Project**")
    st.caption("A portfolio project demonstrating API integration, data visualization, and risk analysis for fintech applications.")

# Fetch Data
with st.spinner(f"Loading data for {selected_coin}..."):
    df = fetch_crypto_data(selected_coin, selected_days)

if not df.empty:
    
    # Create Tabs
    tab1, tab2 = st.tabs(["Dashboard", "Stablecoin Mechanics"])

    with tab1:
        # --- Statistics & Risk ---
        current_price = df['close'].iloc[-1]
        price_std = df['close'].std()
        risk_label, risk_color, risk_desc = calculate_risk_level(price_std)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.4f}")
        with col2:
            st.metric("Period High", f"${df['high'].max():.4f}")
        with col3:
            st.metric("Period Low", f"${df['low'].min():.4f}")
        with col4:
            st.markdown(f"**Peg Risk Level**")
            st.markdown(f":{risk_color}[**{risk_label}**]", help=risk_desc)

        # --- Data Logic for Depegs ---
        depegs = df[(df['low'] < 0.995) | (df['high'] > 1.005)].copy()

        # --- Charts ---
        st.subheader(f"{selected_coin} / USD Price Action")
        
        fig = go.Figure()
        
        # 1. Main Price Line
        fig.add_trace(go.Scatter(
            x=df['time'], 
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 2. The Peg Line
        fig.add_trace(go.Scatter(
            x=[df['time'].iloc[0], df['time'].iloc[-1]],
            y=[1.0, 1.0],
            mode='lines',
            name='Peg ($1.00)',
            line=dict(color='gray', width=1, dash='dash')
        ))

        # 3. Anomaly Markers (The "UX Upgrade")
        if not depegs.empty:
            fig.add_trace(go.Scatter(
                x=depegs['time'],
                y=depegs['close'],
                mode='markers',
                name='De-peg Event',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate="<b>De-peg Event</b><br>Price: $%{y:.4f}<br>Date: %{x}<extra></extra>"
            ))
        
        # Improved Scaling logic
        y_min = np.percentile(df['close'], 1)  
        y_max = np.percentile(df['close'], 99) 
        y_range_min = min(0.998, y_min - 0.001)
        y_range_max = max(1.002, y_max + 0.001)

        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            yaxis=dict(
                tickformat=".4f",
                range=[y_range_min, y_range_max]
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Hall of Pain (De-peg Events) ---
        st.subheader("Hall of Pain: Significant Deviations")
        st.markdown("Instances where the price deviated more than 0.5% from the peg.")
        
        if not depegs.empty:
            depegs['Deviation'] = depegs.apply(lambda x: x['low'] if x['low'] < 0.995 else x['high'], axis=1)
            depegs['Type'] = depegs.apply(lambda x: 'Drop' if x['low'] < 0.995 else 'Spike', axis=1)
            
            display_cols = depegs[['time', 'Type', 'Deviation', 'volumeto']]
            
            # Format dataframe for display
            st.dataframe(
                display_cols.sort_values('time', ascending=False),
                column_config={
                    "time": st.column_config.DatetimeColumn(
                        "Date & Time",
                        format="D MMM YYYY, h:mm a",
                        step=60,
                    ),
                    "Type": "Event Type",
                    "Deviation": st.column_config.NumberColumn(
                        "Price Reached",
                        format="$%.4f"
                    ),
                    "volumeto": st.column_config.NumberColumn(
                        "Volume (USD)",
                        format="$%.2f"
                    )
                },
                use_container_width=True
            )
            
            # Download Button (Recruiter Friendly)
            csv = display_cols.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download De-peg Report (CSV)",
                data=csv,
                file_name=f'{selected_coin}_depegs.csv',
                mime='text/csv',
            )

        else:
            st.info("No significant de-peg events (>0.5%) detected in this timeframe.")

    # --- Educational Section (Tab 2) ---
    with tab2:
        st.markdown("""
        ### Understanding Stablecoin Pegs
        
        **USDT (Tether) & USDC (USD Coin)** are centralized stablecoins backed by reserves of fiat currency and cash equivalents.
        
        * **The Peg:** Ideally, 1 token always equals $1.00.
        * **De-pegging:** Occurs when market panic or liquidity issues cause the price to drift.
        * **Arbitrage:** When price dips (e.g., $0.99), traders buy it to redeem for $1.00, pushing the price back up.
        
        **Risk Factors:**
        1.  **Reserves:** Quality of assets backing the coin.
        2.  **Liquidity:** Ability to process large redemptions instantly.
        3.  **Regulatory:** Government actions affecting the issuer.
        """)

else:
    st.warning("No data found. Please check your API key or try again later.")

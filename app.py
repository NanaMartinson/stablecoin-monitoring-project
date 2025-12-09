import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Stablecoin Monitor",
    page_icon="🪙",
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
    # We look for the key in Streamlit secrets, but we don't crash if it's missing.
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
            
            # Pass headers securely here
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                batch_data = data['Data']['Data']
                if not batch_data:
                    break
                
                # Prepend data since we are fetching backwards in time
                all_data = batch_data + all_data
                current_time = batch_data[0]['time']
                
                progress_bar.progress((batch + 1) / num_requests)
            else:
                st.error(f"API Error: {data.get('Message', 'Unknown error')}")
                break
                
            time.sleep(0.2) # Rate limit kindness
            
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

# --- Main App Layout ---

st.title("🪙 Stablecoin Peg Monitor")
st.markdown("Monitor the stability of major stablecoins against the USD.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    selected_coin = st.selectbox("Select Stablecoin", ["USDT", "USDC", "DAI", "BUSD", "TUSD"])
    selected_days = st.slider("Timeframe (Days)", min_value=7, max_value=730, value=30)
    st.caption(f"Showing data for the last {selected_days} days.")

# Fetch Data
with st.spinner(f"Loading data for {selected_coin}..."):
    df = fetch_crypto_data(selected_coin, selected_days)

if not df.empty:
    # --- Statistics ---
    current_price = df['close'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.4f}")
    with col2:
        max_price = df['high'].max()
        st.metric("Period High", f"${max_price:.4f}")
    with col3:
        min_price = df['low'].min()
        st.metric("Period Low", f"${min_price:.4f}")

    # --- Charts ---
    
    # Peg Deviation Chart
    st.subheader(f"{selected_coin} / USD Price Action")
    
    fig = go.Figure()
    
    # Price Line
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # The $1.00 Peg Line
    fig.add_trace(go.Scatter(
        x=[df['time'].iloc[0], df['time'].iloc[-1]],
        y=[1.0, 1.0],
        mode='lines',
        name='Peg ($1.00)',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        yaxis=dict(
            tickformat=".4f",
            # Auto-zoom y-axis to see peg deviations clearly
            range=[df['low'].min() * 0.999, df['high'].max() * 1.001]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Show Raw Data (Optional)
    with st.expander("View Raw Data"):
        st.dataframe(df.sort_values('time', ascending=False))

else:
    st.warning("No data found. Please check your API key or try again later.")

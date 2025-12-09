import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import IsolationForest 

# --- Page Configuration ---
st.set_page_config(
    page_title="Stablecoin Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
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
            status_text.text(f"Fetching {coin} market data: Batch {batch+1}/{num_requests}...")
            
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
                
            time.sleep(0.1) 
            
    except Exception as e:
        st.error(f"Critical Error fetching data: {e}")
        return pd.DataFrame()
        
    status_text.empty()
    progress_bar.empty()
    
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    cols = ['close', 'high', 'low', 'open', 'volumeto']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df

def get_zscore_status(z_score):
    """
    Determines status based on Statistical Deviation (Z-Score).
    Z = (Price - Mean) / StdDev
    """
    abs_z = abs(z_score)
    
    if abs_z < 1.0:
        return "Normal", "#007C5D" # Green
    elif abs_z < 2.0:
        return "Elevated", "#F4B400" # Yellow (1-2 Sigma)
    elif abs_z < 3.0:
        return "Volatile", "#E37400" # Orange (2-3 Sigma)
    else: 
        return "Extreme (3σ)", "#DB4437" # Red (>3 Sigma)

def detect_anomalies_ml(df):
    """Isolation Forest for Anomaly Detection"""
    X = df[['close', 'Daily Volatility (24h)']].values # Added Volatility to features
    
    # Contamination set to 1% to find only extreme outliers
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    preds = iso_forest.fit_predict(X)
    df['Is Anomaly'] = preds == -1
    return df

@st.cache_data(ttl=3600)
def process_advanced_metrics(df):
    """Adds quantitative finance metrics."""
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. Returns & Volatility
    df['Hourly Change %'] = df['close'].pct_change() * 100
    df['Daily Volatility (24h)'] = df['close'].rolling(window=24).std()
    
    # 2. Bollinger Bands (20 period, 2 Std Dev) - The Trader's Standard
    period = 20
    df['BB_Middle'] = df['close'].rolling(window=period).mean()
    df['BB_Std'] = df['close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    
    # 3. Z-Score (Statistical Distance from the Mean)
    # We use a shorter window (24h) to catch immediate de-pegs dynamically
    df['Z_Score'] = (df['close'] - df['close'].rolling(window=24).mean()) / df['close'].rolling(window=24).std()
    
    # 4. Anomaly Detection
    df = df.fillna(0) # Handle NaNs from rolling windows
    df = detect_anomalies_ml(df)
    
    return df

# --- Main App Layout ---

st.title("Stablecoin Monitor | Pro Desk")
st.markdown("Quantitative tracking of USD-pegged assets using Statistical Process Control (SPC).")

# Sidebar Controls
with st.sidebar:
    st.header("Desk Configuration")
    selected_coin = st.selectbox("Asset Ticker", ["USDT", "USDC"], index=1)
    
    st.subheader("Technical Overlays")
    chart_type = st.radio("Price Representation", ["Line", "Candlestick"], horizontal=True)
    show_bb = st.checkbox("Show Bollinger Bands (2σ)", value=True, help="Visualizes volatility expansion/contraction.")
    
    selected_days = st.select_slider(
        "Lookback Horizon",
        options=[7, 30, 90, 180, 365, 730, 2000],
        value=30,
        format_func=lambda x: f"{x} Days"
    )
    
    st.markdown("---")
    st.info("Uncle's Note: Always check the Z-Score. Price can lie; Math doesn't.")

# Fetch Data
with st.spinner(f"Pulling order book data for {selected_coin}..."):
    raw_df = fetch_crypto_data(selected_coin, selected_days)

if not raw_df.empty:
    df = process_advanced_metrics(raw_df)
    
    # Calculate Deltas
    latest = df.iloc[-1]
    prev_24h = df.iloc[-24] if len(df) > 24 else df.iloc[0]
    
    price_delta = latest['close'] - prev_24h['close']
    vol_delta = latest['volumeto'] - prev_24h['volumeto']
    
    # Z-Score Logic
    current_z = latest['Z_Score']
    status_label, status_color = get_zscore_status(current_z)
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Quant Dashboard", "Crisis Logs", "Raw Feed", "Risk Mechanics"])

    # --- TAB 1: Dashboard ---
    with tab1:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Mark Price", 
                f"${latest['close']:.4f}", 
                f"{price_delta:.4f} (24h)",
                delta_color="normal"
            )
        with col2:
            st.metric(
                "24h Volume", 
                f"${latest['volumeto']/1_000_000:.1f}M", 
                f"{(vol_delta/1_000_000):.1f}M"
            )
        with col3:
            st.metric(
                "Z-Score (24h)", 
                f"{current_z:.2f}σ",
                help="Standard Deviations from the 24h mean. >3.0 is a statistical anomaly."
            )
        with col4:
            # Custom Risk Badge based on Sigma
            st.markdown(f"""
            <div style="font-family: sans-serif;">
                <div style="font-size: 14px; color: #666; margin-bottom: -5px;">Stat. Risk Model</div>
                <div style="font-size: 28px; font-weight: 700; color: {status_color}; text-transform: uppercase; letter-spacing: -0.5px;">
                    {status_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- FINANCIAL CHART (SUBPLOTS) ---
        price_color = '#00897b' 
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02, 
            row_heights=[0.75, 0.25],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        # 1. Bollinger Bands (Behind Price)
        if show_bb:
            # Upper Band
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['BB_Upper'],
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            
            # Lower Band (Fill to Upper)
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['BB_Lower'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 0, 255, 0.05)', # Light Blue Haze
                name='Bollinger Bands (2σ)',
                hoverinfo='skip'
            ), row=1, col=1)

        # 2. Price Trace
        if chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['close'], 
                mode='lines', name='Price', 
                line=dict(color=price_color, width=2)
            ), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(
                x=df['time'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                name='OHLC',
                increasing_line_color=price_color, 
                decreasing_line_color='#DB4437'
            ), row=1, col=1)

        # Peg Line
        fig.add_trace(go.Scatter(
            x=[df['time'].iloc[0], df['time'].iloc[-1]], 
            y=[1.0, 1.0], mode='lines', 
            name='Peg ($1.00)', 
            line=dict(color='#333333', width=1, dash='dash')
        ), row=1, col=1)

        # ML Anomalies
        anomalies = df[df['Is Anomaly'] == True]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['time'], y=anomalies['close'],
                mode='markers', name='Algorithmic Anomaly',
                marker=dict(color='#DB4437', size=6, symbol='x-thin', line=dict(width=2)),
                hovertemplate="<b>Anomaly</b><br>Price: $%{y:.4f}<br>Z-Score: %{customdata:.2f}σ<extra></extra>",
                customdata=anomalies['Z_Score']
            ), row=1, col=1)

        # 3. Volume Trace
        colors = [ '#007C5D' if row['close'] >= row['open'] else '#DB4437' for index, row in df.iterrows() ]
        fig.add_trace(go.Bar(
            x=df['time'], y=df['volumeto'],
            name='Volume', marker_color=colors, opacity=0.6
        ), row=2, col=1)

        # Layout
        y_min = np.percentile(df['close'], 1)  
        y_max = np.percentile(df['close'], 99) 
        # Add buffer for bands
        y_range_min = min(0.998, y_min - 0.002)
        y_range_max = max(1.002, y_max + 0.002)

        fig.update_layout(
            height=650,
            template="plotly_white",
            hovermode="x unified", 
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            dragmode='pan'
        )

        # X-Axis
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
            showspikes=True, spikethickness=1, spikecolor='#888888', spikemode='across',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#f8f9fa", activecolor="#e2e6ea",
                font=dict(color="black", size=11),
                yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            row=1, col=1
        )
        fig.update_xaxes(showgrid=False, row=2, col=1)

        # Y-Axes
        fig.update_yaxes(
            tickformat=".4f", range=[y_range_min, y_range_max],
            side="right", showgrid=True, gridwidth=1, gridcolor='#E5E5E5',
            row=1, col=1
        )
        fig.update_yaxes(side="right", showgrid=False, showticklabels=False, row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: Crisis Logs ---
    with tab2:
        st.header("Crisis Post-Mortems")
        st.markdown("If you don't study the failures, you're destined to repeat them.")

        # Event 1: Terra
        st.subheader("1. The Death Spiral (UST)")
        st.markdown("**May 2022** | **Structure:** Algorithmic")
        col_hist1, col_hist2 = st.columns([3, 1])
        with col_hist1:
            st.markdown("The algorithm was reflective. As LUNA crashed, the minting mechanism Hyperinflated supply. It wasn't a bank run; it was a code run.")
        with col_hist2:
            st.error("Wiped Out")
        st.divider()

        # Event 2: USDC
        st.subheader("2. The Banking Panic (USDC)")
        st.markdown("**March 2023** | **Structure:** Fiat-Backed")
        col_hist3, col_hist4 = st.columns([3, 1])
        with col_hist3:
            st.markdown("Classic duration mismatch. Circle held cash in SVB. SVB bought long-dated bonds. Rates went up, bonds crashed, bank failed. Fed bailout saved the peg.")
        with col_hist4:
            st.success("Restored")
            
    # --- TAB 3: Raw Data ---
    with tab3:
        st.header("Raw Order Data")
        st.dataframe(df.sort_values('time', ascending=False), use_container_width=True, height=500)
        
    # --- TAB 4: Mechanics ---
    with tab4:
        st.header("Risk Vectors")
        st.info("Uncle's Rule #1: If the yield is >5% and you don't know where it comes from, YOU are the yield.")

else:
    st.warning("Data feed disconnected. Check API endpoints.")

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import time

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stablecoin Monitor | Institutional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to tighten spacing and give a "Terminal" feel
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stMetric { background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #eee; }
        h1, h2, h3 { letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LAYER
# -----------------------------------------------------------------------------

@st.cache_data(ttl=1800) # Cache for 30 mins
def fetch_market_data(ticker, lookback_days=730):
    """
    Fetches hourly OHLCV data from CryptoCompare.
    Defaults to 730 days (2 years) to ensure all time-range buttons work.
    """
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    limit = 2000
    all_data = []
    
    # Calculate end time (now)
    to_ts = int(time.time())
    
    # Calculate total hours needed
    hours_needed = lookback_days * 24
    
    # Pagination loop
    while hours_needed > 0:
        fetch_limit = min(limit, hours_needed)
        
        params = {
            "fsym": ticker,
            "tsym": "USD",
            "limit": fetch_limit,
            "toTs": to_ts,
            "api_key": st.secrets.get("CRYPTO_API_KEY", "")
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            
            if data.get("Response") != "Success":
                break
                
            batch = data["Data"]["Data"]
            if not batch:
                break
                
            all_data = batch + all_data
            
            # Update markers for next loop
            to_ts = batch[0]["time"]
            hours_needed -= len(batch)
            
            # Rate limit politeness
            time.sleep(0.05)
            
        except Exception as e:
            st.error(f"Data Feed Error: {e}")
            break
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # Clean numeric columns
    numeric_cols = ["open", "high", "low", "close", "volumeto"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Drop duplicates and sort
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    
    return df

@st.cache_data(ttl=1800)
def calculate_quant_metrics(df):
    """
    Applies quantitative finance metrics: SMA, Bollinger Bands, Z-Score, ML Anomalies.
    """
    if df.empty: return df
    
    # A. Volatility & Returns
    df["pct_change"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # B. Bollinger Bands (20, 2)
    window = 20
    df["sma_20"] = df["close"].rolling(window=window).mean()
    df["std_20"] = df["close"].rolling(window=window).std()
    df["bb_upper"] = df["sma_20"] + (2 * df["std_20"])
    df["bb_lower"] = df["sma_20"] - (2 * df["std_20"])
    
    # C. Z-Score (Statistical Distance from Mean)
    # Using a 24h rolling window for short-term anomaly detection
    df["z_score"] = (df["close"] - df["close"].rolling(24).mean()) / df["close"].rolling(24).std()
    
    # D. Machine Learning: Isolation Forest
    # Features: Price and Volatility
    df_clean = df.dropna(subset=["close", "std_20"])
    if not df_clean.empty:
        model = IsolationForest(contamination=0.005, random_state=42) # 0.5% outliers
        X = df_clean[["close", "std_20"]].values
        df.loc[df_clean.index, "is_anomaly"] = model.fit_predict(X)
        # Convert to boolean: -1 is anomaly, 1 is normal
        df["is_anomaly"] = df["is_anomaly"] == -1
    else:
        df["is_anomaly"] = False
        
    return df

# -----------------------------------------------------------------------------
# 3. UI COMPONENTS
# -----------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Terminal Config")
        
        ticker = st.selectbox("Asset", ["USDT", "USDC", "DAI", "FDUSD"], index=1)
        
        st.subheader("Overlays")
        show_bb = st.checkbox("Bollinger Bands (2σ)", value=True)
        show_anomalies = st.checkbox("ML Anomaly Detection", value=True)
        
        st.markdown("---")
        st.caption(f"Server Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC")
        
        return ticker, show_bb, show_anomalies

def render_metrics_header(df):
    latest = df.iloc[-1]
    prev = df.iloc[-25] if len(df) > 25 else df.iloc[0] # 24h ago
    
    # Price Delta
    price_delta = latest["close"] - prev["close"]
    
    # Z-Score Status
    z = latest["z_score"] if not pd.isna(latest["z_score"]) else 0.0
    if abs(z) < 1.5:
        risk_color = "#00C805" # Green
        risk_text = "NORMAL"
    elif abs(z) < 3.0:
        risk_color = "#FFA500" # Orange
        risk_text = "ELEVATED"
    else:
        risk_color = "#FF3131" # Red
        risk_text = "CRITICAL"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mark Price", f"${latest['close']:.4f}", f"{price_delta:.4f}")
    c2.metric("24h Volume", f"${latest['volumeto']/1_000_000:.1f}M")
    c3.metric("Volatility (20h)", f"{latest['std_20']:.5f}")
    
    # Custom HTML Metric for Risk
    c4.markdown(f"""
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 1px solid #eee; text-align: center;">
            <div style="font-size: 12px; color: #666;">STATISTICAL RISK</div>
            <div style="font-size: 24px; font-weight: bold; color: {risk_color};">{risk_text}</div>
        </div>
    """, unsafe_allow_html=True)

def render_main_chart(df, ticker, show_bb, show_anomalies):
    # Dynamic Y-Axis Scaling
    # We want to see the crash if it exists, otherwise zoom to peg
    y_min = df["close"].min()
    y_max = df["close"].max()
    
    # Safe range logic
    y_range_low = min(0.998, y_min - 0.001)
    y_range_high = max(1.002, y_max + 0.001)

    # Colors
    col_price = "#1f77b4"
    col_peg = "#888888"
    col_bb_fill = "rgba(31, 119, 180, 0.1)"
    col_anomaly = "#d62728"

    # Create Subplot (Price Top, Volume Bottom)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # --- 1. Price Trace (Top) ---
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["close"],
        mode="lines",
        name=f"{ticker} Price",
        line=dict(color=col_price, width=1.5),
        hovertemplate="$%{y:.4f}<extra></extra>"
    ), row=1, col=1)

    # --- 2. Bollinger Bands ---
    if show_bb:
        # Upper
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["bb_upper"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)
        # Lower (Fill)
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["bb_lower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=col_bb_fill,
            name="Bollinger (2σ)",
            hoverinfo="skip"
        ), row=1, col=1)

    # --- 3. The Peg ---
    fig.add_trace(go.Scatter(
        x=[df["time"].iloc[0], df["time"].iloc[-1]],
        y=[1.0, 1.0],
        mode="lines",
        name="Peg ($1.00)",
        line=dict(color=col_peg, width=1, dash="dash")
    ), row=1, col=1)

    # --- 4. ML Anomalies ---
    if show_anomalies:
        anomalies = df[df["is_anomaly"] == True]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["time"], y=anomalies["close"],
                mode="markers",
                name="ML Anomaly",
                marker=dict(color=col_anomaly, size=6, symbol="x"),
                hovertemplate="<b>Anomaly</b><br>%{x}<br>$%{y:.4f}<extra></extra>"
            ), row=1, col=1)

    # --- 5. Volume (Bottom) ---
    # Color volume based on price direction (Open vs Close proxy)
    # Since we have hourly data, we can approximate direction
    colors = [
        "#2ca02c" if r["close"] >= r["open"] else "#d62728"
        for _, r in df.iterrows()
    ]
    
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volumeto"],
        name="Volume",
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

    # --- Layout & Styling ---
    fig.update_layout(
        height=600,
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified", # The critical "Yahoo" feature
        dragmode="pan"
    )

    # Y-Axis Logic (Price)
    fig.update_yaxes(
        row=1, col=1,
        tickformat=".4f",
        range=[y_range_low, y_range_high],
        side="right", # Standard Financial Side
        gridcolor="#f0f0f0",
        showspikes=True, spikecolor="gray", spikethickness=1, spikemode="across"
    )

    # Y-Axis Logic (Volume)
    fig.update_yaxes(
        row=2, col=1,
        side="right",
        showgrid=False,
        showticklabels=False # Reduce clutter
    )

    # X-Axis Logic (Time)
    # Default Zoom: Last 30 Days
    zoom_start = df["time"].iloc[-1] - timedelta(days=30)
    zoom_end = df["time"].iloc[-1]

    fig.update_xaxes(
        range=[zoom_start, zoom_end],
        gridcolor="#f0f0f0",
        showspikes=True, spikecolor="gray", spikethickness=1,
        
        # Range Selector Buttons (The "Yahoo" Buttons)
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#ffffff",
            activecolor="#e6e6e6",
            font=dict(size=11),
            x=0, y=1,
            xanchor="left", yanchor="bottom" # Place buttons top-left above chart
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. CONTENT MODULES
# -----------------------------------------------------------------------------

def render_hall_of_pain():
    st.header("The Hall of Pain 📉")
    st.markdown("A catalog of major stablecoin failures.")
    
    with st.expander("Terra (UST) - May 2022", expanded=True):
        st.error("**Low: $0.00** | Algorithmic Death Spiral")
        st.write("The $40B collapse that triggered the 2022 Crypto Winter.")

    with st.expander("USDC / SVB - March 2023", expanded=True):
        st.warning("**Low: $0.87** | Banking Contagion")
        st.write("Circle had $3.3B stuck in Silicon Valley Bank. Peg restored after Fed intervention.")

    with st.expander("Iron Finance - June 2021", expanded=False):
        st.info("**Low: $0.00** | Bank Run")
        st.write("The infamous 'Titan' collapse that affected Mark Cuban.")

def render_mechanics():
    st.header("Stablecoin Mechanics ⚙️")
    st.markdown("""
    **1. Fiat-Backed (Off-Chain):** Backed 1:1 by cash/treasuries in a bank.  
    *Examples: USDC, USDT.* *Risk:* Bank failure (see SVB).

    **2. Crypto-Backed (On-Chain):** Over-collateralized by ETH/BTC.  
    *Example: DAI.* *Risk:* Market crashes leading to mass liquidations.

    **3. Algorithmic:** Maintained by code and incentives.  
    *Example: USDD, UST (Defunct).* *Risk:* Death spirals.
    """)

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    st.title("Stablecoin Monitor")
    st.markdown("**Real-time Peg Deviation & Volatility Analysis**")
    
    # 1. Sidebar Config
    ticker, show_bb, show_anomalies = render_sidebar()
    
    # 2. Fetch Data
    with st.spinner(f"Fetching 2-Year History for {ticker}..."):
        df = fetch_market_data(ticker)
        
    if df.empty:
        st.error("Unable to load data. Please check API connection.")
        return
        
    # 3. Process Metrics
    df = calculate_quant_metrics(df)
    
    # 4. Render Layout
    render_metrics_header(df)
    
    st.markdown("---")
    
    # TABS
    tab_chart, tab_pain, tab_learn, tab_data = st.tabs([
        "Terminal", "Hall of Pain", "Mechanics", "Raw Data"
    ])
    
    with tab_chart:
        render_main_chart(df, ticker, show_bb, show_anomalies)
        
    with tab_pain:
        render_hall_of_pain()
        
    with tab_learn:
        render_mechanics()
        
    with tab_data:
        st.dataframe(df.sort_values("time", ascending=False), height=500, use_container_width=True)

if __name__ == "__main__":
    main()

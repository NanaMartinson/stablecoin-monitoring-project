"""
Stablecoin Depeg Monitor Dashboard
===================================
Quantitative risk analysis for USD-pegged stablecoins.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

# --- Configuration Constants ---

class Config:
    """Centralized configuration constants."""
    
    # API Settings
    API_BASE_URL: str = "https://min-api.cryptocompare.com/data/v2/histohour"
    API_BATCH_LIMIT: int = 2000
    API_TIMEOUT: int = 15
    API_RATE_LIMIT_DELAY: float = 0.15
    CACHE_TTL: int = 3600  # 1 hour
    
    # Technical Analysis Parameters
    ZSCORE_WINDOW: int = 24  # hours
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    VOLATILITY_WINDOW: int = 24  # hours
    
    # ML Parameters
    ISOLATION_FOREST_CONTAMINATION: float = 0.01  # 1% outliers
    ISOLATION_FOREST_ESTIMATORS: int = 100
    ISOLATION_FOREST_SEED: int = 42
    
    # Chart Settings
    CHART_HEIGHT: int = 620
    CHART_PRICE_ROW_HEIGHT: float = 0.78
    CHART_VOLUME_ROW_HEIGHT: float = 0.22
    
    # Colors
    COLOR_BULLISH: str = "#00897B"  # Teal
    COLOR_BEARISH: str = "#E53935"  # Red
    COLOR_PEG_LINE: str = "#37474F"  # Dark gray
    COLOR_BB_FILL: str = "rgba(66, 165, 245, 0.12)"  # Light blue
    COLOR_BB_LINE: str = "rgba(66, 165, 245, 0.4)"
    COLOR_ANOMALY: str = "#E53935"
    
    # Risk Thresholds (Z-Score based)
    RISK_NORMAL_THRESHOLD: float = 1.0
    RISK_ELEVATED_THRESHOLD: float = 2.0
    RISK_VOLATILE_THRESHOLD: float = 3.0


class RiskLevel(Enum):
    """Risk classification based on statistical deviation."""
    NORMAL = ("Normal", "#00897B", "Market conditions within 1 std dev")
    ELEVATED = ("Elevated", "#FFA726", "Volatility between 1-2 std dev")
    VOLATILE = ("Volatile", "#F57C00", "Volatility between 2-3 std dev")
    EXTREME = ("Extreme", "#E53935", "Statistical anomaly >3 std dev")
    
    def __init__(self, label: str, color: str, description: str):
        self.label = label
        self.color = color
        self.description = description


@dataclass
class MarketMetrics:
    """Container for computed market metrics."""
    current_price: float
    price_change_24h: float
    current_volume: float
    volume_change_24h: float
    z_score: float
    risk_level: RiskLevel
    anomaly_count: int
    data_points: int


# --- Logging Setup ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Page Configuration ---

st.set_page_config(
    page_title="Stablecoin Depeg Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Core Functions ---

def classify_risk(z_score: float) -> RiskLevel:
    """Classify risk level based on Z-Score magnitude."""
    abs_z = abs(z_score)
    
    if abs_z < Config.RISK_NORMAL_THRESHOLD:
        return RiskLevel.NORMAL
    elif abs_z < Config.RISK_ELEVATED_THRESHOLD:
        return RiskLevel.ELEVATED
    elif abs_z < Config.RISK_VOLATILE_THRESHOLD:
        return RiskLevel.VOLATILE
    else:
        return RiskLevel.EXTREME


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def fetch_market_data(coin: str, days: int) -> pd.DataFrame:
    """Fetch historical hourly OHLCV data from CryptoCompare API."""
    all_data: List[Dict[str, Any]] = []
    current_timestamp = int(datetime.now().timestamp())
    hours_needed = days * 24
    num_requests = (hours_needed // Config.API_BATCH_LIMIT) + 1
    
    headers = {}
    api_key = st.secrets.get("CRYPTO_API_KEY")
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    
    progress_bar = st.progress(0, text="Initializing data feed...")
    
    try:
        for batch_num in range(num_requests):
            progress_text = f"Fetching {coin}/USD: Batch {batch_num + 1}/{num_requests}"
            progress_bar.progress((batch_num + 1) / num_requests, text=progress_text)
            
            params = {
                'fsym': coin,
                'tsym': 'USD',
                'limit': min(Config.API_BATCH_LIMIT, hours_needed - len(all_data)),
                'toTs': current_timestamp
            }
            
            response = requests.get(
                Config.API_BASE_URL,
                params=params,
                headers=headers,
                timeout=Config.API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') != 'Success':
                error_msg = data.get('Message', 'Unknown API error')
                logger.error(f"API Error: {error_msg}")
                st.error(f"API Error: {error_msg}")
                break
            
            batch_data = data.get('Data', {}).get('Data', [])
            if not batch_data:
                break
            
            all_data = batch_data + all_data
            current_timestamp = batch_data[0]['time']
            
            if batch_num < num_requests - 1:
                time.sleep(Config.API_RATE_LIMIT_DELAY)
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()
    finally:
        progress_bar.empty()
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volumeto']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close'])
    return df.sort_values('time').reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame) -> np.ndarray:
    """Detect anomalies using Isolation Forest algorithm."""
    features = df[['close', 'volatility']].copy()
    
    if features['volatility'].std() < 1e-10:
        return np.zeros(len(df), dtype=bool)
    
    features_normalized = (features - features.mean()) / (features.std() + 1e-10)
    
    iso_forest = IsolationForest(
        contamination=Config.ISOLATION_FOREST_CONTAMINATION,
        n_estimators=Config.ISOLATION_FOREST_ESTIMATORS,
        random_state=Config.ISOLATION_FOREST_SEED,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(features_normalized.values)
    return predictions == -1


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def compute_technical_indicators(_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and risk metrics."""
    df = _df.copy()
    
    # Returns & Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(
        window=Config.VOLATILITY_WINDOW,
        min_periods=1
    ).std()
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=Config.BOLLINGER_PERIOD, min_periods=1).mean()
    rolling_std = df['close'].rolling(window=Config.BOLLINGER_PERIOD, min_periods=1).std()
    
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + (Config.BOLLINGER_STD * rolling_std)
    df['bb_lower'] = rolling_mean - (Config.BOLLINGER_STD * rolling_std)
    
    # Z-Score
    zscore_mean = df['close'].rolling(window=Config.ZSCORE_WINDOW, min_periods=1).mean()
    zscore_std = df['close'].rolling(window=Config.ZSCORE_WINDOW, min_periods=1).std()
    
    df['z_score'] = np.where(
        zscore_std > 1e-10,
        (df['close'] - zscore_mean) / zscore_std,
        0.0
    )
    
    # Anomaly Detection
    df['volatility'] = df['volatility'].fillna(0)
    df['is_anomaly'] = detect_anomalies(df)
    
    # Fill remaining NaNs
    fill_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'z_score']
    for col in fill_cols:
        df[col] = df[col].ffill().bfill()
    
    return df


def compute_metrics(df: pd.DataFrame) -> MarketMetrics:
    """Extract current market metrics from processed DataFrame."""
    latest = df.iloc[-1]
    prev_24h = df.iloc[-Config.ZSCORE_WINDOW] if len(df) > Config.ZSCORE_WINDOW else df.iloc[0]
    
    z_score = float(latest['z_score'])
    
    return MarketMetrics(
        current_price=float(latest['close']),
        price_change_24h=float(latest['close'] - prev_24h['close']),
        current_volume=float(latest['volumeto']),
        volume_change_24h=float(latest['volumeto'] - prev_24h['volumeto']),
        z_score=z_score,
        risk_level=classify_risk(z_score),
        anomaly_count=int(df['is_anomaly'].sum()),
        data_points=len(df)
    )


def build_price_chart(df: pd.DataFrame, show_bollinger: bool = True) -> go.Figure:
    """Build the main price and volume chart."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[Config.CHART_PRICE_ROW_HEIGHT, Config.CHART_VOLUME_ROW_HEIGHT],
        subplot_titles=None
    )
    
    # Y-axis calculation
    price_min = df['close'].min()
    price_max = df['close'].max()
    
    if show_bollinger:
        price_min = min(price_min, df['bb_lower'].min())
        price_max = max(price_max, df['bb_upper'].max())
    
    price_range = price_max - price_min
    padding = max(price_range * 0.05, 0.001)
    y_min = min(price_min - padding, 0.997)
    y_max = max(price_max + padding, 1.003)
    
    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['bb_upper'], mode='lines',
            line=dict(width=1, color=Config.COLOR_BB_LINE),
            name='BB Upper', showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['bb_lower'], mode='lines',
            line=dict(width=1, color=Config.COLOR_BB_LINE),
            fill='tonexty', fillcolor=Config.COLOR_BB_FILL,
            name='Bollinger Bands (2 std)', hoverinfo='skip'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['bb_middle'], mode='lines',
            line=dict(width=1, color=Config.COLOR_BB_LINE, dash='dot'),
            name='BB Middle', showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
    
    # Price and Peg
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['close'], mode='lines', name='Price',
        line=dict(color=Config.COLOR_BULLISH, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Price: $%{y:.4f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[df['time'].iloc[0], df['time'].iloc[-1]], y=[1.0, 1.0],
        mode='lines', name='$1.00 Peg',
        line=dict(color=Config.COLOR_PEG_LINE, width=2, dash='dash'), hoverinfo='skip'
    ), row=1, col=1)
    
    # Anomalies
    anomalies = df[df['is_anomaly']]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['time'], y=anomalies['close'], mode='markers', name='ML Anomaly',
            marker=dict(color=Config.COLOR_ANOMALY, size=12, symbol='x', line=dict(width=2)),
            hovertemplate='<b>[ANOMALY DETECTED]</b><br>Time: %{x}<br>Price: $%{y:.4f}<extra></extra>'
        ), row=1, col=1)
    
    # Volume
    volume_colors = np.where(df['close'] >= df['open'], Config.COLOR_BULLISH, Config.COLOR_BEARISH)
    fig.add_trace(go.Bar(
        x=df['time'], y=df['volumeto'], name='Volume', marker_color=volume_colors, opacity=0.7,
        hovertemplate='<b>Volume</b>: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Layout updates
    fig.update_layout(
        height=Config.CHART_HEIGHT, template='plotly_white', hovermode='x unified',
        margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text='Price (USD)', tickformat='.4f', range=[y_min, y_max], row=1, col=1)
    fig.update_yaxes(title_text='Volume', tickformat='.2s', showgrid=False, row=2, col=1)
    
    return fig


def render_metrics_row(metrics: MarketMetrics) -> None:
    """Render the KPI metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Mark Price", value=f"${metrics.current_price:.4f}",
                 delta=f"{metrics.price_change_24h:+.4f} (24h)")
    
    with col2:
        st.metric(label="24h Volume", value=f"${metrics.current_volume/1e6:.1f}M",
                 delta=f"{metrics.volume_change_24h/1e6:.1f}M")
    
    with col3:
        st.metric(label="Z-Score", value=f"{metrics.z_score:+.2f} sigma")
    
    with col4:
        risk = metrics.risk_level
        st.markdown(f"""
        <div style="font-family: sans-serif;">
            <div style="color: #666; font-size: 14px;">Risk Classification</div>
            <div style="color: {risk.color}; font-size: 26px; font-weight: bold;">{risk.label}</div>
            <div style="color: #888; font-size: 11px;">{risk.description}</div>
        </div>
        """, unsafe_allow_html=True)


def render_education() -> None:
    """Render the Stablecoin education tab."""
    st.header("Stablecoin 101: Understanding the Peg")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("What are Stablecoins?")
        st.write("""
        Stablecoins are cryptocurrencies designed to minimize price volatility by pegging their 
        market value to an external reference, usually a fiat currency like the US Dollar (USD).
        
        They serve as a bridge between the stability of fiat currencies and the speed and 
        utility of blockchain assets.
        """)
        
        st.subheader("Why Do They Matter?")
        st.write("""
        1. **Liquidity**: They act as the primary quote currency for crypto markets (e.g., BTC/USDT).
        2. **Settlement**: They allow for instant, 24/7 cross-border settlement without banking delays.
        3. **DeFi Collateral**: They form the bedrock of lending and borrowing in Decentralized Finance.
        """)
    
    with col2:
        st.info("""
        **Key Takeaway**
        
        A stablecoin is only "stable" if the market believes 1 Token can always be redeemed for $1.00 USD. 
        Confidence is the real currency.
        """)

    st.markdown("---")
    
    st.subheader("What is Depegging?")
    st.write("""
    Depegging occurs when a stablecoin's price deviates significantly from its fixed target (e.g., $1.00).
    While minor fluctuations ($0.999 - $1.001) are normal due to liquidity flows, larger deviations 
    signal stress.
    """)
    
    st.markdown("#### Types of Depegging")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. Liquidity Depeg**")
            st.caption("Temporary | Low Risk")
            st.write("Caused by large sell orders in a specific pool (e.g., Curve 3pool). Usually arbitraged back quickly.")
        with c2:
            st.markdown("**2. Structural Depeg**")
            st.caption("Critical | High Risk")
            st.write("Caused by doubts about the reserves backing the token. If backing assets < issued tokens, a bank run occurs.")
        with c3:
            st.markdown("**3. Broken Mechanism**")
            st.caption("Terminal | Extreme Risk")
            st.write("Specific to algorithmic stablecoins. When the stabilization mechanism enters a death spiral (e.g., Terra/UST).")

    st.markdown("---")
    
    st.subheader("Why Depegging Matters")
    st.write("""
    A major stablecoin depeg is a systemic risk event. It destroys user wealth, triggers 
    cascading liquidations in DeFi protocols, and can invite harsh regulatory crackdowns. 
    Monitoring deviation (Z-Score) and volatility is the first line of defense.
    """)


def render_crisis_logs() -> None:
    """Render the historical crisis analysis tab."""
    st.header("Crisis Post-Mortems")
    st.markdown("---")
    
    st.subheader("Terra UST Collapse (May 2022)")
    st.markdown("**Type:** Algorithmic Failure")
    st.write("""
    The Terra ecosystem collapsed in a reflexive death spiral. UST's algorithmic peg required minting 
    LUNA to maintain $1.00. When confidence broke, LUNA supply hyperinflated, crushing the price 
    and destroying over $40B in value.
    """)
    st.metric("Lowest Price", "$0.006")
    
    st.markdown("---")
    
    st.subheader("USDC Banking Crisis (March 2023)")
    st.markdown("**Type:** Reserve/Collateral Panic")
    st.write("""
    Circle disclosed $3.3B of reserves were trapped in Silicon Valley Bank during its collapse. 
    Panic selling drove the price down until the Fed announced a backstop for depositors.
    """)
    st.metric("Lowest Price", "$0.87")


def render_methodology() -> None:
    """Render the methodology explanation tab."""
    st.header("Risk Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Z-Score Calculation")
        st.latex(r"Z = \frac{P_t - \mu_{24h}}{\sigma_{24h}}")
        st.write("Measures how many standard deviations the price is from its 24h mean.")
        st.write("- **|Z| > 2**: Elevated Risk")
        st.write("- **|Z| > 3**: Extreme Anomaly")
    
    with col2:
        st.subheader("Bollinger Bands")
        st.latex(r"BB = SMA_{20} \pm 2\sigma_{20}")
        st.write("Visualizes volatility context. Sustained movement outside bands indicates stress.")


# --- Main Application ---

def main():
    """Main application entry point."""
    
    st.title("Stablecoin Depeg Monitor")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        selected_coin = st.selectbox("Asset", options=["USDC", "USDT"], index=0)
        
        st.markdown("---")
        st.subheader("Chart Settings")
        show_bollinger = st.checkbox("Bollinger Bands (2 std)", value=True)
        lookback_days = st.select_slider("Lookback Period", 
                                       options=[7, 14, 30, 60, 90, 180, 365], 
                                       value=30,
                                       format_func=lambda x: f"{x}D")
    
    # Fetch Data
    with st.spinner(f"Loading {selected_coin}/USD data..."):
        raw_df = fetch_market_data(selected_coin, lookback_days)
    
    if raw_df.empty:
        st.error("Unable to load market data.")
        st.stop()
    
    df = compute_technical_indicators(raw_df)
    metrics = compute_metrics(df)
    
    # Tabs
    tab_dash, tab_edu, tab_crisis, tab_data, tab_meth = st.tabs([
        "Dashboard", 
        "Stablecoin 101", 
        "Crisis Logs", 
        "Raw Data", 
        "Methodology"
    ])
    
    with tab_dash:
        render_metrics_row(metrics)
        st.markdown("---")
        chart = build_price_chart(df, show_bollinger)
        st.plotly_chart(chart, use_container_width=True)
        
        with st.expander("Data Statistics"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Data Points", len(df))
            c2.metric("Anomalies", metrics.anomaly_count)
            c3.metric("Max Deviation", f"{df['z_score'].abs().max():.2f} sigma")

    with tab_edu:
        render_education()

    with tab_crisis:
        render_crisis_logs()
        
    with tab_data:
        st.dataframe(df.sort_values('time', ascending=False), use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False), "market_data.csv", "text/csv")
        
    with tab_meth:
        render_methodology()


if __name__ == "__main__":
    main()

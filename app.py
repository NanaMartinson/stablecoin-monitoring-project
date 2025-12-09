"""
Stablecoin Depeg Monitor Dashboard
===================================
Production-grade quantitative risk analysis for USD-pegged stablecoins.
Uses Statistical Process Control (SPC) and ML anomaly detection.

Author: Pro Desk Quant Team
Version: 2.0.0
"""

from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import logging

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
    NORMAL = ("Normal", "#00897B", "Market conditions within 1σ")
    ELEVATED = ("Elevated", "#FFA726", "Volatility between 1-2σ")
    VOLATILE = ("Volatile", "#F57C00", "Volatility between 2-3σ")
    EXTREME = ("Extreme", "#E53935", "Statistical anomaly >3σ")
    
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
    page_title="Stablecoin Depeg Monitor | Pro Desk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Core Functions ---

def classify_risk(z_score: float) -> RiskLevel:
    """
    Classify risk level based on Z-Score magnitude.
    
    Args:
        z_score: Statistical deviation from rolling mean
        
    Returns:
        RiskLevel enum with label, color, and description
    """
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
    """
    Fetch historical hourly OHLCV data from CryptoCompare API.
    
    Args:
        coin: Ticker symbol (e.g., 'USDT', 'USDC')
        days: Number of days of historical data
        
    Returns:
        DataFrame with columns: time, open, high, low, close, volumeto
    """
    all_data: list[dict] = []
    current_timestamp = int(datetime.now().timestamp())
    hours_needed = days * 24
    num_requests = (hours_needed // Config.API_BATCH_LIMIT) + 1
    
    # API key handling (optional)
    headers = {}
    api_key = st.secrets.get("CRYPTO_API_KEY")
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    
    # Progress tracking
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
            
            # Prepend batch (data comes in reverse chronological order)
            all_data = batch_data + all_data
            current_timestamp = batch_data[0]['time']
            
            # Rate limiting
            if batch_num < num_requests - 1:
                import time
                time.sleep(Config.API_RATE_LIMIT_DELAY)
                
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        st.error("Connection timed out. Please try again.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()
    finally:
        progress_bar.empty()
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    # Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volumeto']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with missing price data
    df = df.dropna(subset=['close'])
    
    return df.sort_values('time').reset_index(drop=True)


def detect_anomalies(df: pd.DataFrame) -> np.ndarray:
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Args:
        df: DataFrame with 'close' and 'volatility' columns
        
    Returns:
        Boolean array where True indicates anomaly
    """
    # Prepare features
    features = df[['close', 'volatility']].copy()
    
    # Handle edge case: insufficient variance
    if features['volatility'].std() < 1e-10:
        return np.zeros(len(df), dtype=bool)
    
    # Normalize features for better isolation
    features_normalized = (features - features.mean()) / (features.std() + 1e-10)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=Config.ISOLATION_FOREST_CONTAMINATION,
        n_estimators=Config.ISOLATION_FOREST_ESTIMATORS,
        random_state=Config.ISOLATION_FOREST_SEED,
        n_jobs=-1  # Use all cores
    )
    
    predictions = iso_forest.fit_predict(features_normalized.values)
    
    return predictions == -1


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def compute_technical_indicators(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and risk metrics.
    
    Args:
        _df: Raw OHLCV DataFrame (underscore prefix for Streamlit cache)
        
    Returns:
        DataFrame with added indicator columns
    """
    df = _df.copy()
    
    # --- Returns & Volatility ---
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(
        window=Config.VOLATILITY_WINDOW,
        min_periods=1
    ).std()
    
    # --- Bollinger Bands ---
    rolling_mean = df['close'].rolling(
        window=Config.BOLLINGER_PERIOD,
        min_periods=1
    ).mean()
    rolling_std = df['close'].rolling(
        window=Config.BOLLINGER_PERIOD,
        min_periods=1
    ).std()
    
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + (Config.BOLLINGER_STD * rolling_std)
    df['bb_lower'] = rolling_mean - (Config.BOLLINGER_STD * rolling_std)
    
    # --- Z-Score ---
    zscore_mean = df['close'].rolling(
        window=Config.ZSCORE_WINDOW,
        min_periods=1
    ).mean()
    zscore_std = df['close'].rolling(
        window=Config.ZSCORE_WINDOW,
        min_periods=1
    ).std()
    
    # Safe division
    df['z_score'] = np.where(
        zscore_std > 1e-10,
        (df['close'] - zscore_mean) / zscore_std,
        0.0
    )
    
    # --- Anomaly Detection ---
    # Fill NaN volatility before anomaly detection
    df['volatility'] = df['volatility'].fillna(0)
    df['is_anomaly'] = detect_anomalies(df)
    
    # --- Forward/Backward fill remaining NaNs ---
    fill_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'z_score']
    for col in fill_cols:
        df[col] = df[col].ffill().bfill()
    
    return df


def compute_metrics(df: pd.DataFrame) -> MarketMetrics:
    """
    Extract current market metrics from processed DataFrame.
    
    Args:
        df: Processed DataFrame with indicators
        
    Returns:
        MarketMetrics dataclass with current values
    """
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


def build_price_chart(
    df: pd.DataFrame,
    show_bollinger: bool = True
) -> go.Figure:
    """
    Build the main price and volume chart.
    
    Args:
        df: Processed DataFrame with indicators
        show_bollinger: Whether to display Bollinger Bands
        
    Returns:
        Plotly Figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[Config.CHART_PRICE_ROW_HEIGHT, Config.CHART_VOLUME_ROW_HEIGHT],
        subplot_titles=None
    )
    
    # --- Calculate Y-axis range ---
    price_min = df['close'].min()
    price_max = df['close'].max()
    
    if show_bollinger:
        price_min = min(price_min, df['bb_lower'].min())
        price_max = max(price_max, df['bb_upper'].max())
    
    # Add padding (5% each side)
    price_range = price_max - price_min
    padding = max(price_range * 0.05, 0.001)  # Minimum padding for stablecoins
    y_min = price_min - padding
    y_max = price_max + padding
    
    # Ensure $1.00 peg line is visible
    y_min = min(y_min, 0.997)
    y_max = max(y_max, 1.003)
    
    # --- Bollinger Bands (rendered first, behind price) ---
    if show_bollinger:
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_upper'],
                mode='lines',
                line=dict(width=1, color=Config.COLOR_BB_LINE),
                name='BB Upper',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_lower'],
                mode='lines',
                line=dict(width=1, color=Config.COLOR_BB_LINE),
                fill='tonexty',
                fillcolor=Config.COLOR_BB_FILL,
                name='Bollinger Bands (2σ)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Middle band (SMA)
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_middle'],
                mode='lines',
                line=dict(width=1, color=Config.COLOR_BB_LINE, dash='dot'),
                name='BB Middle',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # --- Price Line ---
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color=Config.COLOR_BULLISH, width=2),
            hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Price: $%{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # --- $1.00 Peg Reference Line ---
    fig.add_trace(
        go.Scatter(
            x=[df['time'].iloc[0], df['time'].iloc[-1]],
            y=[1.0, 1.0],
            mode='lines',
            name='$1.00 Peg',
            line=dict(color=Config.COLOR_PEG_LINE, width=2, dash='dash'),
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # --- Anomaly Markers ---
    anomalies = df[df['is_anomaly']]
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies['time'],
                y=anomalies['close'],
                mode='markers',
                name='ML Anomaly',
                marker=dict(
                    color=Config.COLOR_ANOMALY,
                    size=12,
                    symbol='x',
                    line=dict(width=2, color=Config.COLOR_ANOMALY)
                ),
                hovertemplate=(
                    '<b>⚠️ Anomaly Detected</b><br>'
                    'Time: %{x|%Y-%m-%d %H:%M}<br>'
                    'Price: $%{y:.4f}<br>'
                    'Z-Score: %{customdata:.2f}σ<extra></extra>'
                ),
                customdata=anomalies['z_score']
            ),
            row=1, col=1
        )
    
    # --- Volume Bars ---
    volume_colors = np.where(
        df['close'] >= df['open'],
        Config.COLOR_BULLISH,
        Config.COLOR_BEARISH
    )
    
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volumeto'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7,
            hovertemplate='<b>Volume</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # --- Layout ---
    fig.update_layout(
        height=Config.CHART_HEIGHT,
        template='plotly_white',
        hovermode='x unified',
        dragmode='zoom',
        margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#E0E0E0',
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False
    )
    
    # --- X-Axis (Price) ---
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#EEEEEE',
        showspikes=True,
        spikethickness=1,
        spikecolor='#9E9E9E',
        spikemode='across',
        rangeselector=dict(
            buttons=[
                dict(count=7, label='1W', step='day', stepmode='backward'),
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(step='all', label='ALL')
            ],
            bgcolor='#FAFAFA',
            activecolor='#E3F2FD',
            font=dict(color='#212121', size=11),
            yanchor='bottom',
            y=1.0,
            xanchor='right',
            x=1
        ),
        row=1, col=1
    )
    
    # --- X-Axis (Volume) ---
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#EEEEEE',
        row=2, col=1
    )
    
    # --- Y-Axis (Price) ---
    fig.update_yaxes(
        title_text='Price (USD)',
        title_font=dict(size=12),
        tickformat='.4f',
        range=[y_min, y_max],
        side='right',
        showgrid=True,
        gridwidth=1,
        gridcolor='#EEEEEE',
        zeroline=False,
        row=1, col=1
    )
    
    # --- Y-Axis (Volume) ---
    fig.update_yaxes(
        title_text='Volume',
        title_font=dict(size=12),
        side='right',
        showgrid=False,
        tickformat='.2s',
        row=2, col=1
    )
    
    return fig


def render_metrics_row(metrics: MarketMetrics) -> None:
    """Render the KPI metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Mark Price",
            value=f"${metrics.current_price:.4f}",
            delta=f"{metrics.price_change_24h:+.4f} (24h)"
        )
    
    with col2:
        volume_display = metrics.current_volume / 1_000_000
        volume_delta = metrics.volume_change_24h / 1_000_000
        st.metric(
            label="24h Volume",
            value=f"${volume_display:.1f}M",
            delta=f"{volume_delta:+.1f}M"
        )
    
    with col3:
        st.metric(
            label="Z-Score",
            value=f"{metrics.z_score:+.2f}σ",
            help="Standard deviations from 24h rolling mean. |Z| > 3 indicates statistical anomaly."
        )
    
    with col4:
        risk = metrics.risk_level
        st.markdown(f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
            <div style="font-size: 14px; font-weight: 500; color: #666; margin-bottom: 4px;">
                Risk Classification
            </div>
            <div style="font-size: 26px; font-weight: 700; color: {risk.color}; 
                        text-transform: uppercase; letter-spacing: -0.5px;">
                {risk.label}
            </div>
            <div style="font-size: 11px; color: #888; margin-top: 2px;">
                {risk.description}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_crisis_logs() -> None:
    """Render the historical crisis analysis tab."""
    st.header("Crisis Post-Mortems")
    st.caption("*Those who cannot remember the past are condemned to repeat it.* — George Santayana")
    
    st.markdown("---")
    
    # Terra UST Collapse
    with st.container():
        st.subheader("🔴 The Death Spiral — UST (Terra)")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("**May 2022** · Algorithmic Stablecoin")
            st.markdown("""
            The Terra ecosystem collapsed in a reflexive death spiral. UST's algorithmic peg 
            mechanism required minting/burning LUNA to maintain $1.00. When confidence broke:
            
            1. Large UST sells triggered LUNA minting
            2. LUNA supply hyperinflated → price collapsed
            3. Collateral value evaporated → deeper depeg
            4. Feedback loop accelerated to zero
            
            **$40B+ in value destroyed in 72 hours.**
            
            *Lesson: Reflexive stabilization mechanisms amplify stress instead of absorbing it.*
            """)
        
        with col2:
            st.error("**TERMINAL**")
            st.metric("Lowest Price", "$0.006", "-99.4%")
    
    st.markdown("---")
    
    # USDC SVB Crisis
    with st.container():
        st.subheader("🟡 The Banking Panic — USDC (Circle)")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("**March 2023** · Fiat-Collateralized Stablecoin")
            st.markdown("""
            Circle disclosed $3.3B of USDC reserves held at Silicon Valley Bank during its 
            collapse. Classic duration mismatch caused the bank run:
            
            1. SVB held long-dated bonds as assets
            2. Fed rate hikes → bond values crashed
            3. Depositors fled → liquidity crisis
            4. FDIC seizure → USDC reserve uncertainty
            
            **USDC depegged to $0.87 before Fed/Treasury backstop announcement.**
            
            *Lesson: Even "safe" fiat-backed stables carry counterparty risk through banking relationships.*
            """)
        
        with col2:
            st.success("**RESTORED**")
            st.metric("Lowest Price", "$0.87", "-13%")
    
    st.markdown("---")
    
    # USDT FUD Events
    with st.container():
        st.subheader("🟠 Persistent FUD — USDT (Tether)")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("**2017-Present** · Fiat-Collateralized Stablecoin")
            st.markdown("""
            Tether has faced recurring transparency concerns and brief depegs:
            
            - **Oct 2018**: Dropped to $0.92 amid redemption concerns
            - **May 2022**: Brief depeg during Terra contagion
            - **Ongoing**: Questions about reserve composition and audits
            
            Despite controversies, USDT maintains dominant market share and has 
            consistently returned to peg.
            
            *Lesson: Market confidence and liquidity depth matter as much as technical backing.*
            """)
        
        with col2:
            st.warning("**VOLATILE**")
            st.metric("Lowest Price", "$0.92", "-8%")


def render_methodology() -> None:
    """Render the methodology explanation tab."""
    st.header("Risk Methodology")
    
    st.info(
        "**First Principle**: If the yield is >5% and you don't know where it comes from, "
        "YOU are the yield."
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Z-Score Calculation")
        st.latex(r"Z = \frac{P_t - \mu_{24h}}{\sigma_{24h}}")
        st.markdown("""
        The Z-Score measures how many standard deviations the current price deviates 
        from its 24-hour rolling mean.
        
        | Z-Score | Probability | Classification |
        |---------|-------------|----------------|
        | |Z| < 1 | 68.3% | Normal |
        | |Z| < 2 | 95.4% | Elevated |
        | |Z| < 3 | 99.7% | Volatile |
        | |Z| ≥ 3 | 0.3% | **Extreme** |
        
        A Z-Score beyond ±3 indicates a statistically significant deviation that warrants 
        investigation.
        """)
    
    with col2:
        st.subheader("📊 Bollinger Bands")
        st.latex(r"BB_{upper/lower} = SMA_{20} \pm 2\sigma_{20}")
        st.markdown("""
        Bollinger Bands provide visual volatility context:
        
        - **Band Width**: Expanding bands = increasing volatility
        - **Band Touch**: Price at upper/lower band = overbought/oversold
        - **Band Squeeze**: Narrowing bands often precede volatility expansion
        
        For stablecoins, sustained price outside the bands signals potential depeg stress.
        """)
    
    st.markdown("---")
    
    st.subheader("🤖 Isolation Forest (Anomaly Detection)")
    st.markdown("""
    Isolation Forest is an unsupervised machine learning algorithm optimized for anomaly detection.
    
    **How it works:**
    1. Randomly select a feature and split value
    2. Recursively partition data until points are isolated
    3. Anomalies require fewer splits → shorter path length
    4. Points with short average path lengths are flagged
    
    **Features used:**
    - Closing price
    - 24-hour rolling volatility
    
    **Configuration:**
    - Contamination: 1% (expects ~1% of data to be anomalous)
    - Estimators: 100 trees for stable predictions
    """)


# --- Main Application ---

def main():
    """Main application entry point."""
    
    # Header
    st.title("📊 Stablecoin Depeg Monitor")
    st.caption("Quantitative risk analysis using Statistical Process Control (SPC)")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        selected_coin = st.selectbox(
            "Asset",
            options=["USDC", "USDT"],
            index=0,
            help="Select stablecoin to monitor"
        )
        
        st.markdown("---")
        
        st.subheader("Chart Settings")
        
        show_bollinger = st.checkbox(
            "Bollinger Bands (2σ)",
            value=True,
            help="Display 20-period Bollinger Bands"
        )
        
        lookback_days = st.select_slider(
            "Lookback Period",
            options=[7, 14, 30, 60, 90, 180, 365, 730],
            value=30,
            format_func=lambda x: f"{x}D" if x < 365 else f"{x//365}Y"
        )
        
        st.markdown("---")
        st.caption(
            f"Data: CryptoCompare API\n\n"
            f"Refresh: Hourly\n\n"
            f"Model: Isolation Forest"
        )
    
    # Fetch and process data
    with st.spinner(f"Loading {selected_coin}/USD market data..."):
        raw_df = fetch_market_data(selected_coin, lookback_days)
    
    if raw_df.empty:
        st.error("Unable to load market data. Please check your connection and try again.")
        st.stop()
    
    # Process indicators
    df = compute_technical_indicators(raw_df)
    metrics = compute_metrics(df)
    
    # Create tabs
    tab_dashboard, tab_crisis, tab_data, tab_methodology = st.tabs([
        "📈 Dashboard",
        "📚 Crisis Logs", 
        "🗃️ Raw Data",
        "🔬 Methodology"
    ])
    
    # Dashboard Tab
    with tab_dashboard:
        render_metrics_row(metrics)
        st.markdown("---")
        
        # Build and display chart
        chart = build_price_chart(df, show_bollinger=show_bollinger)
        st.plotly_chart(
            chart,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                'scrollZoom': True
            }
        )
        
        # Stats footer
        with st.expander("📊 Data Statistics"):
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Data Points", f"{metrics.data_points:,}")
            with stat_cols[1]:
                st.metric("Price Range", f"${df['close'].min():.4f} – ${df['close'].max():.4f}")
            with stat_cols[2]:
                st.metric("Anomalies Detected", metrics.anomaly_count)
            with stat_cols[3]:
                st.metric("Max |Z-Score|", f"{df['z_score'].abs().max():.2f}σ")
    
    # Crisis Logs Tab
    with tab_crisis:
        render_crisis_logs()
    
    # Raw Data Tab
    with tab_data:
        st.header("Raw Market Data")
        
        display_columns = st.multiselect(
            "Columns",
            options=df.columns.tolist(),
            default=['time', 'open', 'high', 'low', 'close', 'volumeto', 'z_score', 'is_anomaly']
        )
        
        st.dataframe(
            df[display_columns].sort_values('time', ascending=False),
            use_container_width=True,
            height=500
        )
        
        # Export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"{selected_coin}_depeg_monitor_{lookback_days}d.csv",
            mime="text/csv"
        )
    
    # Methodology Tab
    with tab_methodology:
        render_methodology()


if __name__ == "__main__":
    main()

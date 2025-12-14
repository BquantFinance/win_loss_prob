# -*- coding: utf-8 -*-
"""
Win/Loss Probability Calculator - Demonstrating the Sum vs. Product Error
A visual exploration of why summing returns is mathematically incorrect
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Return Compounding Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --accent-green: #00ff88;
        --accent-red: #ff3366;
        --accent-blue: #00d4ff;
        --accent-purple: #a855f7;
        --text-primary: #ffffff;
        --text-secondary: #8b8b9e;
        --border-color: #2a2a3a;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    p, span, label, .stMarkdown {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Hero title styling */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, #15151f 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .green { color: var(--accent-green); }
    .red { color: var(--accent-red); }
    .blue { color: var(--accent-blue); }
    .purple { color: var(--accent-purple); }
    
    /* Error highlight box */
    .error-box {
        background: linear-gradient(145deg, rgba(255, 51, 102, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%);
        border: 1px solid rgba(255, 51, 102, 0.3);
        border-left: 4px solid var(--accent-red);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    
    .correct-box {
        background: linear-gradient(145deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-left: 4px solid var(--accent-green);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--accent-blue);
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background-color: var(--border-color) !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue)) !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Formula box */
    .formula-box {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Space Grotesk, sans-serif', 'color': '#ffffff'},
    'title': {'font': {'size': 20, 'color': '#ffffff'}},
    'xaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'linecolor': 'rgba(255,255,255,0.1)',
        'tickfont': {'color': '#8b8b9e'},
        'title': {'font': {'color': '#8b8b9e'}}
    },
    'yaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'linecolor': 'rgba(255,255,255,0.1)',
        'tickfont': {'color': '#8b8b9e'},
        'title': {'font': {'color': '#8b8b9e'}}
    }
}

COLORS = {
    'green': '#00ff88',
    'red': '#ff3366',
    'blue': '#00d4ff',
    'purple': '#a855f7',
    'yellow': '#ffd700',
    'orange': '#ff9500'
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: list, start_date: str) -> pd.DataFrame:
    """Load price data from Yahoo Finance"""
    data = yf.download(tickers, start_date, auto_adjust=True, progress=False)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data


def calculate_cumulative_returns(returns: pd.Series, window: int, method: str) -> pd.Series:
    """Calculate cumulative returns using sum or product method"""
    if method == 'sum':
        return returns.rolling(window=window).sum().dropna()
    else:  # product
        return (1 + returns).rolling(window=window).apply(np.prod, raw=True).dropna() - 1


def calculate_win_probability(cumulative_returns: pd.Series) -> float:
    """Calculate the probability of positive returns"""
    return (cumulative_returns > 0).sum() / len(cumulative_returns) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_comparison_chart(returns: pd.Series, periods: list, ticker: str) -> go.Figure:
    """Create comparison chart of sum vs product probabilities"""
    
    sum_probs = []
    prod_probs = []
    
    for period in periods:
        sum_cum = calculate_cumulative_returns(returns, period, 'sum')
        prod_cum = calculate_cumulative_returns(returns, period, 'product')
        sum_probs.append(calculate_win_probability(sum_cum))
        prod_probs.append(calculate_win_probability(prod_cum))
    
    period_labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(periods)]
    
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=period_labels,
        y=sum_probs,
        name='❌ Sum (Incorrect)',
        mode='lines+markers',
        line=dict(color=COLORS['red'], width=3),
        marker=dict(size=12, symbol='x', line=dict(width=2)),
        hovertemplate='<b>%{x}</b><br>Sum Method: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=period_labels,
        y=prod_probs,
        name='✓ Product (Correct)',
        mode='lines+markers',
        line=dict(color=COLORS['green'], width=3),
        marker=dict(size=12, symbol='circle', line=dict(width=2, color='#0a0a0f')),
        hovertemplate='<b>%{x}</b><br>Product Method: %{y:.1f}%<extra></extra>'
    ))
    
    # Add error area
    fig.add_trace(go.Scatter(
        x=period_labels + period_labels[::-1],
        y=sum_probs + prod_probs[::-1],
        fill='toself',
        fillcolor='rgba(255, 51, 102, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Error Gap',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f'Win Probability by Holding Period — {ticker}', x=0.5),
        xaxis_title='Holding Period',
        yaxis_title='Win Probability (%)',
        yaxis=dict(range=[40, 102], **PLOTLY_THEME['yaxis']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_error_magnitude_chart(returns: pd.Series, periods: list) -> go.Figure:
    """Create bar chart showing error magnitude at each period"""
    
    errors = []
    for period in periods:
        sum_cum = calculate_cumulative_returns(returns, period, 'sum')
        prod_cum = calculate_cumulative_returns(returns, period, 'product')
        error = calculate_win_probability(sum_cum) - calculate_win_probability(prod_cum)
        errors.append(error)
    
    period_labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(periods)]
    
    colors = [COLORS['green'] if e <= 0.5 else COLORS['yellow'] if e <= 2 else COLORS['orange'] if e <= 5 else COLORS['red'] for e in errors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=period_labels,
        y=errors,
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9
        ),
        text=[f'{e:.2f}%' for e in errors],
        textposition='outside',
        textfont=dict(color='#ffffff', size=12, family='JetBrains Mono'),
        hovertemplate='<b>%{x}</b><br>Error: %{y:.2f} pp<extra></extra>'
    ))
    
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='Calculation Error by Period (percentage points)', x=0.5),
        xaxis_title='Holding Period',
        yaxis_title='Error (pp)',
        height=400,
        showlegend=False
    )
    
    return fig


def create_path_simulation(initial_price: float = 100, n_periods: int = 20, volatility: float = 0.02, seed: int = 42) -> go.Figure:
    """Create a visual simulation showing sum vs product divergence"""
    
    np.random.seed(seed)
    
    # Generate random returns
    returns = np.random.normal(0, volatility, n_periods)
    
    # Calculate paths
    sum_path = initial_price * (1 + np.cumsum(returns))
    prod_path = initial_price * np.cumprod(1 + returns)
    actual_prices = [initial_price] + list(prod_path)
    sum_prices = [initial_price] + list(sum_path)
    
    periods = list(range(n_periods + 1))
    
    fig = go.Figure()
    
    # Actual price path
    fig.add_trace(go.Scatter(
        x=periods,
        y=actual_prices,
        name='Actual Price (Product)',
        mode='lines',
        line=dict(color=COLORS['green'], width=4),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    # Sum-implied path
    fig.add_trace(go.Scatter(
        x=periods,
        y=sum_prices,
        name='Implied by Sum',
        mode='lines',
        line=dict(color=COLORS['red'], width=3, dash='dash')
    ))
    
    # Starting point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[initial_price],
        mode='markers',
        marker=dict(size=15, color=COLORS['blue'], symbol='diamond'),
        name='Start',
        showlegend=False
    ))
    
    # End points
    fig.add_trace(go.Scatter(
        x=[n_periods, n_periods],
        y=[actual_prices[-1], sum_prices[-1]],
        mode='markers+text',
        marker=dict(size=12, color=[COLORS['green'], COLORS['red']]),
        text=[f'${actual_prices[-1]:.2f}', f'${sum_prices[-1]:.2f}'],
        textposition='middle right',
        textfont=dict(color='#ffffff', size=11),
        showlegend=False
    ))
    
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='Price Path: Sum vs Product Method', x=0.5),
        xaxis_title='Period',
        yaxis_title='Price ($)',
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig, returns, actual_prices[-1], sum_prices[-1]


def create_distribution_comparison(returns: pd.Series, period: int) -> go.Figure:
    """Create histogram comparison of cumulative returns"""
    
    sum_cum = calculate_cumulative_returns(returns, period, 'sum')
    prod_cum = calculate_cumulative_returns(returns, period, 'product')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('❌ Sum Method', '✓ Product Method'),
        horizontal_spacing=0.1
    )
    
    # Sum histogram
    fig.add_trace(go.Histogram(
        x=sum_cum * 100,
        nbinsx=50,
        marker=dict(color=COLORS['red'], opacity=0.7, line=dict(width=0)),
        name='Sum'
    ), row=1, col=1)
    
    # Product histogram
    fig.add_trace(go.Histogram(
        x=prod_cum * 100,
        nbinsx=50,
        marker=dict(color=COLORS['green'], opacity=0.7, line=dict(width=0)),
        name='Product'
    ), row=1, col=2)
    
    # Add vertical line at 0
    for col in [1, 2]:
        fig.add_vline(x=0, line=dict(color='#ffffff', width=2, dash='dash'), row=1, col=col)
    
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=f'Distribution of {period}-Day Cumulative Returns', x=0.5),
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text='Return (%)', **PLOTLY_THEME['xaxis'])
    fig.update_yaxes(title_text='Frequency', **PLOTLY_THEME['yaxis'])
    
    return fig


def create_heatmap(df: pd.DataFrame, tickers: list, periods: list) -> go.Figure:
    """Create heatmap of win probabilities"""
    
    period_labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(periods)]
    
    z_data = []
    for ticker in tickers:
        row = []
        returns = df[ticker].pct_change().dropna()
        for period in periods:
            if len(returns) >= period:
                cum = calculate_cumulative_returns(returns, period, 'product')
                row.append(calculate_win_probability(cum))
            else:
                row.append(np.nan)
        z_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=period_labels,
        y=tickers,
        colorscale=[
            [0.0, COLORS['red']],
            [0.5, '#1a1a24'],
            [1.0, COLORS['green']]
        ],
        zmid=50,
        text=[[f'{val:.1f}%' if not np.isnan(val) else '' for val in row] for row in z_data],
        texttemplate='%{text}',
        textfont=dict(size=11, color='#ffffff'),
        hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Win Prob: %{z:.1f}%<extra></extra>',
        colorbar=dict(
            title='Win %',
            titleside='right',
            tickfont=dict(color='#8b8b9e'),
            titlefont=dict(color='#8b8b9e')
        )
    ))
    
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text='Win Probability Heatmap (Correct Method)', x=0.5),
        height=max(400, len(tickers) * 35),
        xaxis_title='Holding Period',
        yaxis_title='Asset'
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Hero Section
    st.markdown('<h1 class="hero-title">📊 Return Compounding Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Why summing returns is mathematically incorrect — and why it matters</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Ticker selection
        default_tickers = ['^GSPC', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        tickers = st.multiselect(
            "Select Assets",
            options=['^GSPC', '^DJI', '^IXIC', 'QQQ', 'IWM', 'SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            default=default_tickers,
            help="Choose assets to analyze"
        )
        
        if not tickers:
            tickers = default_tickers
        
        primary_ticker = st.selectbox(
            "Primary Asset for Analysis",
            options=tickers,
            index=0
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        start_year = st.slider(
            "Start Year",
            min_value=1990,
            max_value=2020,
            value=2000
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Simulation controls
        st.markdown("### 🎲 Simulation")
        sim_volatility = st.slider(
            "Daily Volatility (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        ) / 100
        
        sim_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=999,
            value=42
        )
    
    # Load data
    with st.spinner('Loading market data...'):
        try:
            data = load_data(tickers, f'{start_year}-01-01')
            returns = data[primary_ticker].pct_change().dropna()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Define periods
    periods = [1, 5, 20, 60, 120, 252, 252*2, 252*5, 252*10, 252*15, 252*20]
    # Filter periods based on data length
    max_period = len(returns) - 1
    periods = [p for p in periods if p <= max_period]
    
    # The Error Explanation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="error-box">
            <h3 style="color: #ff3366; margin-top: 0;">❌ Incorrect: Summing Returns</h3>
            <div class="formula-box">
                R<sub>cumulative</sub> = r₁ + r₂ + r₃ + ... + rₙ
            </div>
            <p style="color: #8b8b9e; font-size: 0.95rem;">
                Summing assumes returns are additive. But if you gain 10% then lose 10%, 
                you don't break even — you lose money!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="correct-box">
            <h3 style="color: #00ff88; margin-top: 0;">✓ Correct: Compounding Returns</h3>
            <div class="formula-box">
                R<sub>cumulative</sub> = (1+r₁)(1+r₂)(1+r₃)...(1+rₙ) - 1
            </div>
            <p style="color: #8b8b9e; font-size: 0.95rem;">
                Compounding captures the multiplicative nature of returns.
                +10% then -10% = (1.10)(0.90) - 1 = <b>-1%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simulation visualization
    st.markdown("---")
    st.markdown("## 🎬 Visual Simulation")
    
    fig_sim, sim_returns, actual_end, sum_end = create_path_simulation(
        initial_price=100,
        n_periods=20,
        volatility=sim_volatility,
        seed=int(sim_seed)
    )
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Simulation metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_sum_return = sum(sim_returns) * 100
    total_prod_return = (actual_end / 100 - 1) * 100
    error = total_sum_return - total_prod_return
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sum Method</div>
            <div class="metric-value red">{total_sum_return:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Product Method</div>
            <div class="metric-value green">{total_prod_return:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calculation Error</div>
            <div class="metric-value purple">{error:+.2f}pp</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price Difference</div>
            <div class="metric-value blue">${abs(actual_end - sum_end):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main comparison chart
    st.markdown("---")
    st.markdown(f"## 📈 Win Probability Analysis — {primary_ticker}")
    
    fig_comparison = create_comparison_chart(returns, periods, primary_ticker)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Error magnitude
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_error = create_error_magnitude_chart(returns, periods)
        st.plotly_chart(fig_error, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <h3 style="color: #00d4ff; margin-top: 0;">💡 Key Insight</h3>
            <p style="color: #8b8b9e;">
                The error compounds over time. For short periods (1 day), 
                the difference is negligible. But for multi-year horizons, 
                the sum method can overestimate win probability by 
                <b style="color: #ff3366;">several percentage points</b>.
            </p>
            <p style="color: #8b8b9e; margin-top: 1rem;">
                This matters for backtesting, risk management, and 
                setting realistic expectations for long-term investing.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribution comparison
    st.markdown("---")
    st.markdown("## 📊 Return Distribution Comparison")
    
    period_options = {
        '1 Month (20 days)': 20,
        '1 Year (252 days)': 252,
        '5 Years (1260 days)': 1260,
        '10 Years (2520 days)': 2520
    }
    
    available_periods = {k: v for k, v in period_options.items() if v <= max_period}
    
    if available_periods:
        selected_period_label = st.selectbox(
            "Select Period for Distribution",
            options=list(available_periods.keys())
        )
        selected_period = available_periods[selected_period_label]
        
        fig_dist = create_distribution_comparison(returns, selected_period)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Heatmap
    if len(tickers) > 1:
        st.markdown("---")
        st.markdown("## 🗺️ Multi-Asset Heatmap")
        
        fig_heatmap = create_heatmap(data, tickers, periods)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b8b9e; padding: 2rem 0;">
        <p style="font-size: 0.9rem;">
            Built with Streamlit & Plotly | Data from Yahoo Finance<br>
            <span style="color: #00d4ff;">BQuant Finance</span> — Quantitative Analysis Tools
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

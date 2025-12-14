# -*- coding: utf-8 -*-
"""
Win Probability Calculator
bquantfinance.com | @Gsnchez
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Win Probability | BQuant Finance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-base: #0a0a0c;
        --bg-surface: #111114;
        --bg-elevated: #18181b;
        --accent: #3b82f6;
        --positive: #22c55e;
        --negative: #dc2626;
        --text: #fafafa;
        --text-secondary: #71717a;
        --border: #27272a;
    }
    
    .stApp {
        background: var(--bg-base);
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    * {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
    }
    
    code, .mono {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0 2.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--positive), var(--accent));
        border-radius: 8px;
    }
    
    .logo-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    
    .tagline {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 400;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--text);
    }
    
    .metric-value.positive { color: var(--positive); }
    .metric-value.accent { color: var(--accent); }
    
    .section-title {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2rem 0 1rem 0;
    }
    
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    
    .data-table th {
        text-align: left;
        padding: 0.75rem 1rem;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border);
    }
    
    .data-table td {
        padding: 0.75rem 1rem;
        color: var(--text);
        border-bottom: 1px solid var(--border);
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .data-table tr:hover {
        background: var(--bg-elevated);
    }
    
    .pill {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .pill.positive {
        background: rgba(34, 197, 94, 0.15);
        color: var(--positive);
    }
    
    .pill.negative {
        background: rgba(220, 38, 38, 0.15);
        color: var(--negative);
    }
    
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
    }
    
    .footer a {
        color: var(--accent);
        text-decoration: none;
    }
    
    section[data-testid="stSidebar"] {
        background: var(--bg-surface);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'positive': '#22c55e',
    'negative': '#dc2626',
    'accent': '#3b82f6',
    'muted': '#71717a',
    'grid': 'rgba(255,255,255,0.03)'
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start_date):
    """Load price data from Yahoo Finance"""
    try:
        data = yf.download(tickers, start_date, auto_adjust=True, progress=False)
        if 'Close' in data.columns or len(tickers) == 1:
            close = data['Close'] if 'Close' in data.columns else data
        else:
            close = data['Close']
        
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        return close
    except Exception as e:
        return pd.DataFrame()


def calculate_probabilities(returns, periods):
    """Calculate win/loss probabilities for each period"""
    results = []
    for period in periods:
        if len(returns) >= period:
            cum_ret = (1 + returns).rolling(window=period).apply(np.prod, raw=True).dropna() - 1
            n = len(cum_ret)
            if n > 0:
                prob = (cum_ret > 0).sum() / n * 100
                results.append({
                    'period': period,
                    'win': round(prob, 1),
                    'loss': round(100 - prob, 1),
                    'n': n
                })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_main_chart(df, ticker):
    """Create main probability line chart"""
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels,
        y=df['win'],
        name='Win',
        mode='lines+markers',
        line=dict(color=COLORS['positive'], width=2),
        marker=dict(size=6, color=COLORS['positive']),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.08)',
        hovertemplate='%{y:.1f}%<extra>Win</extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=labels,
        y=df['loss'],
        name='Loss',
        mode='lines+markers',
        line=dict(color=COLORS['negative'], width=2),
        marker=dict(size=6, color=COLORS['negative']),
        hovertemplate='%{y:.1f}%<extra>Loss</extra>'
    ))
    
    fig.add_hline(y=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', color='#fafafa', size=11),
        margin=dict(l=48, r=24, t=48, b=48),
        title=dict(text=ticker, font=dict(size=14, color='#a1a1aa'), x=0, xanchor='left'),
        xaxis=dict(
            title='Holding Period',
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10),
            title_font=dict(color='#71717a', size=11)
        ),
        yaxis=dict(
            title='Probability (%)',
            range=[0, 105],
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10),
            title_font=dict(color='#71717a', size=11)
        ),
        height=420,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#18181b', font_size=11, font_family='IBM Plex Sans')
    )
    
    return fig


def create_bar_chart(df):
    """Create stacked bar chart"""
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=df['win'],
        name='Win',
        marker=dict(color=COLORS['positive'], opacity=0.85),
        text=[f"{p:.0f}" for p in df['win']],
        textposition='inside',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=labels,
        y=df['loss'],
        name='Loss',
        marker=dict(color=COLORS['negative'], opacity=0.85),
        text=[f"{p:.0f}" for p in df['loss']],
        textposition='inside',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', color='#fafafa', size=11),
        margin=dict(l=48, r=24, t=24, b=48),
        barmode='stack',
        height=320,
        showlegend=False,
        xaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10)
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10)
        ),
        hoverlabel=dict(bgcolor='#18181b', font_size=11, font_family='IBM Plex Sans')
    )
    
    return fig


def create_comparison_chart(data, tickers, period):
    """Create horizontal bar chart comparing tickers"""
    results = []
    
    for t in tickers:
        if t in data.columns:
            ret = data[t].pct_change().dropna()
            if len(ret) >= period:
                cum = (1 + ret).rolling(window=period).apply(np.prod, raw=True).dropna() - 1
                if len(cum) > 0:
                    prob = (cum > 0).sum() / len(cum) * 100
                    results.append({'ticker': t, 'prob': prob})
    
    if not results:
        return None
    
    df = pd.DataFrame(results).sort_values('prob', ascending=True)
    colors = [COLORS['positive'] if p >= 50 else COLORS['negative'] for p in df['prob']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['prob'],
        y=df['ticker'],
        orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f"{p:.1f}%" for p in df['prob']],
        textposition='outside',
        textfont=dict(size=10, color='#a1a1aa'),
        hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', color='#fafafa', size=11),
        margin=dict(l=80, r=60, t=24, b=24),
        height=max(200, len(df) * 36),
        showlegend=False,
        xaxis=dict(
            range=[0, 105],
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10)
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10)
        ),
        hoverlabel=dict(bgcolor='#18181b', font_size=11, font_family='IBM Plex Sans')
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Configuración")
        
        tickers_input = st.text_area(
            "Tickers",
            value="^GSPC, QQQ, AAPL",
            height=70,
            help="Separados por coma o espacio"
        )
        
        # Parse tickers
        raw = tickers_input.replace(',', ' ').replace(';', ' ').split()
        tickers = [t.strip().upper() for t in raw if t.strip()]
        
        if not tickers:
            tickers = ['^GSPC']
        
        ticker_principal = st.selectbox("Principal", tickers)
        year = st.slider("Desde", 1980, 2020, 2000)
    
    # Load data
    with st.spinner('Cargando datos...'):
        data = load_data(tickers, f'{year}-01-01')
    
    if data.empty:
        st.error("Error al cargar datos. Verifica los tickers.")
        st.stop()
    
    if ticker_principal not in data.columns:
        st.error(f"No se encontraron datos para {ticker_principal}")
        st.stop()
    
    returns = data[ticker_principal].pct_change().dropna()
    
    if len(returns) < 20:
        st.error("Datos insuficientes para el análisis.")
        st.stop()
    
    # Define periods
    all_periods = [1, 5, 20, 60, 120, 252, 504, 1260, 2520, 3780, 5040]
    periods = [p for p in all_periods if p <= len(returns) - 1]
    
    # Calculate probabilities
    df_prob = calculate_probabilities(returns, periods)
    
    if df_prob.empty:
        st.error("No se pudieron calcular las probabilidades.")
        st.stop()
    
    # Get specific probabilities
    def get_prob(period_val):
        row = df_prob[df_prob['period'] == period_val]
        if not row.empty:
            return row['win'].values[0]
        return None
    
    p1y = get_prob(252)
    p5y = get_prob(1260)
    p10y = get_prob(2520)
    years_data = len(returns) / 252
    
    # Header
    st.markdown("""
    <div class="header">
        <div class="logo">
            <div class="logo-icon"></div>
            <span class="logo-text">Win Probability</span>
        </div>
        <span class="tagline">Historical win/loss probability by holding period</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1 = f"{p1y:.1f}%" if p1y else "N/A"
    m5 = f"{p5y:.1f}%" if p5y else "N/A"
    m10 = f"{p10y:.1f}%" if p10y else "N/A"
    
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">1 Year Win Rate</div>
            <div class="metric-value positive">{m1}</div>
        </div>
        <div class="metric">
            <div class="metric-label">5 Year Win Rate</div>
            <div class="metric-value positive">{m5}</div>
        </div>
        <div class="metric">
            <div class="metric-label">10 Year Win Rate</div>
            <div class="metric-value positive">{m10}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Data Range</div>
            <div class="metric-value accent">{years_data:.0f} years</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chart
    fig_main = create_main_chart(df_prob, ticker_principal)
    st.plotly_chart(fig_main, use_container_width=True, config={'displayModeBar': False})
    
    # Second row
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
        fig_bars = create_bar_chart(df_prob)
        st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
        
        labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df_prob)]
        
        table_rows = ""
        for i, row in df_prob.iterrows():
            win_class = 'positive' if row['win'] >= 50 else 'negative'
            label = labels[i] if i < len(labels) else f"{row['period']}D"
            table_rows += f"""
            <tr>
                <td>{label}</td>
                <td><span class="pill {win_class}">{row['win']:.1f}%</span></td>
                <td>{row['loss']:.1f}%</td>
            </tr>
            """
        
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Period</th>
                    <th>Win</th>
                    <th>Loss</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    
    # Comparison (only if multiple tickers)
    if len(tickers) > 1:
        st.markdown('<div class="section-title">Comparison · 1 Year</div>', unsafe_allow_html=True)
        fig_comp = create_comparison_chart(data, tickers, 252)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
    
    # Footer
    st.markdown("""
    <div class="footer">
        <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a> · 
        <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

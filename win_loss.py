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
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    :root {
        --bg: #09090b;
        --surface: #18181b;
        --border: #27272a;
        --text: #fafafa;
        --muted: #a1a1aa;
        --green: #22c55e;
        --red: #ef4444;
        --blue: #3b82f6;
    }
    
    .stApp {
        background: var(--bg);
    }
    
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    .main .block-container {
        padding: 2rem 4rem 3rem 4rem;
        max-width: 1200px;
    }
    
    * {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding-bottom: 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
    }
    
    .title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        color: var(--muted);
        font-size: 0.95rem;
    }
    
    /* Controls */
    .controls-row {
        display: flex;
        gap: 1rem;
        align-items: end;
        margin-bottom: 2rem;
        padding: 1.25rem;
        background: var(--surface);
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    /* Metrics */
    .metrics {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    .metric-val {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.75rem;
        font-weight: 500;
    }
    
    .metric-val.green { color: var(--green); }
    .metric-val.blue { color: var(--blue); }
    
    /* Section */
    .section {
        margin-bottom: 2rem;
    }
    
    .section-head {
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    /* Table */
    .tbl {
        width: 100%;
        border-collapse: collapse;
    }
    
    .tbl th {
        text-align: left;
        padding: 0.6rem 0.8rem;
        font-size: 0.65rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border);
        font-weight: 500;
    }
    
    .tbl td {
        padding: 0.6rem 0.8rem;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem;
        color: var(--text);
        border-bottom: 1px solid var(--border);
    }
    
    .tbl tr:last-child td {
        border-bottom: none;
    }
    
    .tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .tag.g {
        background: rgba(34, 197, 94, 0.15);
        color: var(--green);
    }
    
    .tag.r {
        background: rgba(239, 68, 68, 0.15);
        color: var(--red);
    }
    
    /* Footer */
    .foot {
        text-align: center;
        padding-top: 2rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.8rem;
    }
    
    .foot a {
        color: var(--blue);
        text-decoration: none;
    }
    
    /* Hide streamlit */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--bg) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg) !important;
        border-color: var(--border) !important;
    }
    
    div[data-baseweb="select"] > div {
        background: var(--bg) !important;
        border-color: var(--border) !important;
    }
    
    .stSlider label, .stTextInput label, .stSelectbox label {
        font-size: 0.7rem !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    @media (max-width: 768px) {
        .metrics { grid-template-columns: repeat(2, 1fr); }
        .main .block-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════════════════

GREEN = '#22c55e'
RED = '#ef4444'
GRID = 'rgba(255,255,255,0.03)'

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start):
    try:
        data = yf.download(tickers, start, auto_adjust=True, progress=False)
        close = data['Close'] if 'Close' in data.columns else data
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        return close
    except:
        return pd.DataFrame()


def calc_probs(returns, periods):
    results = []
    for p in periods:
        if len(returns) >= p:
            cum = (1 + returns).rolling(window=p).apply(np.prod, raw=True).dropna() - 1
            n = len(cum)
            if n > 0:
                win = (cum > 0).sum() / n * 100
                results.append({'period': p, 'win': round(win, 1), 'loss': round(100 - win, 1)})
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def chart_main(df, ticker):
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels, y=df['win'], name='Win %',
        mode='lines+markers',
        line=dict(color=GREEN, width=2.5),
        marker=dict(size=8, color=GREEN),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.1)',
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=labels, y=df['loss'], name='Loss %',
        mode='lines+markers',
        line=dict(color=RED, width=2.5),
        marker=dict(size=8, color=RED),
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    fig.add_hline(y=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis=dict(
            title='Holding Period',
            gridcolor=GRID,
            linecolor='#27272a',
            tickfont=dict(color='#71717a'),
            title_font=dict(color='#71717a', size=11)
        ),
        yaxis=dict(
            title='Probability (%)',
            range=[0, 105],
            gridcolor=GRID,
            linecolor='#27272a',
            tickfont=dict(color='#71717a'),
            title_font=dict(color='#71717a', size=11)
        ),
        height=380,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)', font=dict(size=11)
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#18181b', font_size=11)
    )
    return fig


def chart_bars(df):
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels, y=df['win'], name='Win',
        marker=dict(color=GREEN, opacity=0.9),
        text=[f"{v:.0f}" for v in df['win']],
        textposition='inside',
        textfont=dict(size=9, color='white'),
        hovertemplate='Win: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=labels, y=df['loss'], name='Loss',
        marker=dict(color=RED, opacity=0.9),
        text=[f"{v:.0f}" for v in df['loss']],
        textposition='inside',
        textfont=dict(size=9, color='white'),
        hovertemplate='Loss: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=40, r=20, t=20, b=40),
        barmode='stack',
        height=280,
        showlegend=False,
        xaxis=dict(gridcolor=GRID, linecolor='#27272a', tickfont=dict(color='#71717a')),
        yaxis=dict(gridcolor=GRID, linecolor='#27272a', tickfont=dict(color='#71717a')),
        hoverlabel=dict(bgcolor='#18181b', font_size=11)
    )
    return fig


def chart_compare(data, tickers, period):
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
    colors = [GREEN if p >= 50 else RED for p in df['prob']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['prob'], y=df['ticker'], orientation='h',
        marker=dict(color=colors, opacity=0.9),
        text=[f"{p:.1f}%" for p in df['prob']],
        textposition='outside',
        textfont=dict(size=11, color='#a1a1aa'),
        hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=70, r=50, t=20, b=30),
        height=max(180, len(df) * 40),
        showlegend=False,
        xaxis=dict(range=[0, 105], gridcolor=GRID, linecolor='#27272a', tickfont=dict(color='#71717a')),
        yaxis=dict(gridcolor=GRID, linecolor='#27272a', tickfont=dict(color='#a1a1aa', size=11)),
        hoverlabel=dict(bgcolor='#18181b', font_size=11)
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Header
    st.markdown("""
    <div class="header">
        <div class="title">📊 Win Probability</div>
        <div class="subtitle">Historical probability of positive returns by holding period</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    c1, c2, c3 = st.columns([3, 1, 1])
    
    with c1:
        tickers_input = st.text_input(
            "TICKERS",
            value="^GSPC, QQQ, AAPL, MSFT",
            help="Comma or space separated"
        )
    
    with c2:
        year = st.selectbox(
            "FROM",
            options=[1980, 1990, 2000, 2010, 2015],
            index=2
        )
    
    # Parse tickers
    raw = tickers_input.replace(',', ' ').replace(';', ' ').split()
    tickers = [t.strip().upper() for t in raw if t.strip()]
    if not tickers:
        tickers = ['^GSPC']
    
    with c3:
        main_ticker = st.selectbox("ANALYZE", tickers)
    
    # Load data
    with st.spinner(''):
        data = load_data(tickers, f'{year}-01-01')
    
    if data.empty or main_ticker not in data.columns:
        st.error("Could not load data. Check tickers.")
        st.stop()
    
    returns = data[main_ticker].pct_change().dropna()
    
    if len(returns) < 20:
        st.error("Not enough data.")
        st.stop()
    
    # Periods
    all_p = [1, 5, 20, 60, 120, 252, 504, 1260, 2520, 3780, 5040]
    periods = [p for p in all_p if p <= len(returns) - 1]
    
    df_prob = calc_probs(returns, periods)
    
    if df_prob.empty:
        st.error("Could not calculate probabilities.")
        st.stop()
    
    # Helper
    def get_p(per):
        r = df_prob[df_prob['period'] == per]
        return r['win'].values[0] if not r.empty else None
    
    p1y, p5y, p10y = get_p(252), get_p(1260), get_p(2520)
    years_data = len(returns) / 252
    
    # Metrics
    st.markdown(f"""
    <div class="metrics">
        <div class="metric-box">
            <div class="metric-label">1 Year</div>
            <div class="metric-val green">{p1y:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">5 Years</div>
            <div class="metric-val green">{p5y:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">10 Years</div>
            <div class="metric-val green">{p10y:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Data</div>
            <div class="metric-val blue">{years_data:.0f} yrs</div>
        </div>
    </div>
    """, unsafe_allow_html=True) if p1y and p5y and p10y else None
    
    # Main chart
    st.plotly_chart(chart_main(df_prob, main_ticker), use_container_width=True, config={'displayModeBar': False})
    
    # Second row
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.markdown('<div class="section-head">Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_bars(df_prob), use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown('<div class="section-head">Details</div>', unsafe_allow_html=True)
        
        labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y']
        
        rows = ""
        for idx, r in enumerate(df_prob.itertuples()):
            lbl = labels[idx] if idx < len(labels) else f"{r.period}D"
            tag_class = 'g' if r.win >= 50 else 'r'
            rows += f'<tr><td>{lbl}</td><td><span class="tag {tag_class}">{r.win:.1f}%</span></td><td>{r.loss:.1f}%</td></tr>'
        
        st.markdown(f"""
        <table class="tbl">
            <thead><tr><th>Period</th><th>Win</th><th>Loss</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
    
    # Comparison
    if len(tickers) > 1:
        st.markdown('<div class="section-head" style="margin-top:2rem;">Comparison · 1 Year Win Rate</div>', unsafe_allow_html=True)
        fig_c = chart_compare(data, tickers, 252)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})
    
    # Footer
    st.markdown("""
    <div class="foot">
        <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a> · 
        <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

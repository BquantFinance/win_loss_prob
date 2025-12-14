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
    
    .stApp { background: var(--bg); }
    [data-testid="collapsedControl"] { display: none; }
    .main .block-container { padding: 2rem 4rem 3rem 4rem; max-width: 1200px; }
    * { font-family: 'IBM Plex Sans', -apple-system, sans-serif !important; }
    
    .header { text-align: center; padding-bottom: 2rem; margin-bottom: 2rem; border-bottom: 1px solid var(--border); }
    .title { font-size: 2rem; font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }
    .subtitle { color: var(--muted); font-size: 0.95rem; }
    
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0 2rem 0; }
    .metric-box { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; text-align: center; }
    .metric-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
    .metric-val { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.75rem; font-weight: 500; }
    .metric-val.green { color: var(--green); }
    .metric-val.blue { color: var(--blue); }
    
    .section-head { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 1rem 0; }
    
    .insight { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin: 1.5rem 0; }
    .insight-text { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
    .insight-text strong { color: var(--green); }
    
    .foot { text-align: center; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem; }
    .foot a { color: var(--blue); text-decoration: none; }
    
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    .stTextInput > div > div > input { background: var(--bg) !important; border-color: var(--border) !important; color: var(--text) !important; }
    .stSelectbox > div > div { background: var(--bg) !important; border-color: var(--border) !important; }
    div[data-baseweb="select"] > div { background: var(--bg) !important; border-color: var(--border) !important; }
    .stTextInput label, .stSelectbox label { font-size: 0.7rem !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.05em; }
    
    @media (max-width: 768px) { .metrics { grid-template-columns: repeat(2, 1fr); } .main .block-container { padding: 1rem; } }
</style>
""", unsafe_allow_html=True)

GREEN = '#22c55e'
RED = '#ef4444'
BLUE = '#3b82f6'
GRID = 'rgba(255,255,255,0.03)'
LABELS = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y']


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start):
    try:
        data = yf.download(tickers, start, auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        close = data['Close'] if 'Close' in data.columns else data
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        return close.dropna(how='all')
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
                results.append({'period': p, 'win': round(win, 1)})
    return pd.DataFrame(results)


def chart_area(df, ticker):
    labels = LABELS[:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels, y=df['win'],
        mode='lines',
        line=dict(color=GREEN, width=3),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.15)',
        hovertemplate='<b>%{x}</b><br>Win: %{y:.1f}%<extra></extra>',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=labels, y=df['win'],
        mode='markers',
        marker=dict(size=10, color=GREEN, line=dict(width=2, color='#09090b')),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.add_hline(y=50, line=dict(color='#3f3f46', width=1, dash='dot'),
                  annotation_text="50%", annotation_position="right",
                  annotation=dict(font_color='#52525b', font_size=10))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=50, r=30, t=40, b=50),
        title=dict(text=f'Win Probability · {ticker}', font=dict(size=14, color='#a1a1aa'), x=0, xanchor='left'),
        xaxis=dict(title='Holding Period', gridcolor=GRID, linecolor='#27272a', 
                   tickfont=dict(color='#71717a'), title_font=dict(color='#71717a', size=11)),
        yaxis=dict(title='Probability (%)', range=[0, 105], gridcolor=GRID, linecolor='#27272a',
                   tickfont=dict(color='#71717a'), title_font=dict(color='#71717a', size=11)),
        height=400,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#18181b', font_size=12, font_family='IBM Plex Sans')
    )
    return fig


def chart_heatmap_strip(df):
    labels = LABELS[:len(df)]
    probs = df['win'].values
    
    def prob_to_color(p):
        if p < 50:
            intensity = (50 - p) / 50
            return f'rgba(239, 68, 68, {0.3 + intensity * 0.7})'
        else:
            intensity = (p - 50) / 50
            return f'rgba(34, 197, 94, {0.3 + intensity * 0.7})'
    
    colors = [prob_to_color(p) for p in probs]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=[1] * len(labels),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{p:.0f}%' for p in probs],
        textposition='inside',
        textfont=dict(size=12, color='white', family='IBM Plex Mono'),
        hovertemplate='<b>%{x}</b>: %{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(tickfont=dict(color='#71717a', size=10), showgrid=False, linecolor='rgba(0,0,0,0)'),
        yaxis=dict(visible=False, showgrid=False),
        height=80,
        bargap=0.08,
        hoverlabel=dict(bgcolor='#18181b', font_size=11)
    )
    return fig


def chart_slope(data, tickers, periods_to_show):
    results = []
    for t in tickers:
        if t in data.columns:
            ret = data[t].pct_change().dropna()
            for p in periods_to_show:
                if len(ret) >= p:
                    cum = (1 + ret).rolling(window=p).apply(np.prod, raw=True).dropna() - 1
                    if len(cum) > 0:
                        prob = (cum > 0).sum() / len(cum) * 100
                        results.append({'ticker': t, 'period': p, 'win': prob})
    
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    period_labels = {1: '1D', 5: '1W', 20: '1M', 60: '3M', 120: '6M', 
                     252: '1Y', 504: '2Y', 1260: '5Y', 2520: '10Y'}
    
    palette = ['#22c55e', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16']
    
    fig = go.Figure()
    
    for i, ticker in enumerate(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker].sort_values('period')
        x_labels = [period_labels.get(p, f'{p}D') for p in ticker_data['period']]
        color = palette[i % len(palette)]
        
        fig.add_trace(go.Scatter(
            x=x_labels, y=ticker_data['win'],
            mode='lines+markers',
            name=ticker,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(width=2, color='#09090b')),
            hovertemplate=f'<b>{ticker}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.add_hline(y=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans', color='#fafafa', size=11),
        margin=dict(l=50, r=30, t=40, b=50),
        title=dict(text='Win Probability Comparison', font=dict(size=14, color='#a1a1aa'), x=0, xanchor='left'),
        xaxis=dict(title='Holding Period', gridcolor=GRID, linecolor='#27272a',
                   tickfont=dict(color='#71717a'), title_font=dict(color='#71717a', size=11)),
        yaxis=dict(title='Probability (%)', range=[30, 105], gridcolor=GRID, linecolor='#27272a',
                   tickfont=dict(color='#71717a'), title_font=dict(color='#71717a', size=11)),
        height=380,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)', font=dict(size=11)
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#18181b', font_size=11)
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="header"><div class="title">📊 Win Probability</div><div class="subtitle">Historical probability of positive returns by holding period</div></div>', unsafe_allow_html=True)

# Controls
c1, c2, c3 = st.columns([3, 1, 1])

with c1:
    tickers_input = st.text_input("TICKERS", value="^GSPC, QQQ, AAPL, MSFT", help="Comma or space separated")

with c2:
    year = st.selectbox("FROM", options=[1980, 1990, 2000, 2010, 2015], index=2)

raw = tickers_input.replace(',', ' ').replace(';', ' ').split()
tickers = [t.strip().upper() for t in raw if t.strip()]
if not tickers:
    tickers = ['^GSPC']

with c3:
    main_ticker = st.selectbox("ANALYZE", tickers)

# Load data
data = load_data(tickers, f'{year}-01-01')

# Validation
if data.empty:
    st.error("Could not load data. Check your tickers.")
    st.stop()

if main_ticker not in data.columns:
    st.error(f"No data found for **{main_ticker}**. Try a different ticker.")
    st.stop()

returns = data[main_ticker].pct_change().dropna()

if len(returns) < 252:
    st.error(f"Not enough data for **{main_ticker}** (need at least 1 year). Try an earlier start date or different ticker.")
    st.stop()

# Calculate
all_p = [1, 5, 20, 60, 120, 252, 504, 1260, 2520, 3780, 5040]
periods = [p for p in all_p if p <= len(returns) - 1]
df_prob = calc_probs(returns, periods)

if df_prob.empty:
    st.error("Could not calculate probabilities.")
    st.stop()

def get_p(per):
    r = df_prob[df_prob['period'] == per]
    return r['win'].values[0] if not r.empty else None

p1y = get_p(252)
p5y = get_p(1260)
p10y = get_p(2520)
years_data = len(returns) / 252

# Metrics
m1 = f"{p1y:.1f}%" if p1y is not None else "—"
m5 = f"{p5y:.1f}%" if p5y is not None else "—"
m10 = f"{p10y:.1f}%" if p10y is not None else "—"

st.markdown(f'''
<div class="metrics">
    <div class="metric-box"><div class="metric-label">1 Year</div><div class="metric-val green">{m1}</div></div>
    <div class="metric-box"><div class="metric-label">5 Years</div><div class="metric-val green">{m5}</div></div>
    <div class="metric-box"><div class="metric-label">10 Years</div><div class="metric-val green">{m10}</div></div>
    <div class="metric-box"><div class="metric-label">Data</div><div class="metric-val blue">{years_data:.1f} yrs</div></div>
</div>
''', unsafe_allow_html=True)

# Heatmap strip
st.markdown('<div class="section-head">At a Glance</div>', unsafe_allow_html=True)
fig_strip = chart_heatmap_strip(df_prob)
st.plotly_chart(fig_strip, use_container_width=True, config={'displayModeBar': False})

# Main area chart
fig_area = chart_area(df_prob, main_ticker)
st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})

# Insight
if p1y is not None and p10y is not None:
    improvement = p10y - p1y
    st.markdown(f'''
    <div class="insight">
        <div class="insight-text">
            Holding <strong>{main_ticker}</strong> for 10 years instead of 1 year increases your win probability 
            from <strong>{p1y:.1f}%</strong> to <strong>{p10y:.1f}%</strong> — 
            a <strong>+{improvement:.1f}pp</strong> improvement. Time in the market beats timing the market.
        </div>
    </div>
    ''', unsafe_allow_html=True)
elif p1y is not None and p5y is not None:
    improvement = p5y - p1y
    st.markdown(f'''
    <div class="insight">
        <div class="insight-text">
            Holding <strong>{main_ticker}</strong> for 5 years instead of 1 year increases your win probability 
            from <strong>{p1y:.1f}%</strong> to <strong>{p5y:.1f}%</strong> — 
            a <strong>+{improvement:.1f}pp</strong> improvement.
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Slope chart
if len(tickers) > 1:
    valid_tickers = [t for t in tickers if t in data.columns and len(data[t].pct_change().dropna()) >= 252]
    
    if len(valid_tickers) > 1:
        st.markdown('<div class="section-head">Multi-Asset Comparison</div>', unsafe_allow_html=True)
        key_periods = [p for p in [20, 252, 1260, 2520] if p <= len(returns) - 1]
        fig_slope = chart_slope(data, valid_tickers, key_periods)
        if fig_slope is not None:
            st.plotly_chart(fig_slope, use_container_width=True, config={'displayModeBar': False})

# Footer
st.markdown('<div class="foot"><a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a> · <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a></div>', unsafe_allow_html=True)

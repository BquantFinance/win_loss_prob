# -*- coding: utf-8 -*-
"""
Probabilidad de Ganancia/Pérdida en los Mercados
Desarrollado por bquantfinance.com | @Gsnchez
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Win Probability | BQuant Finance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS profesional y minimalista
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
    
    /* Header */
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
    
    /* Metrics Grid */
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
    
    /* Section */
    .section-title {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 2rem 0 1rem 0;
    }
    
    /* Table */
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
        font-family: 'IBM Plex Mono', monospace;
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
    
    /* Footer */
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
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    section[data-testid="stSidebar"] label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Hide Streamlit */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COLORES
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'positive': '#22c55e',
    'negative': '#dc2626',
    'accent': '#3b82f6',
    'muted': '#71717a',
    'grid': 'rgba(255,255,255,0.03)'
}

def get_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='IBM Plex Sans, sans-serif', color='#fafafa', size=11),
        margin=dict(l=48, r=24, t=48, b=48),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10),
            title_font=dict(color='#71717a', size=11)
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor='#27272a',
            tickfont=dict(color='#71717a', size=10),
            title_font=dict(color='#71717a', size=11)
        ),
        hoverlabel=dict(bgcolor='#18181b', font_size=11, font_family='IBM Plex Sans')
    )

# ═══════════════════════════════════════════════════════════════════════════════
# DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(tickers: list, fecha_inicio: str) -> pd.DataFrame:
    try:
        data = yf.download(tickers, fecha_inicio, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data
    except Exception:
        return pd.DataFrame()


def calcular_probabilidades(retornos: pd.Series, periodos: list) -> pd.DataFrame:
    resultados = []
    for periodo in periodos:
        if len(retornos) >= periodo:
            cum_ret = (1 + retornos).rolling(window=periodo).apply(np.prod, raw=True).dropna() - 1
            n = len(cum_ret)
            prob = (cum_ret > 0).sum() / n * 100
            resultados.append({
                'periodo': periodo,
                'prob_ganancia': round(prob, 1),
                'prob_perdida': round(100 - prob, 1),
                'n': n
            })
    return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════════

def crear_grafico_principal(df: pd.DataFrame, ticker: str) -> go.Figure:
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    # Ganancia
    fig.add_trace(go.Scatter(
        x=labels, y=df['prob_ganancia'],
        name='Win', mode='lines+markers',
        line=dict(color=COLORS['positive'], width=2),
        marker=dict(size=6, color=COLORS['positive']),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.08)',
        hovertemplate='%{y:.1f}%<extra>Win</extra>'
    ))
    
    # Pérdida
    fig.add_trace(go.Scatter(
        x=labels, y=df['prob_perdida'],
        name='Loss', mode='lines+markers',
        line=dict(color=COLORS['negative'], width=2),
        marker=dict(size=6, color=COLORS['negative']),
        hovertemplate='%{y:.1f}%<extra>Loss</extra>'
    ))
    
    fig.add_hline(y=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    layout = get_layout()
    layout['title'] = dict(text=f'{ticker}', font=dict(size=14, color='#a1a1aa'), x=0, xanchor='left')
    layout['xaxis']['title'] = 'Holding Period'
    layout['yaxis']['title'] = 'Probability (%)'
    layout['yaxis']['range'] = [0, 105]
    layout['height'] = 420
    layout['legend'] = dict(
        orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        bgcolor='rgba(0,0,0,0)', font=dict(size=11)
    )
    layout['hovermode'] = 'x unified'
    
    fig.update_layout(**layout)
    return fig


def crear_barras(df: pd.DataFrame) -> go.Figure:
    labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels, y=df['prob_ganancia'], name='Win',
        marker=dict(color=COLORS['positive'], opacity=0.85),
        text=[f"{p:.0f}" for p in df['prob_ganancia']],
        textposition='inside', textfont=dict(size=10, color='white'),
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=labels, y=df['prob_perdida'], name='Loss',
        marker=dict(color=COLORS['negative'], opacity=0.85),
        text=[f"{p:.0f}" for p in df['prob_perdida']],
        textposition='inside', textfont=dict(size=10, color='white'),
        hovertemplate='%{y:.1f}%<extra></extra>'
    ))
    
    layout = get_layout()
    layout['barmode'] = 'stack'
    layout['height'] = 320
    layout['showlegend'] = False
    layout['margin'] = dict(l=48, r=24, t=24, b=48)
    
    fig.update_layout(**layout)
    return fig


def crear_comparativa(datos: pd.DataFrame, tickers: list, periodo: int) -> go.Figure:
    resultados = []
    for t in tickers:
        if t in datos.columns:
            ret = datos[t].pct_change().dropna()
            if len(ret) >= periodo:
                cum = (1 + ret).rolling(window=periodo).apply(np.prod, raw=True).dropna() - 1
                prob = (cum > 0).sum() / len(cum) * 100
                resultados.append({'ticker': t, 'prob': prob})
    
    if not resultados:
        return None
    
    df = pd.DataFrame(resultados).sort_values('prob', ascending=True)
    colors = [COLORS['positive'] if p >= 50 else COLORS['negative'] for p in df['prob']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['prob'], y=df['ticker'], orientation='h',
        marker=dict(color=colors, opacity=0.85),
        text=[f"{p:.1f}%" for p in df['prob']],
        textposition='outside', textfont=dict(size=10, color='#a1a1aa'),
        hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra></extra>'
    ))
    
    fig.add_vline(x=50, line=dict(color='#3f3f46', width=1, dash='dot'))
    
    layout = get_layout()
    layout['height'] = max(200, len(df) * 36)
    layout['margin'] = dict(l=80, r=60, t=24, b=24)
    layout['xaxis']['range'] = [0, 105]
    layout['showlegend'] = False
    
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APP
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
        
        tickers = [t.strip().upper() for t in tickers_input.replace(',', ' ').split() if t.strip()]
        if not tickers:
            tickers = ['^GSPC']
        
        ticker_principal = st.selectbox("Principal", tickers)
        
        anio = st.slider("Desde", 1980, 2020, 2000)
    
    # Cargar datos
    with st.spinner(''):
        datos = cargar_datos(tickers, f'{anio}-01-01')
        if datos.empty or ticker_principal not in datos.columns:
            st.error("Error cargando datos")
            st.stop()
        retornos = datos[ticker_principal].pct_change().dropna()
    
    # Calcular
    periodos = [1, 5, 20, 60, 120, 252, 504, 1260, 2520, 3780, 5040]
    periodos = [p for p in periodos if p <= len(retornos) - 1]
    df_prob = calcular_probabilidades(retornos, periodos)
    
    if df_prob.empty:
        st.warning("Datos insuficientes")
        st.stop()
    
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
    
    # Métricas
    p1y = df_prob[df_prob['periodo'] == 252]['prob_ganancia'].values
    p5y = df_prob[df_prob['periodo'] == 1260]['prob_ganancia'].values
    p10y = df_prob[df_prob['periodo'] == 2520]['prob_ganancia'].values
    years = len(retornos) / 252
    
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-label">1 Year Win Rate</div>
            <div class="metric-value positive">{p1y[0]:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">5 Year Win Rate</div>
            <div class="metric-value positive">{p5y[0]:.1f}% </div>
        </div>
        <div class="metric">
            <div class="metric-label">10 Year Win Rate</div>
            <div class="metric-value positive">{p10y[0]:.1f}% </div>
        </div>
        <div class="metric">
            <div class="metric-label">Data Range</div>
            <div class="metric-value accent">{years:.0f} years</div>
        </div>
    </div>
    """, unsafe_allow_html=True) if len(p1y) and len(p5y) and len(p10y) else None
    
    # Gráfico principal
    fig = crear_grafico_principal(df_prob, ticker_principal)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Segunda fila
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
        fig_bars = crear_barras(df_prob)
        st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown('<div class="section-title">Data</div>', unsafe_allow_html=True)
        
        labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y'][:len(df_prob)]
        
        table_html = '<table class="data-table"><thead><tr><th>Period</th><th>Win</th><th>Loss</th></tr></thead><tbody>'
        for i, row in df_prob.iterrows():
            win_class = 'positive' if row['prob_ganancia'] >= 50 else 'negative'
            table_html += f"""
            <tr>
                <td>{labels[i]}</td>
                <td><span class="pill {win_class}">{row['prob_ganancia']:.1f}%</span></td>
                <td>{row['prob_perdida']:.1f}%</td>
            </tr>
            """
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)
    
    # Comparativa (solo si hay múltiples tickers)
    if len(tickers) > 1:
        st.markdown('<div class="section-title">Comparison · 1 Year</div>', unsafe_allow_html=True)
        fig_comp = crear_comparativa(datos, tickers, 252)
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

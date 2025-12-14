# -*- coding: utf-8 -*-
"""
Probabilidad de Ganancia/Pérdida en los Mercados
Desarrollado por bquantfinance.com | @Gsnchez
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Probabilidad de Ganancia | BQuant Finance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0b0b0f;
        --bg-card: #13131a;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-cyan: #06b6d4;
        --text-primary: #f1f5f9;
        --text-muted: #64748b;
        --border: #1e293b;
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    h1, h2, h3, p, span, label {
        font-family: 'Inter', sans-serif !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    
    .hero-title {
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-muted);
        font-weight: 400;
    }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        flex: 1;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value.green { color: var(--accent-green); }
    .metric-value.red { color: var(--accent-red); }
    .metric-value.cyan { color: var(--accent-cyan); }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(16, 185, 129, 0.04) 100%);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .insight-title {
        color: var(--accent-cyan);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    
    .insight-text {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    
    .footer a {
        color: var(--accent-cyan);
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: var(--accent-cyan) !important;
        font-size: 1rem !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COLORES Y TEMA
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'green': '#10b981',
    'red': '#ef4444',
    'cyan': '#06b6d4',
    'gray': '#64748b',
    'bg': '#13131a'
}

def get_plotly_layout():
    """Devuelve configuración base de Plotly"""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#f1f5f9', size=12),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#64748b'),
            title_font=dict(color='#64748b')
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#64748b'),
            title_font=dict(color='#64748b')
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            font_size=12,
            font_family='Inter'
        )
    )

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos(tickers: list, fecha_inicio: str) -> pd.DataFrame:
    """Cargar datos de Yahoo Finance"""
    try:
        data = yf.download(tickers, fecha_inicio, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data
    except Exception:
        return pd.DataFrame()


def calcular_retornos_acumulados(retornos: pd.Series, ventana: int) -> pd.Series:
    """Calcular retornos acumulados usando capitalización compuesta (método correcto)"""
    return (1 + retornos).rolling(window=ventana).apply(np.prod, raw=True).dropna() - 1


def calcular_probabilidades(retornos: pd.Series, periodos: list) -> pd.DataFrame:
    """Calcular probabilidades de ganancia/pérdida para cada periodo"""
    resultados = []
    
    for periodo in periodos:
        if len(retornos) >= periodo:
            cum_ret = calcular_retornos_acumulados(retornos, periodo)
            n_obs = len(cum_ret)
            prob_gan = (cum_ret > 0).sum() / n_obs * 100
            prob_per = 100 - prob_gan
            ret_medio = cum_ret.mean() * 100
            
            resultados.append({
                'periodo': periodo,
                'prob_ganancia': round(prob_gan, 1),
                'prob_perdida': round(prob_per, 1),
                'retorno_medio': round(ret_medio, 2),
                'n_observaciones': n_obs
            })
    
    return pd.DataFrame(resultados)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def crear_grafico_probabilidades(df_prob: pd.DataFrame, ticker: str) -> go.Figure:
    """Gráfico principal de probabilidades de ganancia/pérdida"""
    
    etiquetas = ['1D', '1S', '1M', '3M', '6M', '1A', '2A', '5A', '10A', '15A', '20A'][:len(df_prob)]
    
    fig = go.Figure()
    
    # Área de ganancia
    fig.add_trace(go.Scatter(
        x=etiquetas,
        y=df_prob['prob_ganancia'],
        name='Ganancia',
        mode='lines+markers',
        line=dict(color=COLORS['green'], width=3),
        marker=dict(size=10, color=COLORS['green'], line=dict(width=2, color='#0b0b0f')),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)',
        hovertemplate='<b>%{x}</b><br>Prob. Ganancia: %{y:.1f}%<extra></extra>'
    ))
    
    # Línea de pérdida
    fig.add_trace(go.Scatter(
        x=etiquetas,
        y=df_prob['prob_perdida'],
        name='Pérdida',
        mode='lines+markers',
        line=dict(color=COLORS['red'], width=3),
        marker=dict(size=10, color=COLORS['red'], line=dict(width=2, color='#0b0b0f')),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
        hovertemplate='<b>%{x}</b><br>Prob. Pérdida: %{y:.1f}%<extra></extra>'
    ))
    
    # Línea de 50%
    fig.add_hline(y=50, line=dict(color='#475569', width=1, dash='dot'))
    
    layout = get_plotly_layout()
    layout.update(
        title=dict(
            text=f'Probabilidad de Ganancia vs Pérdida — {ticker}',
            font=dict(size=18, color='#f1f5f9'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Periodo de Tenencia',
        yaxis_title='Probabilidad (%)',
        yaxis=dict(**layout['yaxis'], range=[0, 105]),
        height=480,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        ),
        hovermode='x unified'
    )
    fig.update_layout(**layout)
    
    return fig


def crear_heatmap(df_prob: pd.DataFrame, ticker: str) -> go.Figure:
    """Mapa de calor de probabilidades"""
    
    etiquetas = ['1 Día', '1 Semana', '1 Mes', '3 Meses', '6 Meses', 
                 '1 Año', '2 Años', '5 Años', '10 Años', '15 Años', '20 Años'][:len(df_prob)]
    
    z_data = [[p] for p in df_prob['prob_ganancia']]
    
    fig = go.Figure(data=go.Heatmap(
        z=df_prob['prob_ganancia'].values.reshape(-1, 1),
        y=etiquetas,
        x=['Prob. Ganancia'],
        colorscale=[
            [0.0, COLORS['red']],
            [0.5, '#1e293b'],
            [1.0, COLORS['green']]
        ],
        zmin=0,
        zmax=100,
        zmid=50,
        text=[[f'{p:.1f}%'] for p in df_prob['prob_ganancia']],
        texttemplate='%{text}',
        textfont=dict(size=14, color='#f1f5f9', family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>Probabilidad: %{z:.1f}%<extra></extra>',
        showscale=False
    ))
    
    layout = get_plotly_layout()
    layout.update(
        title=dict(
            text=f'Probabilidad de Ganancia por Periodo — {ticker}',
            font=dict(size=16, color='#f1f5f9'),
            x=0.5,
            xanchor='center'
        ),
        height=450,
        xaxis=dict(showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color='#94a3b8'))
    )
    fig.update_layout(**layout)
    
    return fig


def crear_grafico_barras(df_prob: pd.DataFrame, ticker: str) -> go.Figure:
    """Gráfico de barras apiladas"""
    
    etiquetas = ['1D', '1S', '1M', '3M', '6M', '1A', '2A', '5A', '10A', '15A', '20A'][:len(df_prob)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=etiquetas,
        y=df_prob['prob_ganancia'],
        name='Ganancia',
        marker=dict(color=COLORS['green'], opacity=0.9),
        text=[f"{p:.0f}%" for p in df_prob['prob_ganancia']],
        textposition='inside',
        textfont=dict(color='white', size=11, family='JetBrains Mono'),
        hovertemplate='<b>%{x}</b><br>Ganancia: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=etiquetas,
        y=df_prob['prob_perdida'],
        name='Pérdida',
        marker=dict(color=COLORS['red'], opacity=0.9),
        text=[f"{p:.0f}%" for p in df_prob['prob_perdida']],
        textposition='inside',
        textfont=dict(color='white', size=11, family='JetBrains Mono'),
        hovertemplate='<b>%{x}</b><br>Pérdida: %{y:.1f}%<extra></extra>'
    ))
    
    layout = get_plotly_layout()
    layout.update(
        title=dict(
            text=f'Distribución Ganancia/Pérdida — {ticker}',
            font=dict(size=16, color='#f1f5f9'),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        xaxis_title='Periodo de Tenencia',
        yaxis_title='Probabilidad (%)',
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        )
    )
    fig.update_layout(**layout)
    
    return fig


def crear_comparativa_activos(datos: pd.DataFrame, tickers: list, periodo: int) -> go.Figure:
    """Comparativa de probabilidades entre activos"""
    
    resultados = []
    for ticker in tickers:
        if ticker in datos.columns:
            retornos = datos[ticker].pct_change().dropna()
            if len(retornos) >= periodo:
                cum_ret = calcular_retornos_acumulados(retornos, periodo)
                prob_gan = (cum_ret > 0).sum() / len(cum_ret) * 100
                resultados.append({'ticker': ticker, 'prob': prob_gan})
    
    if not resultados:
        return None
    
    df = pd.DataFrame(resultados).sort_values('prob', ascending=True)
    
    colores = [COLORS['green'] if p >= 50 else COLORS['red'] for p in df['prob']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['prob'],
        y=df['ticker'],
        orientation='h',
        marker=dict(color=colores, opacity=0.85),
        text=[f"{p:.1f}%" for p in df['prob']],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=11, family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>Prob. Ganancia: %{x:.1f}%<extra></extra>'
    ))
    
    # Línea de 50%
    fig.add_vline(x=50, line=dict(color='#475569', width=2, dash='dot'))
    
    periodo_texto = {
        252: '1 Año', 504: '2 Años', 1260: '5 Años', 20: '1 Mes', 60: '3 Meses'
    }.get(periodo, f'{periodo} días')
    
    layout = get_plotly_layout()
    layout.update(
        title=dict(
            text=f'Probabilidad de Ganancia a {periodo_texto}',
            font=dict(size=16, color='#f1f5f9'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(**layout['xaxis'], range=[0, 105]),
        xaxis_title='Probabilidad (%)',
        yaxis_title='',
        height=max(350, len(df) * 40),
        showlegend=False
    )
    fig.update_layout(**layout)
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    
    # Header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📊 Probabilidad de Ganancia</div>
        <div class="hero-subtitle">Análisis histórico de rentabilidad por periodo de inversión</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        tickers_disponibles = {
            'Índices': ['^GSPC', '^DJI', '^IXIC'],
            'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VOO'],
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'Finanzas': ['JPM', 'BAC', 'GS', 'V', 'MA']
        }
        
        categoria = st.selectbox("Categoría", list(tickers_disponibles.keys()))
        
        tickers = st.multiselect(
            "Activos",
            options=tickers_disponibles[categoria],
            default=tickers_disponibles[categoria][:3]
        )
        
        if not tickers:
            tickers = ['^GSPC']
        
        ticker_principal = st.selectbox("Activo Principal", tickers)
        
        st.markdown("---")
        
        anio_inicio = st.slider("Desde", 1990, 2020, 2000)
        
        st.markdown("---")
        st.markdown("### 📊 Comparativa")
        periodo_comp = st.selectbox(
            "Periodo",
            options=['1 Mes', '3 Meses', '1 Año', '2 Años', '5 Años'],
            index=2
        )
        
        mapa_periodos = {
            '1 Mes': 20, '3 Meses': 60, '1 Año': 252, '2 Años': 504, '5 Años': 1260
        }
        periodo_seleccionado = mapa_periodos[periodo_comp]
    
    # Cargar datos
    with st.spinner(''):
        datos = cargar_datos(tickers, f'{anio_inicio}-01-01')
        
        if datos.empty or ticker_principal not in datos.columns:
            st.error("Error al cargar datos. Intenta con otros activos.")
            st.stop()
        
        retornos = datos[ticker_principal].pct_change().dropna()
    
    # Periodos de análisis
    periodos = [1, 5, 20, 60, 120, 252, 504, 1260, 2520, 3780, 5040]
    periodos = [p for p in periodos if p <= len(retornos) - 1]
    
    # Calcular probabilidades
    df_prob = calcular_probabilidades(retornos, periodos)
    
    if df_prob.empty:
        st.warning("No hay suficientes datos para el análisis.")
        st.stop()
    
    # Métricas principales
    prob_1a = df_prob[df_prob['periodo'] == 252]['prob_ganancia'].values
    prob_5a = df_prob[df_prob['periodo'] == 1260]['prob_ganancia'].values
    prob_10a = df_prob[df_prob['periodo'] == 2520]['prob_ganancia'].values
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = f"{prob_1a[0]:.1f}%" if len(prob_1a) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prob. Ganancia 1 Año</div>
            <div class="metric-value green">{val}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        val = f"{prob_5a[0]:.1f}%" if len(prob_5a) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prob. Ganancia 5 Años</div>
            <div class="metric-value green">{val}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        val = f"{prob_10a[0]:.1f}%" if len(prob_10a) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prob. Ganancia 10 Años</div>
            <div class="metric-value green">{val}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        n_years = len(retornos) / 252
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Datos Históricos</div>
            <div class="metric-value cyan">{n_years:.0f} años</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico principal
    fig_principal = crear_grafico_probabilidades(df_prob, ticker_principal)
    st.plotly_chart(fig_principal, use_container_width=True, config={'displayModeBar': False})
    
    # Segunda fila de gráficos
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        fig_barras = crear_grafico_barras(df_prob, ticker_principal)
        st.plotly_chart(fig_barras, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        fig_heatmap = crear_heatmap(df_prob, ticker_principal)
        st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})
    
    # Insight box
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">💡 Conclusión</div>
        <div class="insight-text">
            A mayor horizonte temporal, mayor es la probabilidad de obtener rentabilidad positiva.
            Históricamente, mantener inversiones a largo plazo (>5 años) ha resultado en ganancias 
            en la gran mayoría de los casos para índices diversificados.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparativa entre activos
    if len(tickers) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Comparativa entre Activos")
        
        fig_comp = crear_comparativa_activos(datos, tickers, periodo_seleccionado)
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})
    
    # Footer
    st.markdown("""
    <div class="footer">
        <a href="https://bquantfinance.com" target="_blank">bquantfinance.com</a> · 
        <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a>
        <br><span style="opacity: 0.6;">Datos: Yahoo Finance · Retornos calculados con capitalización compuesta</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

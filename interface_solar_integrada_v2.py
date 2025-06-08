import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime, date, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🌞 Predição Solar - NASA Data",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .nasa-data-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Localização fixa conforme solicitado
LATITUDE_FIXA = -16.505517
LONGITUDE_FIXA = -50.385796
LOCAL_NOME = "SL - Willian"

class SolarRNA(nn.Module):
    """Mesma arquitetura do modelo treinado SEM normalização"""
    def __init__(self):
        super(SolarRNA, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado SEM normalização"""
    try:
        checkpoint = torch.load('modelo_sem_normalizacao.pth', map_location='cpu', weights_only=False)
        
        modelo = SolarRNA()
        # O modelo foi salvo com a chave 'model', não 'model_state_dict'
        modelo.load_state_dict(checkpoint['model'])
        modelo.eval()
        
        st.success("✅ Modelo carregado com sucesso!")
        return modelo
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

def obter_dados_nasa_power_melhorado(latitude, longitude, data_inicio, data_fim):
    """
    Versão melhorada da NASA POWER adaptada do ColetaAPI.py
    """
    configuracoes = [
        {
            'community': 'SB',
            'parametros': ['PSM_GHI', 'PSM_DHI', 'PSM_DNI', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M']
        },
        {
            'community': 'AG',
            'parametros': ['ALLSKY_SFC_SW_DWN', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PRECTOTCORR']
        },
        {
            'community': 'RE',
            'parametros': ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M']
        }
    ]
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    for i, config in enumerate(configuracoes):
        try:
            params = {
                'start': data_inicio,
                'end': data_fim,
                'latitude': latitude,
                'longitude': longitude,
                'community': config['community'],
                'parameters': ','.join(config['parametros']),
                'format': 'JSON',
                'user': 'anonymous'
            }
            
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            dados_parametros = data['properties']['parameter']
            
            df = pd.DataFrame(dados_parametros)
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
            df = df.replace(-999, pd.NA)
            
            if len(df) > 0:
                colunas_irradiacao = [col for col in df.columns if any(x in col.upper() for x in ['GHI', 'SW_DWN'])]
                
                if colunas_irradiacao:
                    col_principal = colunas_irradiacao[0]
                    valores_validos = df[col_principal].notna().sum()
                    percentual_valido = (valores_validos / len(df)) * 100
                    
                    if percentual_valido > 50:
                        # Padronizar nomes das colunas
                        mapeamento_colunas = {
                            'PSM_GHI': 'Irradiacao_Global_kWh_m2',
                            'PSM_DHI': 'Irradiacao_Difusa_kWh_m2',
                            'PSM_DNI': 'Irradiacao_Direta_kWh_m2',
                            'ALLSKY_SFC_SW_DWN': 'Irradiacao_Global_kWh_m2',
                            'CLRSKY_SFC_SW_DWN': 'Irradiacao_CeuLimpo_kWh_m2',
                            'T2M': 'Temperatura_Media_C',
                            'T2M_MAX': 'Temperatura_Maxima_C',
                            'T2M_MIN': 'Temperatura_Minima_C',
                            'RH2M': 'Umidade_Relativa_%',
                            'WS2M': 'Velocidade_Vento_ms',
                            'PRECTOTCORR': 'Precipitacao_mm'
                        }
                        
                        df = df.rename(columns=mapeamento_colunas)
                        df['Data'] = df.index.date
                        df['Fonte'] = f'NASA_POWER_{config["community"]}'
                        
                        return df
            
            time.sleep(1)
            
        except Exception as e:
            continue
    
    return None

def criar_mapa_localização_fixa():
    """Cria mapa com localização fixa e marcador destacado"""
    # Criar mapa centrado na localização fixa
    m = folium.Map(
        location=[LATITUDE_FIXA, LONGITUDE_FIXA],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Adicionar marcador destacado na localização fixa
    folium.Marker(
        location=[LATITUDE_FIXA, LONGITUDE_FIXA],
        popup=f"""
        <div style="font-family: Arial; font-size: 14px;">
            <b>{LOCAL_NOME}</b><br>
            📍 Lat: {LATITUDE_FIXA}<br>
            📍 Lon: {LONGITUDE_FIXA}<br>
            🌞 Usina Solar Fotovoltaica<br>
            ⚡ Potência: 8,8 kWp
        </div>
        """,
        icon=folium.Icon(
            color='red',
            icon='sun',
            prefix='fa'
        )
    ).add_to(m)
    
    # Adicionar círculo de destaque
    folium.Circle(
        location=[LATITUDE_FIXA, LONGITUDE_FIXA],
        radius=200,
        popup=f'Usina Solar - {LOCAL_NOME} (8,8 kWp)',
        color='red',
        fillColor='red',
        fillOpacity=0.2
    ).add_to(m)
    
    return m

def fazer_predicao(modelo, irradiacao, temperatura):
    """Faz predição usando o modelo"""
    if modelo is None:
        return None
    
    try:
        # Preparar dados de entrada (SEM normalização)
        entrada = torch.tensor([[irradiacao, temperatura]], dtype=torch.float32)
        
        with torch.no_grad():
            predicao = modelo(entrada)
            return float(predicao.item())
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        return None

def criar_superficie_predicao(modelo):
    """Cria gráfico 3D da superfície de predição"""
    if modelo is None:
        return None
    
    # Range de valores
    irradiacao_range = np.linspace(0, 10, 30)
    temperatura_range = np.linspace(10, 40, 30)
    
    # Criar grade
    I, T = np.meshgrid(irradiacao_range, temperatura_range)
    
    # Calcular predições para toda a grade
    predicoes = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            pred = fazer_predicao(modelo, I[i,j], T[i,j])
            predicoes[i,j] = pred if pred else 0
    
    # Criar gráfico 3D
    fig = go.Figure(data=[
        go.Surface(
            x=I, y=T, z=predicoes,
            colorscale='Viridis',
            name='Predição de Energia'
        )
    ])
    
    fig.update_layout(
        title='🌞 Superfície de Predição - Energia Solar',
        scene=dict(
            xaxis_title='Irradiação (kWh/m²)',
            yaxis_title='Temperatura (°C)',
            zaxis_title='Energia Predita (kWh)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        height=500
    )
    
    return fig

def main():
    # Header principal
    st.markdown(f"""
    <div class="main-header">
        🌞 PREDIÇÃO SOLAR - USINA FOTOVOLTAICA 🛰️
        <br><small>Análise baseada em dados NASA POWER</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Informações sobre o modelo
    st.markdown(f"""
    <div class="info-card">
        <h4>📊 Informações do Modelo</h4>
        <p><strong>Treinamento baseado em:</strong> Usina Solar Fotovoltaica de 8,8 kWp</p>
        <p><strong>Localização:</strong> {LOCAL_NOME} ({LATITUDE_FIXA}°, {LONGITUDE_FIXA}°)</p>
        <p><strong>Fonte de Dados:</strong> NASA POWER (Satélite)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar modelo
    modelo = carregar_modelo()
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Mapa com localização fixa
        st.subheader("🗺️ Localização da Usina Solar")
        mapa = criar_mapa_localização_fixa()
        st_folium(mapa, width=700, height=400)
    
    with col2:
        st.subheader("⚙️ Configurações de Análise")
        
        st.info("🌐 **Fonte dos Dados:** NASA POWER API")
        
        # Campos de data melhorados - permitir digitação
        col_data1, col_data2 = st.columns(2)
        with col_data1:
            data_inicio = st.date_input(
                "📅 Data Início",
                value=date(2024, 1, 1),
                min_value=date(1981, 1, 1),  # NASA POWER vai até 1981
                max_value=date.today(),
                help="Digite a data ou clique no calendário"
            )
        with col_data2:
            data_fim = st.date_input(
                "📅 Data Fim",
                value=date(2024, 6, 30),
                min_value=date(1981, 1, 1),
                max_value=date.today(),
                help="Digite a data ou clique no calendário"
            )
        
        # Validação de período
        if data_inicio > data_fim:
            st.error("❌ Data de início deve ser anterior à data de fim!")
        elif (data_fim - data_inicio).days > 3650:  # 10 anos
            st.warning("⚠️ Período muito longo. Recomendado máximo de 10 anos.")
        
        # Botão para buscar dados
        if st.button("🔍 Buscar Dados NASA POWER", type="primary", disabled=(data_inicio > data_fim)):
            with st.spinner("🌍 Buscando dados meteorológicos da NASA..."):
                data_inicio_str = data_inicio.strftime("%Y%m%d")
                data_fim_str = data_fim.strftime("%Y%m%d")
                
                dados = obter_dados_nasa_power_melhorado(
                    LATITUDE_FIXA, LONGITUDE_FIXA, 
                    data_inicio_str, data_fim_str
                )
                
                if dados is not None and len(dados) > 0:
                    st.session_state['dados_meteorologicos'] = dados
                    st.success(f"✅ {len(dados)} dias de dados obtidos com sucesso!")
                    
                    # Mostrar período real dos dados
                    data_real_inicio = dados['Data'].min()
                    data_real_fim = dados['Data'].max()
                    st.info(f"📅 Período dos dados: {data_real_inicio} até {data_real_fim}")
                else:
                    st.error("❌ Não foi possível obter dados da NASA POWER")
    
    # Análise dos dados se disponíveis
    if 'dados_meteorologicos' in st.session_state:
        dados = st.session_state['dados_meteorologicos']
        
        st.markdown("---")
        st.subheader("📊 Análise dos Dados NASA POWER")
        
        # Estatísticas básicas
        col1, col2, col3, col4 = st.columns(4)
        
        if 'Irradiacao_Global_kWh_m2' in dados.columns:
            irrad_media = dados['Irradiacao_Global_kWh_m2'].mean()
            with col1:
                st.metric("☀️ Irradiação Média", f"{irrad_media:.2f} kWh/m²")
        
        if 'Temperatura_Media_C' in dados.columns:
            temp_media = dados['Temperatura_Media_C'].mean()
            with col2:
                st.metric("🌡️ Temperatura Média", f"{temp_media:.1f} °C")
        
        with col3:
            st.metric("📅 Dias de Dados", len(dados))
        
        with col4:
            fonte_atual = dados['Fonte'].iloc[0] if 'Fonte' in dados.columns else "NASA_POWER"
            st.metric("🛰️ Fonte", fonte_atual.replace('NASA_POWER_', ''))
        
        # Gráficos temporais
        st.subheader("📈 Tendências Temporais")
        
        if 'Data' in dados.columns and 'Irradiacao_Global_kWh_m2' in dados.columns:
            fig_temporal = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Irradiação Solar (kWh/m²)', 'Temperatura (°C)'),
                vertical_spacing=0.1
            )
            
            # Irradiação
            fig_temporal.add_trace(
                go.Scatter(
                    x=dados['Data'],
                    y=dados['Irradiacao_Global_kWh_m2'],
                    name='Irradiação Global',
                    line=dict(color='orange', width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Temperatura
            if 'Temperatura_Media_C' in dados.columns:
                fig_temporal.add_trace(
                    go.Scatter(
                        x=dados['Data'],
                        y=dados['Temperatura_Media_C'],
                        name='Temperatura Média',
                        line=dict(color='red', width=2),
                        mode='lines'
                    ),
                    row=2, col=1
                )
            
            fig_temporal.update_layout(
                height=600, 
                showlegend=True,
                title_text="🌞 Dados Meteorológicos da NASA POWER"
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Predições com dados reais
        st.subheader("🤖 Predições com Dados Reais da NASA")
        
        if modelo and 'Irradiacao_Global_kWh_m2' in dados.columns and 'Temperatura_Media_C' in dados.columns:
            # Calcular predições para todos os dados
            predicoes = []
            for _, row in dados.iterrows():
                pred = fazer_predicao(modelo, row['Irradiacao_Global_kWh_m2'], row['Temperatura_Media_C'])
                predicoes.append(pred if pred else 0)
            
            dados['Energia_Predita_kWh'] = predicoes
            
            # Estatísticas das predições
            energia_media = np.mean(predicoes)
            energia_total_mensal = energia_media * 30
            energia_total_anual = energia_media * 365
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⚡ Energia Média/Dia", f"{energia_media:.2f} kWh")
            with col2:
                st.metric("📊 Estimativa Mensal", f"{energia_total_mensal:.1f} kWh")
            with col3:
                st.metric("📈 Estimativa Anual", f"{energia_total_anual:.0f} kWh")
            with col4:
                co2_evitado = energia_total_anual * 0.4  # kg CO2 evitado por kWh
                st.metric("🌱 CO2 Evitado/Ano", f"{co2_evitado:.0f} kg")
            
            # Gráfico de predições
            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=dados['Data'],
                    y=predicoes,
                    name='Energia Predita (kWh)',
                    line=dict(color='green', width=2),
                    mode='lines+markers',
                    marker=dict(size=4)
                )
            )
            
            fig_pred.update_layout(
                title='🔮 Predições de Energia Solar - Usina 8,8 kWp',
                xaxis_title='Data',
                yaxis_title='Energia Produzida (kWh)',
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Estatísticas mensais se houver dados suficientes
            if len(dados) > 30:
                dados['Mes'] = pd.to_datetime(dados['Data']).dt.to_period('M')
                dados_mensais = dados.groupby('Mes').agg({
                    'Energia_Predita_kWh': ['mean', 'sum'],
                    'Irradiacao_Global_kWh_m2': 'mean',
                    'Temperatura_Media_C': 'mean'
                }).round(2)
                
                st.subheader("📅 Resumo Mensal")
                
                # Flatten column names
                dados_mensais.columns = ['_'.join(col).strip() if col[1] else col[0] for col in dados_mensais.columns.values]
                
                st.dataframe(
                    dados_mensais,
                    use_container_width=True,
                    column_config={
                        "Energia_Predita_kWh_mean": st.column_config.NumberColumn(
                            "Energia Média/Dia (kWh)",
                            format="%.2f"
                        ),
                        "Energia_Predita_kWh_sum": st.column_config.NumberColumn(
                            "Energia Total/Mês (kWh)",
                            format="%.1f"
                        ),
                        "Irradiacao_Global_kWh_m2_mean": st.column_config.NumberColumn(
                            "Irradiação Média (kWh/m²)",
                            format="%.2f"
                        ),
                        "Temperatura_Media_C_mean": st.column_config.NumberColumn(
                            "Temperatura Média (°C)",
                            format="%.1f"
                        )
                    }
                )
    
    # Análise interativa manual
    st.markdown("---")
    st.subheader("🎛️ Simulador Interativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ajuste os parâmetros para simular diferentes condições:**")
        
        irradiacao_manual = st.slider(
            "☀️ Irradiação Solar (kWh/m²)",
            min_value=0.0, max_value=10.0, value=5.0, step=0.1,
            help="Irradiação solar global horizontal"
        )
        
        temperatura_manual = st.slider(
            "🌡️ Temperatura (°C)",
            min_value=10.0, max_value=45.0, value=25.0, step=0.5,
            help="Temperatura ambiente"
        )
        
        if modelo:
            predicao_manual = fazer_predicao(modelo, irradiacao_manual, temperatura_manual)
            if predicao_manual:
                # Calcular percentual da capacidade
                capacidade_maxima = 8.8  # kWp da usina
                percentual_capacidade = (predicao_manual / capacidade_maxima) * 100
                
                st.markdown(f"""
                <div class="nasa-data-card">
                    <h3>⚡ Predição de Energia</h3>
                    <h2>{predicao_manual:.3f} kWh</h2>
                    <p><strong>Irradiação:</strong> {irradiacao_manual} kWh/m²</p>
                    <p><strong>Temperatura:</strong> {temperatura_manual}°C</p>
                    <p><strong>Capacidade utilizada:</strong> {percentual_capacidade:.1f}% dos 8,8 kWp</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Gráfico 3D da superfície de predição
        st.write("**Superfície de Predição 3D:**")
        fig_superficie = criar_superficie_predicao(modelo)
        if fig_superficie:
            st.plotly_chart(fig_superficie, use_container_width=True)

if __name__ == "__main__":
    main()

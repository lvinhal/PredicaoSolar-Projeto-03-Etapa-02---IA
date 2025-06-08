import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

def obter_dados_pvgis(latitude, longitude, ano=2024):
    """
    Obtém dados do PVGIS (Photovoltaic Geographical Information System)
    Fonte: Comissão Europeia - geralmente mais confiável que NASA POWER
    """
    base_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    
    params = {
        'lat': latitude,
        'lon': longitude,
        'startyear': ano,
        'endyear': ano,
        'outputformat': 'json',
        'components': 1,  # Incluir irradiação e temperatura
        'browser': 0
    }
    
    try:
        print(f"  Tentando PVGIS para ({latitude}, {longitude})...")
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Extrair dados horários
        dados_horarios = data['outputs']['hourly']
        
        # Converter para DataFrame
        df = pd.DataFrame(dados_horarios)
        
        # Converter timestamp para datetime
        df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
        df['date'] = df['datetime'].dt.date
        
        # Agregar dados diários
        df_diario = df.groupby('date').agg({
            'G(h)': 'sum',      # Irradiação global horizontal (Wh/m²)
            'Gd(h)': 'sum',     # Irradiação difusa horizontal (Wh/m²)
            'Gb(h)': 'sum',     # Irradiação direta horizontal (Wh/m²)
            'T2m': 'mean',      # Temperatura média (°C)
            'WS10m': 'mean',    # Velocidade do vento (m/s)
            'RH': 'mean'        # Umidade relativa (%)
        }).reset_index()
        
        # Converter Wh/m² para kWh/m²
        df_diario['Irradiacao_Global_kWh_m2'] = df_diario['G(h)'] / 1000
        df_diario['Irradiacao_Difusa_kWh_m2'] = df_diario['Gd(h)'] / 1000
        df_diario['Irradiacao_Direta_kWh_m2'] = df_diario['Gb(h)'] / 1000
        
        # Renomear colunas
        df_diario = df_diario.rename(columns={
            'date': 'Data',
            'T2m': 'Temperatura_Media_C',
            'WS10m': 'Velocidade_Vento_ms',
            'RH': 'Umidade_Relativa_%'
        })
        
        # Manter apenas colunas relevantes
        colunas_finais = ['Data', 'Irradiacao_Global_kWh_m2', 'Irradiacao_Difusa_kWh_m2', 
                         'Irradiacao_Direta_kWh_m2', 'Temperatura_Media_C', 
                         'Velocidade_Vento_ms', 'Umidade_Relativa_%']
        
        df_final = df_diario[colunas_finais].copy()
        df_final['Fonte'] = 'PVGIS'
        
        print(f"  ✓ PVGIS: {len(df_final)} dias obtidos")
        return df_final
        
    except Exception as e:
        print(f"  ✗ PVGIS falhou: {str(e)}")
        return None

def obter_dados_inmet_api(latitude, longitude, ano=2024):
    """
    Obtém dados do INMET (Instituto Nacional de Meteorologia)
    API oficial do governo brasileiro
    """
    # Encontrar estação mais próxima (simplificado)
    # Para implementação completa, seria necessário consultar a lista de estações
    
    # Estações principais em Goiás (aproximação)
    estacoes_goias = {
        'GOIANIA': 'A001',  # Código exemplo - precisa verificar códigos reais
        'BRASILIA': 'A003'
    }
    
    # Esta função seria implementada com a API real do INMET
    # Por limitações da API atual, retornamos None
    print(f"  INMET: API requer códigos específicos de estação")
    return None

def obter_dados_nasa_power_melhorado(latitude, longitude, data_inicio, data_fim):
    """
    Versão melhorada da NASA POWER com múltiplas tentativas
    """
    # Tentar diferentes comunidades/parâmetros
    configuracoes = [
        {
            'community': 'SB',  # Sustainable Buildings
            'parametros': ['PSM_GHI', 'PSM_DHI', 'PSM_DNI', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M']
        },
        {
            'community': 'AG',  # Agroclimatology  
            'parametros': ['ALLSKY_SFC_SW_DWN', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'WS2M', 'PRECTOTCORR']
        },
        {
            'community': 'RE',  # Renewable Energy
            'parametros': ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M']
        }
    ]
    
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    for i, config in enumerate(configuracoes):
        try:
            print(f"  Tentando NASA POWER config {i+1}/3 ({config['community']})...")
            
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
            
            # Verificar qualidade dos dados
            if len(df) > 0:
                # Verificar se temos dados de irradiação válidos
                colunas_irradiacao = [col for col in df.columns if any(x in col.upper() for x in ['GHI', 'SW_DWN'])]
                
                if colunas_irradiacao:
                    col_principal = colunas_irradiacao[0]
                    valores_validos = df[col_principal].notna().sum()
                    percentual_valido = (valores_validos / len(df)) * 100
                    
                    if percentual_valido > 50:  # Se mais de 50% dos dados são válidos
                        print(f"  ✓ NASA POWER {config['community']}: {percentual_valido:.1f}% dados válidos")
                        
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
            
            time.sleep(1)  # Pausa entre tentativas
            
        except Exception as e:
            print(f"  ✗ NASA POWER {config['community']} falhou: {str(e)}")
            continue
    
    print(f"  ✗ Todas as configurações NASA POWER falharam")
    return None

def obter_dados_openweather_historico(latitude, longitude, ano=2024):
    """
    OpenWeatherMap Historical API (requer API key gratuita)
    Comentado porque precisa de API key, mas deixo o código como exemplo
    """
    # API_KEY = "SUA_API_KEY_AQUI"  # Obter em openweathermap.org
    # 
    # if not API_KEY or API_KEY == "SUA_API_KEY_AQUI":
    #     print("  OpenWeather: API key necessária")
    #     return None
    # 
    # # Código para OpenWeatherMap seria implementado aqui
    # # com loop pelos dias do ano fazendo requisições
    
    print("  OpenWeather: Implementação requer API key")
    return None

def coletar_dados_multiplas_fontes():
    """
    Coleta dados usando múltiplas fontes e escolhe a melhor para cada local
    """
    
    # Coordenadas específicas solicitadas
    coordenadas = {
        'SL - Joao Estevao': (-16.505146, -50.386017),
        'SL - Willian': (-16.505517, -50.385796),
        'GYN - Euler': (-16.663361, -49.206463),
        'GYN - Adinirso': (-16.736145, -49.3071899)
    }
    
    # ====== CONFIGURAÇÃO DE DATAS ======
    # Altere estas linhas para mudar o período dos dados:
    
    # Para um ano completo:
    ano = 2024  # Altere para o ano desejado (ex: 2023, 2022, 2021...)
    # ata_inicio = f"{ano}0101"  # Primeiro dia do ano
    # data_fim = f"{ano}1231"     # Último dia do ano
    
    # OU para um período específico (descomente as linhas abaixo):
    data_inicio = "20250420"  # 1º de junho de 2023
    data_fim = "20250606"     # 31 de agosto de 2023
    # ano = 2024  # Ano correspondente
    
    todos_dados = {}
    
    for local, (lat, lon) in coordenadas.items():
        print(f"\n=== Processando {local} ===")
        
        melhores_dados = None
        melhor_fonte = None
        melhor_score = 0
        
        # Lista de funções para tentar (em ordem de preferência)
        fontes = [
            ('PVGIS', lambda: obter_dados_pvgis(lat, lon, ano)),
            ('NASA_POWER', lambda: obter_dados_nasa_power_melhorado(lat, lon, data_inicio, data_fim)),
            # ('INMET', lambda: obter_dados_inmet_api(lat, lon, ano)),
            # ('OpenWeather', lambda: obter_dados_openweather_historico(lat, lon, ano))
        ]
        
        for nome_fonte, funcao_fonte in fontes:
            try:
                dados = funcao_fonte()
                
                if dados is not None and len(dados) > 0:
                    # Calcular score de qualidade
                    score = calcular_score_qualidade(dados)
                    print(f"  {nome_fonte}: Score de qualidade = {score:.1f}")
                    
                    if score > melhor_score:
                        melhor_score = score
                        melhores_dados = dados.copy()
                        melhor_fonte = nome_fonte
                
                time.sleep(2)  # Pausa entre APIs
                
            except Exception as e:
                print(f"  ✗ {nome_fonte} falhou: {str(e)}")
        
        if melhores_dados is not None:
            print(f"  ✓ Melhor fonte para {local}: {melhor_fonte} (score: {melhor_score:.1f})")
            todos_dados[local] = melhores_dados
            
            # Adicionar relatório de qualidade
            relatorio = gerar_relatorio_detalhado(melhores_dados, local, melhor_fonte)
            todos_dados[f'{local}_Relatorio'] = relatorio
        else:
            print(f"  ✗ Nenhuma fonte funcionou para {local}")
    
    # Salvar dados
    if todos_dados:
        nome_arquivo = "dados_meteorologicos_multiplas_fontes_2024.xlsx"
        
        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            for nome_aba, df in todos_dados.items():
                nome_aba_excel = nome_aba[:31]
                df.to_excel(writer, sheet_name=nome_aba_excel, index=False)
        
        print(f"\n✓ Dados salvos em {nome_arquivo}")
        print(f"✓ Total de locais processados: {len([k for k in todos_dados.keys() if not k.endswith('_Relatorio')])}")
    else:
        print("\n✗ Nenhum dado foi coletado com sucesso")

def calcular_score_qualidade(df):
    """
    Calcula um score de qualidade dos dados (0-100)
    """
    if df is None or len(df) == 0:
        return 0
    
    score = 0
    peso_total = 0
    
    # Verificar presença e qualidade das colunas importantes
    colunas_importantes = {
        'Irradiacao_Global_kWh_m2': 40,  # Peso 40
        'Temperatura_Media_C': 25,       # Peso 25
        'Irradiacao_Difusa_kWh_m2': 15,  # Peso 15
        'Irradiacao_Direta_kWh_m2': 10,  # Peso 10
        'Umidade_Relativa_%': 5,         # Peso 5
        'Velocidade_Vento_ms': 5         # Peso 5
    }
    
    for coluna, peso in colunas_importantes.items():
        if coluna in df.columns:
            valores_validos = df[coluna].notna().sum()
            percentual_valido = (valores_validos / len(df)) * 100
            score += (percentual_valido * peso / 100)
        peso_total += peso
    
    # Bonus por ter mais dias de dados
    if len(df) >= 350:  # Ano quase completo
        score += 10
    elif len(df) >= 300:
        score += 5
    
    return min(score, 100)  # Máximo 100

def gerar_relatorio_detalhado(df, local, fonte):
    """
    Gera relatório detalhado da qualidade dos dados
    """
    relatorio = []
    relatorio.append(['RELATÓRIO DE QUALIDADE DOS DADOS'])
    relatorio.append([])
    relatorio.append(['Local:', local])
    relatorio.append(['Fonte:', fonte])
    relatorio.append(['Período:', f"{df['Data'].min()} a {df['Data'].max()}" if 'Data' in df.columns else 'N/A'])
    relatorio.append(['Total de dias:', len(df)])
    relatorio.append(['Score de qualidade:', f"{calcular_score_qualidade(df):.1f}/100"])
    relatorio.append([])
    relatorio.append(['DETALHES POR PARÂMETRO'])
    relatorio.append(['Parâmetro', 'Dados Válidos', 'Dados Ausentes', '% Completo', 'Valor Médio'])
    
    for coluna in df.columns:
        if coluna not in ['Data', 'Fonte'] and df[coluna].dtype in ['float64', 'int64']:
            validos = df[coluna].notna().sum()
            ausentes = df[coluna].isna().sum()
            percentual = (validos / len(df)) * 100
            media = df[coluna].mean() if validos > 0 else 'N/A'
            
            if isinstance(media, (int, float)):
                media = f"{media:.2f}"
            
            relatorio.append([coluna, validos, ausentes, f"{percentual:.1f}%", media])
    
    return pd.DataFrame(relatorio)

# Executar coleta
if __name__ == "__main__":
    coletar_dados_multiplas_fontes()
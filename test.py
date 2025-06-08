import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SimpleSolarRNA(nn.Module):
    """Mesma arquitetura do modelo treinado"""
    def __init__(self, input_size=13):
        super(SimpleSolarRNA, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def criar_features_engineered(irrad, temp):
    """Cria features engineered - DEVE SER IDÊNTICA AO TREINAMENTO"""
    
    # Features base
    features = [irrad, temp]
    
    # Features não-lineares simples
    features.extend([
        irrad ** 2,
        np.sqrt(max(irrad, 0)),
        temp ** 2
    ])
    
    # Interações físicas
    features.extend([
        irrad * temp,
        irrad / (temp + 1),  # Evitar divisão por zero
        irrad * max(0, 1 - 0.004 * (temp - 25))  # Eficiência térmica
    ])
    
    # Features de regime
    features.extend([
        irrad if irrad < 15 else 0,  # Irradiação baixa
        (irrad - 20) if irrad > 20 else 0,  # Irradiação alta
        (temp - 30) if temp > 30 else 0  # Temperatura alta
    ])
    
    # Features trigonométricas
    features.extend([
        np.sin(np.pi * irrad / 30),  # Normalizar por valor máximo típico
        np.cos(2 * np.pi * temp / 40)  # Normalizar por range típico
    ])
    
    return np.array(features, dtype=np.float32)

def carregar_modelo_solar():
    """Carrega o modelo treinado corrigido"""
    try:
        checkpoint = torch.load('modelo_solar_corrigido.pth', map_location='cpu', weights_only=False)
        
        input_size = checkpoint['input_size']
        modelo = SimpleSolarRNA(input_size=input_size)
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()
        
        scaler = checkpoint['scaler']
        target_name = checkpoint['target_name']
        test_mae = checkpoint.get('test_mae', 'N/A')
        
        print("✅ Modelo neural carregado com sucesso!")
        print(f"   Input size: {input_size}")
        print(f"   Target: {target_name}")
        print(f"   MAE esperado: {test_mae:.4f}" if test_mae != 'N/A' else "   MAE: N/A")
        
        return modelo, scaler, target_name
        
    except FileNotFoundError:
        print("❌ Arquivo 'modelo_solar_corrigido.pth' não encontrado!")
        print("   Execute primeiro o treinamento corrigido.")
        return None, None, None
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None, None, None

def prever_desempenho(modelo, scaler, irradiacao, temperatura, mostrar_detalhes=True):
    """
    Prediz desempenho usando o modelo neural treinado
    
    Retorna:
    - Desempenho como fator (ex: 1.0 = 100%)
    """
    try:
        # Criar features engineered
        features = criar_features_engineered(irradiacao, temperatura)
        
        # Normalizar usando o mesmo scaler do treinamento
        features_norm = scaler.transform(features.reshape(1, -1))
        
        # Converter para tensor
        entrada = torch.FloatTensor(features_norm)
        
        # Predição
        with torch.no_grad():
            resultado = modelo(entrada)
            desempenho = resultado.item()
        
        # Garantir que está no range razoável
        desempenho = max(0.4, min(1.6, desempenho))
        
        if mostrar_detalhes:
            print(f"\n⚡ PREDIÇÃO DE DESEMPENHO (RNA):")
            print(f"   Irradiação: {irradiacao:.2f} kWh/m²")
            print(f"   Temperatura: {temperatura:.1f}°C")
            print(f"   Fator previsto: {desempenho:.3f} ({desempenho*100:.1f}%)")
            
            # Contexto do resultado
            if desempenho >= 1.2:
                status = "🔥 EXCEPCIONAL"
            elif desempenho >= 1.0:
                status = "☀️ EXCELENTE"
            elif desempenho >= 0.85:
                status = "🌤️ MUITO BOM"
            elif desempenho >= 0.7:
                status = "☁️ BOM"
            elif desempenho >= 0.5:
                status = "🌥️ MODERADO"
            else:
                status = "🌧️ BAIXO"
            
            print(f"   Status: {status}")
            
            # Mostrar algumas features calculadas
            print(f"\n🔧 Features principais:")
            print(f"   Irrad²: {features[2]:.1f}")
            print(f"   Irrad×Temp: {features[5]:.1f}")
            print(f"   Eficiência térmica: {features[7]:.2f}")
        
        return desempenho
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        import traceback
        traceback.print_exc()
        return None

def calcular_geracao_estimada(desempenho_fator, potencia_instalada):
    """Converte fator de desempenho em geração kWh"""
    if desempenho_fator is None or potencia_instalada <= 0:
        return None
    
    # Assumindo ~5 horas de pico solar por dia
    horas_pico_solar = 5
    geracao_teorica = potencia_instalada * horas_pico_solar
    geracao_estimada = desempenho_fator * geracao_teorica
    
    return geracao_estimada

def teste_interativo():
    """Interface interativa para testar o modelo"""
    print("🌞 TESTADOR DE DESEMPENHO SOLAR (RNA)")
    print("="*45)
    
    modelo, scaler, target_name = carregar_modelo_solar()
    if modelo is None:
        return
    
    print("\n📝 INSTRUÇÕES:")
    print("   • Digite irradiação e temperatura")
    print("   • O modelo prediz fator de desempenho")
    print("   • 1.0 = 100% de eficiência esperada")
    print("   • Digite 'sair' para terminar")
    print("\n💡 RANGES TÍPICOS:")
    print("   • Irradiação: 10-28 kWh/m² (ótimo: 20-25)")
    print("   • Temperatura: 21-34°C (ótimo: 24-26)")
    
    while True:
        print("\n" + "-"*50)
        
        # Entrada da irradiação
        try:
            irrad_input = input("🌞 Irradiação solar (kWh/m²): ").strip()
            if irrad_input.lower() == 'sair':
                break
            irradiacao = float(irrad_input)
            
            if irradiacao < 0 or irradiacao > 50:
                print("⚠️  Irradiação fora do range típico (0-50)")
                
        except ValueError:
            print("❌ Digite um número válido para irradiação")
            continue
        
        # Entrada da temperatura
        try:
            temp_input = input("🌡️  Temperatura (°C): ").strip()
            temperatura = float(temp_input)
            
            if temperatura < 10 or temperatura > 50:
                print("⚠️  Temperatura fora do range típico (10-50°C)")
                
        except ValueError:
            print("❌ Digite um número válido para temperatura")
            continue
        
        # Fazer predição
        desempenho = prever_desempenho(modelo, scaler, irradiacao, temperatura)
        
        if desempenho is not None:
            # Perguntar sobre potência instalada
            try:
                potencia_input = input("\n🔌 Potência instalada (kW) [Enter para pular]: ").strip()
                
                if potencia_input:
                    potencia = float(potencia_input)
                    geracao = calcular_geracao_estimada(desempenho, potencia)
                    
                    if geracao is not None:
                        print(f"\n💡 GERAÇÃO ESTIMADA:")
                        print(f"   Potência instalada: {potencia:.1f} kW")
                        print(f"   Fator de desempenho: {desempenho:.3f}")
                        print(f"   Geração estimada: {geracao:.1f} kWh/dia")
                        print(f"   Geração mensal: {geracao * 30:.0f} kWh")
                        print(f"   Receita mensal (R$0,65/kWh): R$ {geracao * 30 * 0.65:.0f}")
                
            except ValueError:
                print("⚠️  Potência inválida - pulando cálculo de geração")

def teste_cenarios():
    """Testa cenários predefinidos"""
    print("\n🧪 TESTE DE CENÁRIOS PREDEFINIDOS")
    print("="*50)
    
    modelo, scaler, target_name = carregar_modelo_solar()
    if modelo is None:
        return
    
    # Cenários baseados nos dados reais
    cenarios = [
        (12.9, 24.7, "Exemplo real #1 do dataset"),
        (14.4, 24.9, "Exemplo real #2 do dataset"),
        (18.53, 24.32, "Exemplo real #3 do dataset"),
        (25.0, 25, "Dia muito ensolarado, temperatura ideal"),
        (20.0, 28, "Dia ensolarado, temperatura boa"),
        (15.0, 30, "Dia com boa irradiação, temperatura alta"),
        (10.0, 22, "Dia parcialmente nublado, temperatura baixa"),
        (27.85, 33.43, "Máximos do dataset"),
        (9.64, 21.19, "Mínimos do dataset")
    ]
    
    print(f"{'Cenário':<35} {'Irrad':>7} {'Temp':>6} {'Fator':>7} {'%':>7}")
    print("-" * 65)
    
    for irradiacao, temperatura, descricao in cenarios:
        desempenho = prever_desempenho(modelo, scaler, irradiacao, temperatura, mostrar_detalhes=False)
        
        if desempenho is not None:
            print(f"{descricao:<35} {irradiacao:>7.1f} {temperatura:>6.1f} {desempenho:>7.3f} {desempenho*100:>7.1f}")

def simular_dia_completo():
    """Simula desempenho ao longo de um dia"""
    print("\n📅 SIMULAÇÃO DE UM DIA COMPLETO")
    print("="*40)
    
    modelo, scaler, target_name = carregar_modelo_solar()
    if modelo is None:
        return
    
    try:
        temp_base = float(input("🌡️  Temperatura base do dia (°C): "))
        irrad_max = float(input("🌞 Irradiação máxima do dia (kWh/m²): "))
        
        print(f"\n⏰ SIMULAÇÃO HORA A HORA:")
        print(f"{'Hora':>5} {'Irrad':>8} {'Temp':>6} {'Fator':>7} {'%':>7}")
        print("-" * 40)
        
        desempenhos = []
        
        for hora in range(6, 19):  # 6h às 18h
            # Simular irradiação (curva senoidal)
            progresso = (hora - 6) / 12  # 0 a 1
            irrad = irrad_max * np.sin(np.pi * progresso)
            
            # Simular temperatura (varia ao longo do dia)
            temp_variacao = 4 * np.sin(2 * np.pi * progresso)
            temp = temp_base + temp_variacao
            
            # Predizer desempenho
            desempenho = prever_desempenho(modelo, scaler, irrad, temp, mostrar_detalhes=False)
            
            if desempenho is not None:
                desempenhos.append(desempenho)
                print(f"{hora:>5}h {irrad:>8.1f} {temp:>6.1f} {desempenho:>7.3f} {desempenho*100:>7.1f}")
        
        # Estatísticas do dia
        if desempenhos:
            desempenho_medio = np.mean(desempenhos)
            desempenho_max = max(desempenhos)
            desempenho_min = min(desempenhos)
            
            print(f"\n📊 RESUMO DO DIA:")
            print(f"   Fator médio: {desempenho_medio:.3f} ({desempenho_medio*100:.1f}%)")
            print(f"   Pico: {desempenho_max:.3f} ({desempenho_max*100:.1f}%)")
            print(f"   Menor: {desempenho_min:.3f} ({desempenho_min*100:.1f}%)")
            
            # Estimar geração do dia
            try:
                potencia = float(input(f"\n🔌 Potência instalada (kW): "))
                geracao_total = sum(calcular_geracao_estimada(d, potencia) for d in desempenhos if d is not None) / len(desempenhos)
                print(f"   Geração média estimada: {geracao_total:.1f} kWh/dia")
                print(f"   Geração mensal: {geracao_total * 30:.0f} kWh")
            except:
                pass
            
    except ValueError:
        print("❌ Digite valores numéricos válidos")

def menu_principal():
    """Menu principal do testador"""
    while True:
        print("\n" + "="*55)
        print("🌞 TESTADOR DE DESEMPENHO SOLAR (RNA CORRIGIDA)")
        print("="*55)
        print("📊 NOTA: Usando modelo neural treinado corrigido")
        print("   • Fator 1.0 = 100% de eficiência esperada")
        print("   • Range esperado: 0.6 a 1.4 (60% a 140%)")
        print("-" * 55)
        print("1. Teste interativo (inserir dados)")
        print("2. Teste de cenários predefinidos")
        print("3. Simulação de dia completo")
        print("4. Sair")
        print("-" * 55)
        
        escolha = input("Escolha uma opção (1-4): ").strip()
        
        if escolha == "1":
            teste_interativo()
        elif escolha == "2":
            teste_cenarios()
        elif escolha == "3":
            simular_dia_completo()
        elif escolha == "4":
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida. Digite 1, 2, 3 ou 4.")

if __name__ == "__main__":
    print("🔍 INICIALIZANDO TESTADOR SOLAR...")
    print("   Carregando modelo neural corrigido...")
    
    menu_principal()
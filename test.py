import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

def carregar_modelo():
    """Carrega o modelo treinado SEM normalização"""
    try:
        checkpoint = torch.load('modelo_sem_normalizacao.pth', map_location='cpu', weights_only=False)
        
        modelo = SolarRNA()
        modelo.load_state_dict(checkpoint['model'])
        modelo.eval()
        
        feature_names = checkpoint['feature_names']
        target_name = checkpoint['target_name']
        input_ranges = checkpoint['input_ranges']
        
        print("✅ Modelo carregado com sucesso!")
        print(f"   Entrada: {feature_names[0]} + {feature_names[1]}")
        print(f"   Saída: {target_name}")
        print(f"   Range Irradiação: {input_ranges['irradiacao_min']:.1f} - {input_ranges['irradiacao_max']:.1f}")
        print(f"   Range Temperatura: {input_ranges['temperatura_min']:.1f} - {input_ranges['temperatura_max']:.1f}")
        
        return modelo, input_ranges
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None, None

def prever_geracao(modelo, irradiacao, temperatura, mostrar_detalhes=True):
    """
    Faz predição de geração solar (DIRETO - sem normalização)
    
    Parâmetros:
    - irradiacao: Irradiação solar em kWh/m²
    - temperatura: Temperatura em °C
    
    Retorna:
    - Geração prevista em kWh
    """
    try:
        # Usar valores DIRETOS (sem normalização)
        entrada = torch.FloatTensor([[irradiacao, temperatura]])
        
        # Predição direta
        with torch.no_grad():
            resultado = modelo(entrada)
            geracao_prevista = resultado.item()
        
        if mostrar_detalhes:
            print(f"\n⚡ PREDIÇÃO:")
            print(f"   Irradiação: {irradiacao:.2f} kWh/m²")
            print(f"   Temperatura: {temperatura:.1f}°C")
            print(f"   Geração prevista: {geracao_prevista:.2f} kWh")
        
        return geracao_prevista
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return None

def teste_interativo():
    """Interface para testar o modelo com dados que você digita"""
    print("🌞 TESTADOR DO MODELO SOLAR (SEM NORMALIZAÇÃO)")
    print("="*55)
    
    modelo, ranges = carregar_modelo()
    if modelo is None:
        return
    
    print("\n📝 VALORES BASEADOS NOS SEUS DADOS:")
    print("   • Irradiação: 0-30+ kWh/m² (baseado na tabela real)")
    print("   • Temperatura: 21-34°C (baseado na tabela real)")
    print("   • Digite 'sair' para terminar")
    
    while True:
        print("\n" + "-"*50)
        
        # Entrada da irradiação
        try:
            irrad_input = input("🌞 Irradiação solar (kWh/m²): ").strip()
            if irrad_input.lower() == 'sair':
                break
            irradiacao = float(irrad_input)
            
            if irradiacao < 0:
                print("⚠️  Irradiação não pode ser negativa")
                continue
            elif irradiacao > 35:
                print("⚠️  Irradiação muito alta (acima de 35)")
            
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
        geracao = prever_geracao(modelo, irradiacao, temperatura)
        
        if geracao is not None:
            # Contexto baseado nos valores reais da tabela (22-51 kWh)
            if geracao > 48:
                status = "🔥 Excelente geração! (acima de 48 kWh)"
            elif geracao > 40:
                status = "☀️ Muito boa geração (40-48 kWh)"
            elif geracao > 32:
                status = "🌤️ Boa geração (32-40 kWh)"
            elif geracao > 25:
                status = "☁️ Geração moderada (25-32 kWh)"
            else:
                status = "🌧️ Baixa geração (abaixo de 25 kWh)"
            
            print(f"   {status}")

def teste_cenarios():
    """Testa cenários baseados nos dados reais"""
    print("\n🧪 TESTE DE CENÁRIOS (BASEADOS NOS DADOS REAIS)")
    print("="*55)
    
    modelo, ranges = carregar_modelo()
    if modelo is None:
        return
    
    # Cenários baseados nos valores reais da tabela
    cenarios = [
        (25.0, 25, "Irradiação alta, temperatura amena"),
        (20.0, 30, "Irradiação boa, temperatura alta"),
        (15.0, 23, "Irradiação média, temperatura baixa"),
        (10.0, 28, "Irradiação baixa, temperatura alta"),
        (5.0, 22, "Irradiação muito baixa"),
        (30.0, 32, "Irradiação máxima, temperatura alta"),
        (18.0, 26, "Condições típicas"),
        (1.0, 25, "Quase sem irradiação"),
        (0.0, 24, "Sem irradiação (noite)")
    ]
    
    print(f"{'Cenário':<32} {'Irrad':>6} {'Temp':>5} {'Geração':>8}")
    print("-" * 55)
    
    for irradiacao, temperatura, descricao in cenarios:
        geracao = prever_geracao(modelo, irradiacao, temperatura, mostrar_detalhes=False)
        
        if geracao is not None:
            print(f"{descricao:<32} {irradiacao:>6.1f} {temperatura:>5.0f} {geracao:>8.1f}")

def calcular_estimativa_mensal():
    """Calcula estimativa mensal baseada nos dados reais"""
    print("\n📊 ESTIMATIVA DE GERAÇÃO MENSAL")
    print("="*40)
    
    modelo, ranges = carregar_modelo()
    if modelo is None:
        return
    
    try:
        print("Digite as condições médias do mês:")
        print("(Baseado nos seus dados: Irradiação 0-30+, Temperatura 21-34°C)")
        
        irrad_media = float(input("Irradiação média diária (kWh/m²): "))
        temp_media = float(input("Temperatura média (°C): "))
        dias_mes = int(input("Dias no mês (default 30): ") or "30")
        
        # Calcular geração diária média
        geracao_diaria = prever_geracao(modelo, irrad_media, temp_media, mostrar_detalhes=False)
        
        if geracao_diaria is not None:
            geracao_mensal = geracao_diaria * dias_mes
            
            print(f"\n📈 ESTIMATIVA MENSAL:")
            print(f"   Geração diária média: {geracao_diaria:.1f} kWh")
            print(f"   Geração mensal: {geracao_mensal:.0f} kWh")
            print(f"   Receita estimada (R$0,65/kWh): R$ {geracao_mensal * 0.65:.0f}")
            
            # Contexto baseado nos dados reais
            if geracao_mensal > 1400:
                print("   🔥 Excelente mês!")
            elif geracao_mensal > 1000:
                print("   ☀️ Bom mês de geração")
            elif geracao_mensal > 700:
                print("   🌤️ Mês razoável")
            else:
                print("   ☁️ Mês com baixa geração")
            
    except ValueError:
        print("❌ Digite valores numéricos válidos")

def teste_extremos():
    """Testa valores extremos dos dados"""
    print("\n🔬 TESTE DE VALORES EXTREMOS")
    print("="*35)
    
    modelo, ranges = carregar_modelo()
    if modelo is None:
        return
    
    # Baseado nos ranges reais dos dados
    extremos = [
        (ranges['irradiacao_max'], ranges['temperatura_max'], "Máximo absoluto"),
        (ranges['irradiacao_max'], ranges['temperatura_min'], "Irrad máx + Temp mín"),
        (ranges['irradiacao_min'], ranges['temperatura_max'], "Irrad mín + Temp máx"),
        (ranges['irradiacao_min'], ranges['temperatura_min'], "Mínimo absoluto"),
        (15.0, 26.0, "Valores médios típicos"),
    ]
    
    print(f"{'Teste':<20} {'Irrad':>8} {'Temp':>6} {'Geração':>8}")
    print("-" * 45)
    
    for irrad, temp, desc in extremos:
        geracao = prever_geracao(modelo, irrad, temp, mostrar_detalhes=False)
        if geracao is not None:
            print(f"{desc:<20} {irrad:>8.1f} {temp:>6.1f} {geracao:>8.1f}")

def menu_principal():
    """Menu principal do testador"""
    while True:
        print("\n" + "="*60)
        print("🌞 TESTADOR DO MODELO SOLAR (DADOS REAIS)")
        print("="*60)
        print("1. Teste interativo (inserir seus próprios valores)")
        print("2. Teste de cenários típicos")
        print("3. Estimativa de geração mensal")
        print("4. Teste de valores extremos")
        print("5. Sair")
        print("-" * 60)
        
        escolha = input("Escolha uma opção (1-5): ").strip()
        
        if escolha == "1":
            teste_interativo()
        elif escolha == "2":
            teste_cenarios()
        elif escolha == "3":
            calcular_estimativa_mensal()
        elif escolha == "4":
            teste_extremos()
        elif escolha == "5":
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida. Digite 1, 2, 3, 4 ou 5.")

if __name__ == "__main__":
    menu_principal()
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SolarRNA(nn.Module):
    """Modelo adaptado para dados não normalizados"""
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
        
        # Inicialização adaptada para dados não normalizados
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def train_model_sem_normalizacao(excel_path):
    """Treinamento SEM normalização das entradas"""
    print("🎯 RNA Solar - DADOS ORIGINAIS (Sem Normalização)")
    print("="*55)
    
    # Carregar dados do Excel
    df = pd.read_excel(excel_path)
    
    print(f"Colunas encontradas: {list(df.columns)}")
    print(f"Dataset: {len(df)} registros")
    
    # Detectar automaticamente as colunas corretas
    irrad_col = None
    temp_col = None
    geracao_col = None
    
    # Possíveis nomes para cada tipo de coluna
    irrad_keywords = ['irrad', 'solar', 'sun', 'radiação', 'radiacao', 'watt', 'w/m2', 'kwh/m2']
    temp_keywords = ['temp', 'temperatura', 'celsius', 'grau', '°c', 'ambiente']
    geracao_keywords = ['gera', 'produc', 'energia', 'output', 'kwh', 'potencia', 'potência']
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        if not irrad_col and any(keyword in col_lower for keyword in irrad_keywords):
            irrad_col = col
        elif not temp_col and any(keyword in col_lower for keyword in temp_keywords):
            temp_col = col
        elif not geracao_col and any(keyword in col_lower for keyword in geracao_keywords):
            geracao_col = col
    
    # Se não encontrou automaticamente, usar as primeiras 3 colunas
    if not all([irrad_col, temp_col, geracao_col]):
        cols = list(df.columns)
        if len(cols) >= 3:
            irrad_col = cols[0]
            temp_col = cols[1] 
            geracao_col = cols[2]
    
    print(f"🔍 Colunas identificadas:")
    print(f"   Irradiação: {irrad_col}")
    print(f"   Temperatura: {temp_col}")
    print(f"   Geração: {geracao_col}")
    
    if not all([irrad_col, temp_col, geracao_col]):
        print("❌ Erro: Não foi possível identificar todas as colunas necessárias")
        return None, None, None
    
    # Extrair dados ORIGINAIS (sem normalização)
    X = df[[irrad_col, temp_col]].values.astype(np.float32)
    y = df[geracao_col].values.reshape(-1, 1).astype(np.float32)
    
    # Verificar dados
    print(f"\n📊 DADOS ORIGINAIS (SEM NORMALIZAÇÃO):")
    print(f"Irradiação - Min: {X[:, 0].min():.2f}, Max: {X[:, 0].max():.2f}, Média: {X[:, 0].mean():.2f}")
    print(f"Temperatura - Min: {X[:, 1].min():.2f}, Max: {X[:, 1].max():.2f}, Média: {X[:, 1].mean():.2f}")
    print(f"Geração - Min: {y.min():.2f}, Max: {y.max():.2f}, Média: {y.mean():.2f}")
    
    # Remover valores inválidos (NaN, infinitos)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y).flatten()
    X = X[mask]
    y = y[mask]
    
    if len(X) < len(df):
        print(f"⚠️  Removidos {len(df) - len(X)} registros com valores inválidos")
    
    print(f"Dataset final: {len(X)} registros válidos")
    
    # Divisão dos dados (SEM normalização)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
    )
    
    print(f"Divisão: Treino={len(X_train)}, Val={len(X_val)}, Teste={len(X_test)}")
    
    # Converter para tensores (dados originais)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Modelo adaptado para dados não normalizados
    torch.manual_seed(42)
    model = SolarRNA().to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {params}")
    
    # Configuração para dados não normalizados
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    
    # Loss simples (MSE funciona bem para dados não normalizados)
    criterion = nn.MSELoss()
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Treinamento
    print(f"\n{'Época':>5} {'Train':>8} {'Val':>8} {'Test':>8} {'R²Test':>7}")
    print("-" * 42)
    
    best_val_loss = float('inf')
    patience = 40
    patience_counter = 0
    
    for epoch in range(200):
        # Treino
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping leve
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        scheduler.step()
        
        # Validação e teste
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            
            # Calcular R²
            val_pred = val_outputs.cpu().numpy()
            test_pred = test_outputs.cpu().numpy()
            
            test_r2 = r2_score(y_test, test_pred)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Salvar modelo (sem scalers pois não há normalização)
            torch.save({
                'model': model.state_dict(),
                'feature_names': [irrad_col, temp_col],
                'target_name': geracao_col,
                'input_ranges': {
                    'irradiacao_min': float(X[:, 0].min()),
                    'irradiacao_max': float(X[:, 0].max()),
                    'temperatura_min': float(X[:, 1].min()),
                    'temperatura_max': float(X[:, 1].max())
                }
            }, 'modelo_sem_normalizacao.pth')
        else:
            patience_counter += 1
        
        # Log
        if epoch % 20 == 0 or epoch < 10:
            print(f"{epoch+1:>5} {train_loss:>8.2f} {val_loss:>8.2f} {test_loss:>8.2f} {test_r2:>7.3f}")
        
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch+1}")
            break
    
    # Carregar melhor modelo
    checkpoint = torch.load('modelo_sem_normalizacao.pth', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    print("\n" + "="*50)
    print("RESULTADO FINAL - DADOS ORIGINAIS")
    print("="*50)
    
    # Avaliação final (dados já estão em escala original)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        test_outputs = model(X_test_tensor)
        
        val_pred = val_outputs.cpu().numpy()
        test_pred = test_outputs.cpu().numpy()
        
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"R² Validação: {val_r2:.4f}")
    print(f"R² Teste:     {test_r2:.4f}")
    print(f"MAE:          {test_mae:.2f} kWh")
    print(f"RMSE:         {test_rmse:.2f} kWh")
    
    # Análise do erro (dados já em kWh)
    print(f"\n📊 ANÁLISE DE ERRO:")
    print(f"Erro médio diário: {test_mae:.2f} kWh")
    print(f"Erro RMS diário:   {test_rmse:.2f} kWh")
    
    # Erro anual
    erro_anual = test_mae * 365
    print(f"Erro anual total:  {erro_anual:,.0f} kWh")
    
    # Performance
    geracao_media = y.mean()
    geracao_anual = geracao_media * 365
    
    print(f"\n🏠 ANÁLISE DE PERFORMANCE:")
    print(f"Geração média diária: {geracao_media:.2f} kWh")
    print(f"Geração anual estimada: {geracao_anual:,.0f} kWh")
    
    if geracao_anual > 0:
        perc_erro = (erro_anual / geracao_anual) * 100
        print(f"Erro como % da geração: {perc_erro:.1f}%")
        
        if perc_erro <= 10:
            status = "🟢 EXCELENTE"
        elif perc_erro <= 20:
            status = "🟡 MUITO BOA"
        elif perc_erro <= 30:
            status = "🟠 BOA"
        else:
            status = "🔴 ACEITÁVEL"
        
        print(f"{status} - Performance do modelo")
        
        # Custo do erro
        custo_erro = erro_anual * 0.65
        print(f"Custo do erro: R$ {custo_erro:,.0f}/ano")
    
    print(f"\n💾 Modelo salvo como 'modelo_sem_normalizacao.pth'")
    print(f"   (SEM scalers - usa dados originais diretamente)")
    
    # Comparação
    baseline_r2 = 0.516
    if test_r2 > baseline_r2:
        melhoria = ((test_r2 - baseline_r2) / baseline_r2) * 100
        print(f"✅ Melhoria de {melhoria:.1f}% vs baseline")
    else:
        piora = ((baseline_r2 - test_r2) / baseline_r2) * 100
        print(f"❌ Piora de {piora:.1f}% vs baseline")
    
    # Verificação de predições
    print(f"\n🔍 VERIFICAÇÃO (primeiras 5 predições do teste):")
    print(f"{'Real':>8} {'Predito':>8} {'Erro':>8} {'Erro%':>8}")
    print("-" * 35)
    for i in range(min(5, len(y_test))):
        real = y_test[i, 0]
        pred = test_pred[i, 0]
        erro = abs(real - pred)
        erro_perc = (erro / real) * 100 if real != 0 else 0
        print(f"{real:>8.1f} {pred:>8.1f} {erro:>8.1f} {erro_perc:>7.1f}%")
    
    # Teste de sanidade (valores esperados)
    print(f"\n🧪 TESTE DE SANIDADE:")
    test_cases = [
        [6.0, 25],  # Dia ensolarado normal
        [3.0, 30],  # Dia parcialmente nublado e quente
        [1.0, 20],  # Dia nublado e fresco
        [0.0, 25]   # Noite
    ]
    
    print(f"{'Irrad':>6} {'Temp':>6} {'Predição':>10}")
    print("-" * 25)
    
    model.eval()
    with torch.no_grad():
        for irrad, temp in test_cases:
            test_input = torch.FloatTensor([[irrad, temp]]).to(device)
            pred = model(test_input).cpu().numpy()[0, 0]
            print(f"{irrad:>6.1f} {temp:>6.0f} {pred:>10.1f}")
    
    return model, test_r2, erro_anual

if __name__ == "__main__":
    try:
        model, r2, erro = train_model_sem_normalizacao('Dados Willian.xlsx')
        print(f"\n🏆 RESUMO FINAL:")
        print(f"   R² = {r2:.3f}")
        print(f"   Erro anual = {erro:,.0f} kWh")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
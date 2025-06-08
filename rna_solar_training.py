import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleSolarRNA(nn.Module):
    """Modelo simplificado sem BatchNorm para evitar problemas"""
    def __init__(self, input_size=13):
        super(SimpleSolarRNA, self).__init__()
        
        # Arquitetura mais simples e estável
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
            # SEM Sigmoid - deixar a saída livre
        )
        
        # Inicialização cuidadosa
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def detectar_colunas_desempenho(df):
    """Detecta colunas no arquivo Dados Willian"""
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    mapeamento = {}
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Irradiação
        if any(keyword in col_lower for keyword in ['irrad', 'solar', 'kwh/m2', 'kwh_m2']):
            mapeamento['irradiacao'] = col
        
        # Temperatura média
        elif 'temperatura' in col_lower and 'media' in col_lower:
            mapeamento['temp_media'] = col
        
        # Desempenho (target)
        elif any(keyword in col_lower for keyword in ['desempenho', 'performance', 'porcentagem', 'percent', '%']):
            mapeamento['desempenho'] = col
    
    return mapeamento

def criar_features_engineered(irrad, temp):
    """Cria features engineered de forma mais conservadora"""
    
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

def train_desempenho_model_fixed(excel_path):
    """Treinamento corrigido para prever desempenho"""
    print("🎯 RNA SOLAR - TREINAMENTO CORRIGIDO")
    print("="*50)
    
    # Carregar dados
    df = pd.read_excel(excel_path)
    print(f"Dataset: {len(df)} registros")
    
    # Detectar colunas
    mapeamento = detectar_colunas_desempenho(df)
    print(f"🔍 Colunas detectadas: {mapeamento}")
    
    if 'desempenho' not in mapeamento:
        print("❌ Coluna de desempenho não encontrada!")
        return None, None, None
    
    # Preparar features
    print("🔧 Criando features...")
    
    irrad_values = df[mapeamento['irradiacao']].values
    temp_values = df[mapeamento['temp_media']].values
    
    # Criar features engineered para todo o dataset
    X_list = []
    for i in range(len(df)):
        features = criar_features_engineered(irrad_values[i], temp_values[i])
        X_list.append(features)
    
    X = np.vstack(X_list)
    
    # Target: Desempenho (manter como está no arquivo)
    y = df[mapeamento['desempenho']].values.reshape(-1, 1).astype(np.float32)
    
    print(f"📊 DADOS ANALISADOS:")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Target stats: Min={y.min():.3f}, Max={y.max():.3f}, Mean={y.mean():.3f}")
    
    # CRUCIAL: Verificar se os targets estão na escala correta
    if y.max() <= 2.0:  # Se parecem estar em escala 0-2
        print("✅ Target parece estar em escala decimal (0-2)")
        print("   Interpretação: 1.0 = 100% de eficiência")
    else:
        print("✅ Target parece estar em escala percentual")
        y = y / 100  # Converter para escala 0-1
        print("   Convertido para escala decimal")
    
    # Normalizar features
    print("📐 Normalizando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   Features após normalização: Mean≈0, Std≈1")
    
    # Remover valores inválidos
    mask = np.isfinite(X_scaled).all(axis=1) & np.isfinite(y).flatten()
    X_scaled = X_scaled[mask]
    y = y[mask]
    
    print(f"Dataset final: {len(X_scaled)} registros válidos")
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Divisão: Treino={len(X_train)}, Val={len(X_val)}, Teste={len(X_test)}")
    
    # Converter para tensores
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Modelo simplificado
    torch.manual_seed(42)
    model = SimpleSolarRNA(input_size=X.shape[1]).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modelo: {params:,} parâmetros")
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Otimizador e loss mais conservadores
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=20
    )
    criterion = nn.MSELoss()
    
    # Treinamento
    print(f"\n🚀 TREINAMENTO:")
    print(f"{'Época':>5} {'Train Loss':>10} {'Val Loss':>10} {'Val R²':>8} {'Test MAE':>8}")
    print("-" * 50)
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(300):
        # Treino
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validação
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            test_outputs = model(X_test_tensor)
            
            # Métricas
            val_pred = val_outputs.cpu().numpy()
            test_pred = test_outputs.cpu().numpy()
            
            val_r2 = r2_score(y_val, val_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Salvar melhor modelo
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
                'target_name': mapeamento['desempenho'],
                'input_size': X.shape[1],
                'val_loss': val_loss,
                'test_mae': test_mae,
                'model_class': 'SimpleSolarRNA'
            }, 'modelo_solar_corrigido.pth')
        else:
            patience_counter += 1
        
        # Log
        if epoch % 25 == 0 or epoch < 10:
            print(f"{epoch+1:>5} {train_loss:>10.6f} {val_loss:>10.6f} {val_r2:>8.4f} {test_mae:>8.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping na época {epoch+1}")
            break
    
    # Carregar melhor modelo
    checkpoint = torch.load('modelo_solar_corrigido.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*60)
    print("📊 AVALIAÇÃO FINAL")
    print("="*60)
    
    # Avaliação final
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = test_outputs.cpu().numpy()
        
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"🎯 MÉTRICAS FINAIS:")
    print(f"   R² Test: {test_r2:.4f}")
    print(f"   MAE Test: {test_mae:.4f}")
    print(f"   RMSE Test: {test_rmse:.4f}")
    
    # Mostrar alguns exemplos de predição
    print(f"\n💡 EXEMPLOS DE PREDIÇÃO:")
    print(f"{'Real':>8} {'Previsto':>10} {'Erro':>8}")
    print("-" * 28)
    
    for i in range(min(10, len(y_test))):
        real = y_test[i, 0]
        prev = test_pred[i, 0]
        erro = abs(real - prev)
        print(f"{real:>8.3f} {prev:>10.3f} {erro:>8.3f}")
    
    # Avaliação da qualidade
    if test_mae < 0.05:
        status = "🟢 EXCELENTE"
    elif test_mae < 0.1:
        status = "🟡 MUITO BOM"
    elif test_mae < 0.2:
        status = "🟠 RAZOÁVEL"
    else:
        status = "🔴 PRECISA MELHORAR"
    
    print(f"\n📈 QUALIDADE DO MODELO: {status}")
    print(f"   MAE de {test_mae:.4f} significa erro médio de ±{test_mae*100:.1f} pontos percentuais")
    
    return model, test_r2, test_mae

if __name__ == "__main__":
    try:
        result = train_desempenho_model_fixed('Dados Willian.xlsx')
        
        if result[0] is not None:
            model, r2, mae = result
            print(f"\n🏆 RESUMO:")
            print(f"   R² = {r2:.4f}")
            print(f"   MAE = {mae:.4f}")
            print(f"   Modelo salvo como 'modelo_solar_corrigido.pth'")
        else:
            print(f"\n❌ Treinamento falhou")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
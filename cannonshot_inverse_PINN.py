# %% [markdown]
# # Operador Neural Paramétrico para Balística Inversa
#
# ---
#
# **O Santo Graal da Balística!**
#
# Este operador neural recebe:
# - `x_impact`: distância do alvo (m)
# - `α ∈ [0,1]`: parâmetro que varre todas as soluções possíveis
#
# E retorna diretamente:
# - `V₀(α)`: velocidade inicial
# - `θ₀(α)`: ângulo de lançamento  
# - `T(α)`: tempo de voo
#
# Sem precisar de NENHUM solver iterativo!
#
# A rede aprende a função inversa satisfazendo:
# - x(T) = x_impact (condição de contorno no alvo)
# - y(T) = 0 (impacto no solo)
# - vy(T) < 0 (chegando de cima)
#

# %% Configuração
# ======================================================================
# HIPERPARÂMETROS
# ======================================================================

# Range do alvo
X_IMPACT_MIN = 100.0    # metros
X_IMPACT_MAX = 2000.0   # metros

# Range de V0
V0_MIN = 10.0           # m/s
V0_MAX = 250.0          # m/s

# Range de θ0
THETA_MIN = 15.0        # graus
THETA_MAX = 75.0        # graus

# Física
g = 9.81
Cd = 0.47
m = 1.0
rho = 1.225
densidade_chumbo = 11340
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3) if 'np' in dir() else 0.027
A = 3.14159 * (diametro / 2) ** 2
k_drag = (rho * Cd * A) / (2 * m)

# Treinamento
N_EPOCHS = 10000
BATCH_SIZE = 256
LR = 1e-3
N_INTEGRATION_STEPS = 200  # Passos de integração

# Pesos da loss
LAMBDA_X = 1.0      # Condição x(T) = x_target
LAMBDA_Y = 10.0     # Condição y(T) = 0
LAMBDA_VY = 1.0     # Condição vy(T) < 0
LAMBDA_THETA = 0.1  # Regularização θ ~ α

# ======================================================================

# %% Importações
import time
import inspect
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Garante que o diretório de trabalho seja a raiz do projeto
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# Recalcula k_drag com numpy disponível
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)
A = np.pi * (diametro / 2) ** 2
k_drag = (rho * Cd * A) / (2 * m)

print(f"k_drag = {k_drag:.6f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% Funções auxiliares para normalização
def normalize_x(x):
    """Normaliza x_impact para [-1, 1]."""
    return 2 * (x - X_IMPACT_MIN) / (X_IMPACT_MAX - X_IMPACT_MIN) - 1

def denormalize_x(x_norm):
    """Desnormaliza x_impact de [-1, 1]."""
    return (x_norm + 1) / 2 * (X_IMPACT_MAX - X_IMPACT_MIN) + X_IMPACT_MIN

def normalize_v0(v0):
    """Normaliza V0 para [0, 1]."""
    return (v0 - V0_MIN) / (V0_MAX - V0_MIN)

def denormalize_v0(v0_norm):
    """Desnormaliza V0 de [0, 1]."""
    return v0_norm * (V0_MAX - V0_MIN) + V0_MIN

def normalize_theta(theta_deg):
    """Normaliza θ para [0, 1]."""
    return (theta_deg - THETA_MIN) / (THETA_MAX - THETA_MIN)

def denormalize_theta(theta_norm):
    """Desnormaliza θ de [0, 1] para graus."""
    return theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN

# %% Integrador diferenciável (RK4 em PyTorch)
def integrate_trajectory(v0, theta_rad, T, n_steps=N_INTEGRATION_STEPS):
    """
    Integra a trajetória balística com arrasto usando RK4.
    
    Totalmente diferenciável para backpropagation!
    
    Args:
        v0: Velocidade inicial [batch]
        theta_rad: Ângulo em radianos [batch]
        T: Tempo total [batch]
        n_steps: Número de passos de integração
    
    Returns:
        x_final, y_final, vx_final, vy_final: Estados finais [batch]
    """
    batch_size = v0.shape[0]
    
    # Condições iniciais
    x = torch.zeros(batch_size, device=v0.device)
    y = torch.zeros(batch_size, device=v0.device)
    vx = v0 * torch.cos(theta_rad)
    vy = v0 * torch.sin(theta_rad)
    
    # Passo de tempo (diferente para cada amostra)
    dt = T / n_steps
    
    def derivatives(x, y, vx, vy):
        """Calcula derivadas com arrasto."""
        v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)  # +eps para estabilidade
        ax = -k_drag * v_mag * vx
        ay = -g - k_drag * v_mag * vy
        return vx, vy, ax, ay
    
    # Integração RK4
    for _ in range(n_steps):
        # k1
        dx1, dy1, dvx1, dvy1 = derivatives(x, y, vx, vy)
        
        # k2
        dx2, dy2, dvx2, dvy2 = derivatives(
            x + 0.5 * dt * dx1,
            y + 0.5 * dt * dy1,
            vx + 0.5 * dt * dvx1,
            vy + 0.5 * dt * dvy1
        )
        
        # k3
        dx3, dy3, dvx3, dvy3 = derivatives(
            x + 0.5 * dt * dx2,
            y + 0.5 * dt * dy2,
            vx + 0.5 * dt * dvx2,
            vy + 0.5 * dt * dvy2
        )
        
        # k4
        dx4, dy4, dvx4, dvy4 = derivatives(
            x + dt * dx3,
            y + dt * dy3,
            vx + dt * dvx3,
            vy + dt * dvy3
        )
        
        # Atualiza
        x = x + (dt / 6) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y = y + (dt / 6) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        vx = vx + (dt / 6) * (dvx1 + 2*dvx2 + 2*dvx3 + dvx4)
        vy = vy + (dt / 6) * (dvy1 + 2*dvy2 + 2*dvy3 + dvy4)
    
    return x, y, vx, vy

# %% Definição da Rede Neural
class InverseBallisticPINN(nn.Module):
    """
    Operador Neural Paramétrico para Balística Inversa.
    
    Input: [x_impact_norm, α] onde α ∈ [0,1] parametriza as soluções
    Output: [V₀, θ₀, T] que satisfazem x(T) = x_impact, y(T) = 0
    """
    
    def __init__(self, hidden_layers=[128, 128, 128, 128]):
        super().__init__()
        
        # Arquitetura MLP
        layers = []
        input_dim = 2  # [x_impact_norm, α]
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 3))  # [V0_raw, theta_raw, T_raw]
        
        self.net = nn.Sequential(*layers)
        
        # Escalas para outputs
        self.V0_min = V0_MIN
        self.V0_max = V0_MAX
        self.theta_min_rad = np.radians(THETA_MIN)
        self.theta_max_rad = np.radians(THETA_MAX)
        
        # T_max estimado (baseado em vácuo com V0_max e θ=90°)
        self.T_max = 2 * V0_MAX / g * 1.5  # margem de segurança
    
    def forward(self, x_impact_norm, alpha):
        """
        Forward pass.
        
        Args:
            x_impact_norm: x_impact normalizado [-1, 1] [batch]
            alpha: parâmetro α ∈ [0, 1] [batch]
        
        Returns:
            v0: velocidade inicial [batch]
            theta_rad: ângulo em radianos [batch]
            T: tempo de voo [batch]
        """
        # Concatena inputs
        inputs = torch.stack([x_impact_norm, alpha], dim=1)
        
        # Forward
        raw = self.net(inputs)
        
        # Ativações para garantir ranges físicos
        # V0: sigmoid → [V0_min, V0_max]
        v0 = torch.sigmoid(raw[:, 0]) * (self.V0_max - self.V0_min) + self.V0_min
        
        # θ: sigmoid → [theta_min, theta_max], mas influenciado por α
        # A ideia é que α=0 → θ≈θ_min, α=1 → θ≈θ_max
        theta_base = alpha * (self.theta_max_rad - self.theta_min_rad) + self.theta_min_rad
        theta_correction = torch.tanh(raw[:, 1]) * 0.2  # correção de até ±0.2 rad (~11°)
        theta_rad = theta_base + theta_correction
        theta_rad = torch.clamp(theta_rad, self.theta_min_rad, self.theta_max_rad)
        
        # T: softplus para garantir T > 0, escalado
        T = torch.nn.functional.softplus(raw[:, 2]) * self.T_max / 5 + 1.0
        
        return v0, theta_rad, T

# %% Função de Loss
def physics_loss(model, x_impact_norm, alpha, x_impact_real):
    """
    Calcula a loss baseada nas condições de contorno físicas.
    
    Args:
        model: InverseBallisticPINN
        x_impact_norm: x_impact normalizado [batch]
        alpha: parâmetro α [batch]
        x_impact_real: x_impact em metros [batch]
    
    Returns:
        loss_total, loss_x, loss_y, loss_vy, loss_theta
    """
    # Forward pass
    v0, theta_rad, T = model(x_impact_norm, alpha)
    
    # Integra trajetória
    x_final, y_final, vx_final, vy_final = integrate_trajectory(v0, theta_rad, T)
    
    # Loss de condição de contorno em x
    loss_x = torch.mean((x_final - x_impact_real)**2)
    
    # Loss de condição de contorno em y (deve ser zero)
    loss_y = torch.mean(y_final**2)
    
    # Loss de velocidade vertical (deve ser negativa - descendo)
    # Penaliza se vy > 0
    loss_vy = torch.mean(torch.relu(vy_final)**2)
    
    # Regularização: θ deve seguir α aproximadamente
    theta_expected = alpha * (model.theta_max_rad - model.theta_min_rad) + model.theta_min_rad
    loss_theta = torch.mean((theta_rad - theta_expected)**2)
    
    # Loss total
    loss_total = (LAMBDA_X * loss_x + 
                  LAMBDA_Y * loss_y + 
                  LAMBDA_VY * loss_vy + 
                  LAMBDA_THETA * loss_theta)
    
    return loss_total, loss_x, loss_y, loss_vy, loss_theta

# %% Gerador de dados de treinamento
def generate_batch(batch_size, device):
    """
    Gera um batch de dados de treinamento.
    
    Amostra x_impact e α uniformemente.
    """
    # x_impact uniforme em [X_MIN, X_MAX]
    x_impact = torch.rand(batch_size, device=device) * (X_IMPACT_MAX - X_IMPACT_MIN) + X_IMPACT_MIN
    x_impact_norm = normalize_x(x_impact)
    
    # α uniforme em [0, 1]
    alpha = torch.rand(batch_size, device=device)
    
    return x_impact_norm, alpha, x_impact

# %% Instancia modelo
model = InverseBallisticPINN(hidden_layers=[128, 128, 128, 128]).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)

print(f"\nModelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")

# %% Treinamento
print("\n" + "="*70)
print("TREINAMENTO")
print("="*70)

history = {
    'loss': [],
    'loss_x': [],
    'loss_y': [],
    'loss_vy': [],
    'loss_theta': []
}

best_loss = float('inf')
best_state = None

start_time = time.time()

pbar = tqdm(range(N_EPOCHS), desc="Treinando")
for epoch in pbar:
    model.train()
    
    # Gera batch
    x_impact_norm, alpha, x_impact_real = generate_batch(BATCH_SIZE, device)
    
    # Forward + loss
    optimizer.zero_grad()
    loss, loss_x, loss_y, loss_vy, loss_theta = physics_loss(
        model, x_impact_norm, alpha, x_impact_real
    )
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Scheduler
    scheduler.step(loss)
    
    # Logging
    history['loss'].append(loss.item())
    history['loss_x'].append(loss_x.item())
    history['loss_y'].append(loss_y.item())
    history['loss_vy'].append(loss_vy.item())
    history['loss_theta'].append(loss_theta.item())
    
    # Best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Progress
    if epoch % 100 == 0:
        pbar.set_postfix({
            'loss': f'{loss.item():.2e}',
            'x': f'{loss_x.item():.2e}',
            'y': f'{loss_y.item():.2e}'
        })

elapsed = time.time() - start_time
print(f"\nTreinamento concluído em {elapsed/60:.1f} minutos")
print(f"Melhor loss: {best_loss:.6f}")

# Carrega melhor modelo
model.load_state_dict(best_state)
model.eval()

# %% Visualização do treinamento
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax1 = axes[0, 0]
ax1.semilogy(history['loss'], 'b-', alpha=0.7)
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss Total')
ax1.set_title('Convergência do Treinamento')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.semilogy(history['loss_x'], label='Loss x', alpha=0.7)
ax2.semilogy(history['loss_y'], label='Loss y', alpha=0.7)
ax2.semilogy(history['loss_vy'], label='Loss vy', alpha=0.7)
ax2.set_xlabel('Época')
ax2.set_ylabel('Loss')
ax2.set_title('Componentes da Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# %% Validação: testa para diferentes x_impact e α
print("\n" + "="*70)
print("VALIDAÇÃO")
print("="*70)

x_test_values = [200, 500, 800, 1000, 1500]
alpha_test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"\n{'x_target':>10} {'α':>6} │ {'V0':>8} {'θ°':>8} {'T':>8} │ {'x_sim':>10} {'y_sim':>10} {'vy_sim':>10} │ {'Erro x%':>8}")
print("-" * 100)

with torch.no_grad():
    for x_target in x_test_values:
        for alpha in alpha_test_values:
            x_norm = torch.tensor([normalize_x(x_target)], device=device)
            a = torch.tensor([alpha], device=device)
            
            v0, theta_rad, T = model(x_norm, a)
            x_f, y_f, vx_f, vy_f = integrate_trajectory(v0, theta_rad, T)
            
            v0_val = v0.item()
            theta_deg = np.degrees(theta_rad.item())
            T_val = T.item()
            x_sim = x_f.item()
            y_sim = y_f.item()
            vy_sim = vy_f.item()
            
            erro_x = abs(x_sim - x_target) / x_target * 100
            
            status = "✓" if erro_x < 5 and abs(y_sim) < 10 and vy_sim < 0 else "✗"
            
            print(f"{x_target:10.0f} {alpha:6.2f} │ {v0_val:8.1f} {theta_deg:8.2f} {T_val:8.2f} │ "
                  f"{x_sim:10.1f} {y_sim:10.2f} {vy_sim:10.2f} │ {erro_x:8.2f} {status}")
        print()

# %% Visualização das soluções
ax3 = axes[1, 0]
ax4 = axes[1, 1]

x_target = 800  # metros
alphas = np.linspace(0, 1, 20)
colors = plt.cm.plasma(alphas)

with torch.no_grad():
    for i, alpha in enumerate(alphas):
        x_norm = torch.tensor([normalize_x(x_target)], device=device)
        a = torch.tensor([alpha], device=device)
        
        v0, theta_rad, T = model(x_norm, a)
        
        # Simula trajetória completa para plotar
        v0_val = v0.item()
        theta_val = theta_rad.item()
        T_val = T.item()
        
        t_traj = np.linspace(0, T_val, 100)
        x_traj = np.zeros_like(t_traj)
        y_traj = np.zeros_like(t_traj)
        
        # Euler para plotar (não precisa ser diferenciável)
        x, y = 0.0, 0.0
        vx = v0_val * np.cos(theta_val)
        vy = v0_val * np.sin(theta_val)
        dt = T_val / 100
        
        for j in range(100):
            x_traj[j] = x
            y_traj[j] = y
            
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
            
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
        
        ax3.plot(x_traj, y_traj, '-', color=colors[i], alpha=0.7, linewidth=1.5)

ax3.scatter([x_target], [0], c='red', s=200, marker='o', zorder=10, label=f'Alvo: {x_target}m')
ax3.axhline(y=0, color='saddlebrown', linewidth=2, alpha=0.5)
ax3.set_xlabel('Distância (m)')
ax3.set_ylabel('Altura (m)')
ax3.set_title(f'Família de Trajetórias para x_target = {x_target}m\n(α varia de 0 a 1)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot V0 e θ vs α
alphas_dense = np.linspace(0, 1, 50)
v0_vals = []
theta_vals = []

with torch.no_grad():
    for alpha in alphas_dense:
        x_norm = torch.tensor([normalize_x(x_target)], device=device)
        a = torch.tensor([alpha], device=device)
        v0, theta_rad, T = model(x_norm, a)
        v0_vals.append(v0.item())
        theta_vals.append(np.degrees(theta_rad.item()))

ax4.plot(alphas_dense, v0_vals, 'b-', linewidth=2, label='V₀ (m/s)')
ax4_twin = ax4.twinx()
ax4_twin.plot(alphas_dense, theta_vals, 'r-', linewidth=2, label='θ₀ (°)')
ax4.set_xlabel('α')
ax4.set_ylabel('V₀ (m/s)', color='blue')
ax4_twin.set_ylabel('θ₀ (°)', color='red')
ax4.set_title(f'V₀ e θ₀ como função de α (x_target = {x_target}m)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Salva figura
images_dir = PROJECT_ROOT / "imagens" / "inverse_pinn"
images_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(images_dir / "treinamento_validacao.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# %% Salva checkpoint
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "inverse_pinn"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'config': {
        'X_IMPACT_MIN': X_IMPACT_MIN,
        'X_IMPACT_MAX': X_IMPACT_MAX,
        'V0_MIN': V0_MIN,
        'V0_MAX': V0_MAX,
        'THETA_MIN': THETA_MIN,
        'THETA_MAX': THETA_MAX,
        'k_drag': k_drag,
        'Cd': Cd,
        'hidden_layers': [128, 128, 128, 128]
    },
    'history': history,
    'best_loss': best_loss
}

torch.save(checkpoint, checkpoint_dir / "inverse_pinn_latest.pth")
print(f"\n✅ Checkpoint salvo em {checkpoint_dir / 'inverse_pinn_latest.pth'}")

# %% Função utilitária para uso
def find_trajectory(x_target, alpha=0.5):
    """
    Encontra parâmetros de tiro para atingir x_target.
    
    Args:
        x_target: Distância do alvo em metros
        alpha: Parâmetro α ∈ [0,1] que seleciona a solução
               α=0 → ângulo baixo, α=1 → ângulo alto
    
    Returns:
        dict com V0, theta_deg, T_impact
    """
    model.eval()
    with torch.no_grad():
        x_norm = torch.tensor([normalize_x(x_target)], device=device)
        a = torch.tensor([alpha], device=device)
        
        v0, theta_rad, T = model(x_norm, a)
        
        return {
            'V0': v0.item(),
            'theta_deg': np.degrees(theta_rad.item()),
            'T_impact': T.item(),
            'alpha': alpha
        }

# Demo
print("\n" + "="*70)
print("💡 COMO USAR")
print("="*70)
print("""
# Encontrar parâmetros para atingir 1000m com α=0.5:
result = find_trajectory(1000, alpha=0.5)
print(f"V0 = {result['V0']:.1f} m/s, θ = {result['theta_deg']:.1f}°")

# Varrer todas as soluções para um alvo:
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = find_trajectory(800, alpha=alpha)
    print(f"α={alpha:.2f}: V0={r['V0']:.1f}, θ={r['theta_deg']:.1f}°, T={r['T_impact']:.2f}s")
""")

# %%


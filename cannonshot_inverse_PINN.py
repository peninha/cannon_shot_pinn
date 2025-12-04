# %% [markdown]
# # Operador Neural para Balística Inversa
#
# ---
#
# **O Santo Graal da Balística!**
#
# Este operador neural recebe:
# - `x_impact`: distância do alvo (m)
# - `θ₀`: ângulo de lançamento desejado (rad)
#
# E retorna diretamente:
# - `V₀`: velocidade inicial necessária
# - `T`: tempo de voo
#
# Sem precisar de NENHUM solver iterativo!
#
# A rede aprende a função inversa satisfazendo:
# - x(T) = x_impact (condição de contorno no alvo)
# - y(T) = 0 (impacto no solo)
# - vy(T) < 0 (chegando de cima)
#

# %% Importações (primeiro!)
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
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Garante que o diretório de trabalho seja a raiz do projeto
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# %% Configuração
# ======================================================================
# HIPERPARÂMETROS
# ======================================================================

# Range de V0
V0_MIN = 10.0           # m/s
V0_MAX = 250.0          # m/s

# Range de θ0
THETA_MIN = 10.0        # graus
THETA_MAX = 80.0        # graus

# Arquitetura da rede
HIDDEN_LAYERS = [64, 64, 64]  # Camadas ocultas

# Física
g = 9.81
Cd = 0.47
m = 1.0
rho = 1.225
densidade_chumbo = 11340
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)
A = np.pi * (diametro / 2) ** 2
k_drag = (rho * Cd * A) / (2 * m)

# Treinamento
ADAM_EPOCHS = 50000
LBFGS_STEPS = 2000      # 0 para desativar L-BFGS
BATCH_SIZE = 256*16
LR = 1e-3

# Retomar de checkpoint (None para começar do zero)
RESUME_FROM = None  # ou "inverse_pinn_latest.pth" para continuar
RESUME_FROM = "inverse_pinn_latest.pth"  # ou "inverse_pinn_latest.pth" para continuar

# Integrador: 'heun', 'rk4', 'dopri5'
INTEGRATOR = 'dopri5'
N_INTEGRATION_STEPS = 200  # Usado por heun e rk4 (dopri5 é adaptativo)

# Pesos da loss
LAMBDA_X = 1.0      # Condição x(T) = x_target
LAMBDA_Y = 1.0     # Condição y(T) = 0
LAMBDA_VY = 0.1     # Condição vy(T) < 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Descomente para forçar CPU
print(f"Device: {device}")
print(f"Integrador: {INTEGRATOR}" + (f" (n_steps={N_INTEGRATION_STEPS})" if INTEGRATOR != 'dopri5' else " (adaptativo)"))
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print(f"Lambda X: {LAMBDA_X}")
print(f"Lambda Y: {LAMBDA_Y}")
print(f"Lambda Vy: {LAMBDA_VY}")

# ----------------------------------------------------------------------
# Cálculo automático do range do alvo (X_IMPACT_MIN, X_IMPACT_MAX)
# ----------------------------------------------------------------------
def _calcular_alcance(v0, theta_deg):
    """
    Calcula o alcance de um tiro com arrasto usando scipy.
    Integra até y=0 e retorna x no impacto.
    """
    theta_rad = np.radians(theta_deg)
    
    def dynamics(t, state):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        return [vx, vy, -k_drag * v_mag * vx, -g - k_drag * v_mag * vy]
    
    def hit_ground(t, state):
        return state[1]  # y = 0
    hit_ground.terminal = True
    hit_ground.direction = -1  # Detecta quando y cruza zero descendo
    
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    
    # Tempo máximo estimado (vácuo com margem)
    t_max = 3 * v0 * np.sin(theta_rad) / g
    
    sol = solve_ivp(dynamics, [0, t_max], [0, 0, vx0, vy0], 
                    events=hit_ground, dense_output=True)
    
    if sol.t_events[0].size > 0:
        return sol.sol(sol.t_events[0][0])[0]  # x no impacto
    return sol.y[0, -1]  # Fallback

# X_IMPACT_MIN: alcance MÁXIMO com V0_MIN (ângulo ótimo)
# Assim, qualquer x >= X_IMPACT_MIN é alcançável com V0 >= V0_MIN para algum θ
_resultado_v0min = minimize_scalar(
    lambda theta: -_calcular_alcance(V0_MIN, theta),  # Negativo para maximizar
    bounds=(THETA_MIN, THETA_MAX),
    method='bounded'
)
_theta_otimo_v0min = _resultado_v0min.x
_X_IMPACT_MIN_teorico = _calcular_alcance(V0_MIN, _theta_otimo_v0min)

# X_IMPACT_MAX: alcance MÁXIMO com V0_MAX (ângulo ótimo)
_resultado_v0max = minimize_scalar(
    lambda theta: -_calcular_alcance(V0_MAX, theta),  # Negativo para maximizar
    bounds=(20, 50),
    method='bounded'
)
_theta_otimo_v0max = _resultado_v0max.x
_X_IMPACT_MAX_teorico = _calcular_alcance(V0_MAX, _theta_otimo_v0max)

# Margem de segurança: CONTRAI o range para garantir soluções físicas
X_IMPACT_MIN = _X_IMPACT_MIN_teorico * 1.05  # 5% maior (contrai para cima)
X_IMPACT_MAX = _X_IMPACT_MAX_teorico * 0.95  # 5% menor (contrai para baixo)

print(f"k_drag = {k_drag:.6f}")
print(f"Alcance máximo com V0_MIN={V0_MIN}: {_X_IMPACT_MIN_teorico:.1f}m (θ_ótimo ≈ {_theta_otimo_v0min:.1f}°)")
print(f"Alcance máximo com V0_MAX={V0_MAX}: {_X_IMPACT_MAX_teorico:.1f}m (θ_ótimo ≈ {_theta_otimo_v0max:.1f}°)")
print(f"Range de treino:  [{X_IMPACT_MIN:.1f}m, {X_IMPACT_MAX:.1f}m]")

# Limpa variáveis temporárias
del _resultado_v0min, _resultado_v0max
del _X_IMPACT_MIN_teorico, _X_IMPACT_MAX_teorico
del _theta_otimo_v0min, _theta_otimo_v0max

# ======================================================================


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

# %% Integradores diferenciáveis
# ----------------------------------------------------------------------
# Heun (RK2) - Rápido e razoavelmente preciso
# ----------------------------------------------------------------------
def _integrate_heun(v0, theta_rad, T, n_steps):
    """Método de Heun (RK2) - bom balanço velocidade/precisão."""
    batch_size = v0.shape[0]
    
    x = torch.zeros(batch_size, device=v0.device)
    y = torch.zeros(batch_size, device=v0.device)
    vx = v0 * torch.cos(theta_rad)
    vy = v0 * torch.sin(theta_rad)
    
    dt = T / n_steps
    
    def derivs(vx, vy):
        v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
        ax = -k_drag * v_mag * vx
        ay = -g - k_drag * v_mag * vy
        return ax, ay
    
    for _ in range(n_steps):
        # Predictor (Euler)
        ax1, ay1 = derivs(vx, vy)
        vx_pred = vx + ax1 * dt
        vy_pred = vy + ay1 * dt
        
        # Corrector
        ax2, ay2 = derivs(vx_pred, vy_pred)
        
        x = x + vx * dt + 0.5 * ax1 * dt**2
        y = y + vy * dt + 0.5 * ay1 * dt**2
        vx = vx + 0.5 * (ax1 + ax2) * dt
        vy = vy + 0.5 * (ay1 + ay2) * dt
    
    return x, y, vx, vy

# ----------------------------------------------------------------------
# RK4 Manual - Mais preciso, mais lento
# ----------------------------------------------------------------------
def _integrate_rk4(v0, theta_rad, T, n_steps):
    """RK4 manual - alta precisão."""
    batch_size = v0.shape[0]
    
    x = torch.zeros(batch_size, device=v0.device)
    y = torch.zeros(batch_size, device=v0.device)
    vx = v0 * torch.cos(theta_rad)
    vy = v0 * torch.sin(theta_rad)
    
    dt = T / n_steps
    
    def derivs(x, y, vx, vy):
        v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
        ax = -k_drag * v_mag * vx
        ay = -g - k_drag * v_mag * vy
        return vx, vy, ax, ay
    
    for _ in range(n_steps):
        dx1, dy1, dvx1, dvy1 = derivs(x, y, vx, vy)
        dx2, dy2, dvx2, dvy2 = derivs(x + 0.5*dt*dx1, y + 0.5*dt*dy1, 
                                       vx + 0.5*dt*dvx1, vy + 0.5*dt*dvy1)
        dx3, dy3, dvx3, dvy3 = derivs(x + 0.5*dt*dx2, y + 0.5*dt*dy2,
                                       vx + 0.5*dt*dvx2, vy + 0.5*dt*dvy2)
        dx4, dy4, dvx4, dvy4 = derivs(x + dt*dx3, y + dt*dy3,
                                       vx + dt*dvx3, vy + dt*dvy3)
        
        x = x + (dt/6) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        y = y + (dt/6) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        vx = vx + (dt/6) * (dvx1 + 2*dvx2 + 2*dvx3 + dvx4)
        vy = vy + (dt/6) * (dvy1 + 2*dvy2 + 2*dvy3 + dvy4)
    
    return x, y, vx, vy

# ----------------------------------------------------------------------
# dopri5 via torchdiffeq - Adaptativo, muito preciso
# ----------------------------------------------------------------------
def _integrate_dopri5(v0, theta_rad, T, n_steps=None):
    """dopri5 (Dormand-Prince) via torchdiffeq - adaptativo."""
    batch_size = v0.shape[0]
    
    vx0 = v0 * torch.cos(theta_rad)
    vy0 = v0 * torch.sin(theta_rad)
    
    state0 = torch.stack([
        torch.zeros(batch_size, device=v0.device),
        torch.zeros(batch_size, device=v0.device),
        vx0, vy0
    ], dim=1)
    
    T_batch = T  # closure
    
    def dynamics(tau, state):
        x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
        
        dx_dt = vx
        dy_dt = vy
        dvx_dt = -k_drag * v_mag * vx
        dvy_dt = -g - k_drag * v_mag * vy
        
        return torch.stack([
            T_batch * dx_dt,
            T_batch * dy_dt,
            T_batch * dvx_dt,
            T_batch * dvy_dt
        ], dim=1)
    
    tau_span = torch.tensor([0.0, 1.0], device=v0.device)
    solution = odeint(dynamics, state0, tau_span, method='dopri5')
    final_state = solution[-1]
    
    return final_state[:, 0], final_state[:, 1], final_state[:, 2], final_state[:, 3]

# ----------------------------------------------------------------------
# Wrapper que seleciona o integrador baseado em INTEGRATOR
# ----------------------------------------------------------------------
def integrate_trajectory(v0, theta_rad, T):
    """
    Integra a trajetória balística com arrasto.
    
    O integrador usado é definido pela variável global INTEGRATOR:
    - 'heun': Método de Heun (RK2) - rápido
    - 'rk4': Runge-Kutta 4ª ordem - preciso
    - 'dopri5': Dormand-Prince adaptativo - muito preciso
    
    Args:
        v0: Velocidade inicial [batch]
        theta_rad: Ângulo em radianos [batch]
        T: Tempo total [batch]
    
    Returns:
        x_final, y_final, vx_final, vy_final: Estados finais [batch]
    """
    if INTEGRATOR == 'heun':
        return _integrate_heun(v0, theta_rad, T, N_INTEGRATION_STEPS)
    elif INTEGRATOR == 'rk4':
        return _integrate_rk4(v0, theta_rad, T, N_INTEGRATION_STEPS)
    elif INTEGRATOR == 'dopri5':
        return _integrate_dopri5(v0, theta_rad, T)
    else:
        raise ValueError(f"Integrador desconhecido: {INTEGRATOR}. Use 'heun', 'rk4' ou 'dopri5'.")

# %% Definição da Rede Neural
class InverseBallisticPINN(nn.Module):
    """
    Operador Neural para Balística Inversa.
    
    Input: [x_impact_norm, theta_norm] - distância e ângulo normalizados
    Output: [V₀, T] que satisfazem x(T) = x_impact, y(T) = 0
    """
    
    def __init__(self, hidden_layers=None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = HIDDEN_LAYERS
        
        # Arquitetura MLP
        layers = []
        input_dim = 2  # [x_impact_norm, theta_norm]
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 2))  # [V0_raw, T_raw]
        
        self.net = nn.Sequential(*layers)
        
        # Escalas para outputs
        self.V0_min = V0_MIN
        self.V0_max = V0_MAX
        self.theta_min_rad = np.radians(THETA_MIN)
        self.theta_max_rad = np.radians(THETA_MAX)
        
        # T_max estimado (baseado em vácuo com V0_max e θ=90°)
        self.T_max = 2 * V0_MAX / g * 1.5  # margem de segurança
    
    def forward(self, x_impact_norm, theta_norm):
        """
        Forward pass.
        
        Args:
            x_impact_norm: x_impact normalizado [-1, 1] [batch]
            theta_norm: θ normalizado [0, 1] [batch]
        
        Returns:
            v0: velocidade inicial [batch]
            T: tempo de voo [batch]
        """
        # Concatena inputs
        inputs = torch.stack([x_impact_norm, theta_norm], dim=1)
        
        # Forward
        raw = self.net(inputs)
        
        # Ativações para garantir ranges físicos
        # V0: softplus → [V0_min, +∞)
        # Isso permite que a rede proponha V0 > V0_MAX em casos inalcançáveis,
        # e usamos V0_MAX apenas como limite físico conhecido do canhão.
        v0 = self.V0_min + torch.nn.functional.softplus(raw[:, 0])
        
        # T: softplus para garantir T > 0, escalado
        T = torch.nn.functional.softplus(raw[:, 1]) * self.T_max / 5 + 0.2
        
        # Converte theta_norm para radianos
        theta_rad = theta_norm * (self.theta_max_rad - self.theta_min_rad) + self.theta_min_rad
        
        return v0, theta_rad, T

# %% Função de Loss
def physics_loss(model, x_impact_norm, theta_norm, x_impact_real):
    """
    Calcula a loss baseada nas condições de contorno físicas.
    
    Args:
        model: InverseBallisticPINN
        x_impact_norm: x_impact normalizado [batch]
        theta_norm: θ normalizado [0, 1] [batch]
        x_impact_real: x_impact em metros [batch]
    
    Returns:
        loss_total, loss_x, loss_y, loss_vy
    """
    # Forward pass
    v0, theta_rad, T = model(x_impact_norm, theta_norm)
    
    # Integra trajetória
    x_final, y_final, vx_final, vy_final = integrate_trajectory(v0, theta_rad, T)
    
    # Loss de condição de contorno em x
    loss_x = torch.mean((x_final - x_impact_real)**2)
    
    # Loss de condição de contorno em y (deve ser zero)
    loss_y = torch.mean(y_final**2)
    
    # Loss de velocidade vertical (deve ser negativa - descendo)
    # Penaliza se vy > 0
    loss_vy = torch.mean(torch.relu(vy_final)**2)
    
    # Loss total
    loss_total = (LAMBDA_X * loss_x + 
                  LAMBDA_Y * loss_y + 
                  LAMBDA_VY * loss_vy)
    
    return loss_total, loss_x, loss_y, loss_vy

# %% Gerador de dados de treinamento
def generate_batch(batch_size, device):
    """
    Gera um batch de dados de treinamento.
    
    Amostra x_impact e θ uniformemente.
    """
    # x_impact uniforme em [X_MIN, X_MAX]
    x_impact = torch.rand(batch_size, device=device) * (X_IMPACT_MAX - X_IMPACT_MIN) + X_IMPACT_MIN
    x_impact_norm = normalize_x(x_impact)
    
    # θ normalizado uniforme em [0, 1] → mapeia para [THETA_MIN, THETA_MAX]
    theta_norm = torch.rand(batch_size, device=device)
    
    return x_impact_norm, theta_norm, x_impact

# %% Instancia modelo
model = InverseBallisticPINN(hidden_layers=HIDDEN_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

print(f"\nModelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
print(f"Arquitetura: 2 → {' → '.join(map(str, HIDDEN_LAYERS))} → 2")

# %% Carrega checkpoint se especificado
previous_epochs = 0
history = {
    'loss': [],
    'loss_x': [],
    'loss_y': [],
    'loss_vy': []
}
best_loss = float('inf')
best_state = None

if RESUME_FROM is not None:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "inverse_pinn" / RESUME_FROM
    if checkpoint_path.exists():
        print(f"\n📂 Carregando checkpoint: {RESUME_FROM}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Carrega estado do modelo
        model.load_state_dict(checkpoint['model_state'])
        
        # Carrega otimizador se disponível
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Carrega histórico se disponível
        if 'history' in checkpoint:
            history = checkpoint['history']
            previous_epochs = len(history['loss'])
        
        # Carrega melhor loss
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
            best_state = checkpoint['model_state']
        
        print(f"✅ Checkpoint carregado! Épocas anteriores: {previous_epochs}, Melhor loss: {best_loss:.6f}")
    else:
        print(f"⚠️ Checkpoint não encontrado: {checkpoint_path}")
        print("Iniciando do zero...")

# %% Treinamento - Fase 1: Adam
print("\n" + "="*70)
print("TREINAMENTO - FASE 1: ADAM")
if previous_epochs > 0:
    print(f"(Continuando de {previous_epochs} épocas)")
print("="*70)

start_time = time.time()

pbar = tqdm(range(ADAM_EPOCHS), desc="Adam")
for epoch in pbar:
    model.train()
    
    # Gera batch
    x_impact_norm, theta_norm, x_impact_real = generate_batch(BATCH_SIZE, device)
    
    # Forward + loss
    optimizer.zero_grad()
    loss, loss_x, loss_y, loss_vy = physics_loss(
        model, x_impact_norm, theta_norm, x_impact_real
    )
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Scheduler
    scheduler.step(loss.item())
    
    # Logging
    history['loss'].append(loss.item())
    history['loss_x'].append(loss_x.item())
    history['loss_y'].append(loss_y.item())
    history['loss_vy'].append(loss_vy.item())
    
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

adam_time = time.time() - start_time
print(f"\nAdam concluído em {adam_time/60:.1f} minutos")
print(f"Melhor loss (Adam): {best_loss:.6f}")

# %% Treinamento - Fase 2: L-BFGS (opcional)
if LBFGS_STEPS > 0:
    print("\n" + "="*70)
    print("TREINAMENTO - FASE 2: L-BFGS")
    print("="*70)
    
    # Carrega o melhor estado do Adam antes de começar L-BFGS
    model.load_state_dict(best_state)
    
    # Dados fixos para L-BFGS (batch maior para estabilidade)
    lbfgs_batch_size = min(BATCH_SIZE * 4, 1024)
    x_lbfgs, theta_lbfgs, x_real_lbfgs = generate_batch(lbfgs_batch_size, device)
    
    # Otimizador L-BFGS
    lbfgs_optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_STEPS,
        history_size=50,
        line_search_fn='strong_wolfe'
    )
    
    lbfgs_losses = []
    lbfgs_iter = [0]  # Contador de iterações internas do L-BFGS
    
    def closure():
        lbfgs_optimizer.zero_grad()
        loss, loss_x, loss_y, loss_vy = physics_loss(
            model, x_lbfgs, theta_lbfgs, x_real_lbfgs
        )
        loss.backward()
        
        lbfgs_losses.append(loss.item())
        history['loss'].append(loss.item())
        history['loss_x'].append(loss_x.item())
        history['loss_y'].append(loss_y.item())
        history['loss_vy'].append(loss_vy.item())
        lbfgs_iter[0] += 1
        
        # Logging periódico
        if lbfgs_iter[0] % 50 == 0 or lbfgs_iter[0] == 1:
            print(f"[L-BFGS] iter={lbfgs_iter[0]:04d}  loss={loss.item():.3e}")
        
        return loss
    
    start_lbfgs = time.time()
    
    # Uma chamada a step realiza até LBFGS_STEPS iterações internas,
    # chamando o closure várias vezes (como em cannonshot_PINN_vacuo).
    _ = lbfgs_optimizer.step(closure)
    
    # Atualiza melhor modelo com o melhor valor observado durante o L-BFGS
    if lbfgs_losses:
        min_lbfgs_loss = min(lbfgs_losses)
        if min_lbfgs_loss < best_loss:
            best_loss = min_lbfgs_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    lbfgs_time = time.time() - start_lbfgs
    print(f"\nL-BFGS concluído em {lbfgs_time/60:.1f} minutos (iterações internas: {lbfgs_iter[0]})")
    print(f"Melhor loss (L-BFGS): {best_loss:.6f}")

# Tempo total
total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Treinamento total: {total_time/60:.1f} minutos")
print(f"Melhor loss final: {best_loss:.6f}")
print(f"{'='*70}")

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

# %% Validação: testa para diferentes x_impact e θ
print("\n" + "="*70)
print("VALIDAÇÃO")
print("="*70)

x_test_values = [200, 500, 800, 1000, 1500]
theta_test_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]  # graus

print(f"\n{'x_target':>10} {'θ_input':>8} │ {'V0':>8} {'θ_out':>8} {'T':>8} │ {'x_sim':>10} {'y_sim':>10} {'vy_sim':>10} │ {'Erro x%':>8}")
print("-" * 110)

with torch.no_grad():
    for x_target in x_test_values:
        for theta_input in theta_test_values:
            x_norm = torch.tensor([normalize_x(x_target)], device=device, dtype=torch.float32)
            theta_n = torch.tensor([normalize_theta(theta_input)], device=device, dtype=torch.float32)
            
            v0, theta_rad, T = model(x_norm, theta_n)
            x_f, y_f, vx_f, vy_f = integrate_trajectory(v0, theta_rad, T)
            
            v0_val = v0.item()
            theta_deg = np.degrees(theta_rad.item())
            T_val = T.item()
            x_sim = x_f.item()
            y_sim = y_f.item()
            vy_sim = vy_f.item()
            
            erro_x = abs(x_sim - x_target) / x_target * 100
            
            status = "✓" if erro_x < 5 and abs(y_sim) < 10 and vy_sim < 0 else "✗"
            
            print(f"{x_target:10.0f} {theta_input:8.1f} │ {v0_val:8.1f} {theta_deg:8.2f} {T_val:8.2f} │ "
                  f"{x_sim:10.1f} {y_sim:10.2f} {vy_sim:10.2f} │ {erro_x:8.2f} {status}")
        print()

# %% Visualização das soluções
ax3 = axes[1, 0]
ax4 = axes[1, 1]

x_target = 800  # metros
theta_inputs = np.linspace(THETA_MIN, THETA_MAX, 20)  # graus
colors = plt.cm.plasma(np.linspace(0, 1, len(theta_inputs)))

def _dynamics_plot(t, state):
    """Dynamics for plotting (used by scipy)."""
    x, y, vx, vy = state
    v_mag = np.sqrt(vx**2 + vy**2)
    return [vx, vy, -k_drag * v_mag * vx, -g - k_drag * v_mag * vy]

with torch.no_grad():
    for i, theta_input in enumerate(theta_inputs):
        x_norm = torch.tensor([normalize_x(x_target)], device=device, dtype=torch.float32)
        theta_n = torch.tensor([normalize_theta(theta_input)], device=device, dtype=torch.float32)
        
        v0, theta_rad, T = model(x_norm, theta_n)
        
        # Simula trajetória completa para plotar
        v0_val = v0.item()
        theta_val = theta_rad.item()
        T_val = T.item()
        
        # Usa scipy para integração precisa (em vez de Euler impreciso)
        vx0 = v0_val * np.cos(theta_val)
        vy0 = v0_val * np.sin(theta_val)
        
        t_eval = np.linspace(0, T_val, 200)
        sol = solve_ivp(_dynamics_plot, [0, T_val], [0, 0, vx0, vy0], 
                        t_eval=t_eval, method='RK45')
        
        x_traj = sol.y[0]
        y_traj = sol.y[1]
        
        ax3.plot(x_traj, y_traj, '-', color=colors[i], alpha=0.7, linewidth=1.5)

ax3.scatter([x_target], [0], c='red', s=200, marker='o', zorder=10, label=f'Alvo: {x_target}m')
ax3.axhline(y=0, color='saddlebrown', linewidth=2, alpha=0.5)
ax3.set_xlabel('Distância (m)')
ax3.set_ylabel('Altura (m)')
ax3.set_title(f'Família de Trajetórias para x_target = {x_target}m\n(θ₀ varia de {THETA_MIN}° a {THETA_MAX}°)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot V0 vs θ
theta_dense = np.linspace(THETA_MIN, THETA_MAX, 50)
v0_vals = []
T_vals = []

with torch.no_grad():
    for theta_deg in theta_dense:
        x_norm = torch.tensor([normalize_x(x_target)], device=device, dtype=torch.float32)
        theta_n = torch.tensor([normalize_theta(theta_deg)], device=device, dtype=torch.float32)
        v0, theta_rad, T = model(x_norm, theta_n)
        v0_vals.append(v0.item())
        T_vals.append(T.item())

ax4.plot(theta_dense, v0_vals, 'b-', linewidth=2, label='V₀ (m/s)')
ax4_twin = ax4.twinx()
ax4_twin.plot(theta_dense, T_vals, 'g-', linewidth=2, label='T (s)')
ax4.set_xlabel('θ₀ (°)')
ax4.set_ylabel('V₀ (m/s)', color='blue')
ax4_twin.set_ylabel('T (s)', color='green')
ax4.set_title(f'V₀ e T como função de θ₀ (x_target = {x_target}m)')
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

total_epochs = previous_epochs + ADAM_EPOCHS

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
        'hidden_layers': HIDDEN_LAYERS,
        'integrator': INTEGRATOR
    },
    'history': history,
    'best_loss': best_loss,
    'total_epochs': total_epochs,
    'lbfgs_done': LBFGS_STEPS > 0
}

torch.save(checkpoint, checkpoint_dir / "inverse_pinn_latest.pth")
print(f"\n✅ Checkpoint salvo em {checkpoint_dir / 'inverse_pinn_latest.pth'}")
print(f"   Épocas totais: {total_epochs}, Melhor loss: {best_loss:.6f}")

# %% Função utilitária para uso
def find_trajectory(x_target, theta_deg=45.0):
    """
    Encontra parâmetros de tiro para atingir x_target com ângulo especificado.
    
    Args:
        x_target: Distância do alvo em metros
        theta_deg: Ângulo de lançamento em graus
    
    Returns:
        dict com V0, theta_deg, T_impact
    """
    model.eval()
    with torch.no_grad():
        x_norm = torch.tensor([normalize_x(x_target)], device=device, dtype=torch.float32)
        theta_n = torch.tensor([normalize_theta(theta_deg)], device=device, dtype=torch.float32)
        
        v0, theta_rad, T = model(x_norm, theta_n)
        
        return {
            'V0': v0.item(),
            'theta_deg': np.degrees(theta_rad.item()),
            'T_impact': T.item()
        }

# %%


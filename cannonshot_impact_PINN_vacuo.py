# %% [markdown]
# # PINN para encontrar ponto de impacto de tiro de canhão no vácuo
#
# ---
#
# Rede mapeia `(V0, theta0) -> (T_impact, x_impact)`
#
# Usamos integração numérica (trapézio) para impor a física:
# >  x = ∫₀^T vx dt = ∫₀^T v0*cos(θ) dt<br>
# >  y = ∫₀^T vy dt = ∫₀^T (v0*sin(θ) - g*t) dt<br>
#
# Contornos:
# >  y(T_impact) = 0  (projétil atinge o solo)<br>
# >  x(T_impact) = x_impact  (consistência)<br>
# >  vy(T_impact) < 0  (evita solução trivial T=0)<br>
#

# %% Importações
# ====== Importações ======
import inspect
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt

# Garante que o diretório de trabalho seja a raiz do projeto
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# %% Parâmetros físicos
# ====== Parâmetros físicos ======
g = 9.81  # m/s^2

# Ranges de entrada (para normalização e amostragem)
V0_min, V0_max = 10.0, 250.0        # m/s
theta0_min, theta0_max = 10.0, 80.0  # graus

# Escalas esperadas para outputs (baseadas na física)
# T_max ≈ 2 * V0_max * sin(theta0_max) / g
T_scale = 2 * V0_max * np.sin(np.radians(theta0_max)) / g  # ≈ 50s
# x_max ≈ V0_max² / g (para theta=45°)
X_scale = V0_max**2 / g  # ≈ 6370m

# %% Parâmetros de treino
# ====== Parâmetros de treino ======

# Pontos de amostragem
adam_steps = 10000      # número de passos do Adam
lbfgs_steps = 3000     # máximo de iterações do L-BFGS (0 = disabled)
N_samples = 512        # número de amostras (V0, theta0) por batch
N_integration = 1000    # pontos para integração numérica

# Rede neural
layers = [2, 64, 64, 64, 2]  # PINN: 2D (V0, theta0) -> 2D (T_impact, x_impact)
learning_rate = 1e-3         # taxa de aprendizado do Adam
deterministic = False        # se True, torna o treinamento determinístico
seed = 42                    # seed para reprodução

# Pesos das perdas
lambda_y = 0.45    # peso da perda y (y_impact = 0)
lambda_x = 0.45    # peso da perda x (x integrado = x_impact)
lambda_vy = 1 - lambda_y - lambda_x  # peso da perda vy (vy_impact < 0)

# Parâmetro de penalização vy_impact
epsilon_vy = 0.5   # margem para penalizar vy_impact não negativo (m/s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Treinamento determinístico com seed {seed}")
else:
    print("Treinamento não determinístico")

# %% Ground Truth analítico
# ====== Ground Truth analítico ======
def impact_ground_truth(v0, theta0_rad):
    """
    Calcula T_impact e x_impact analiticamente (sem arrasto, y0=0).
    v0: velocidade inicial (m/s)
    theta0_rad: ângulo em radianos
    Retorna: T_impact, x_impact
    """
    v0 = np.asarray(v0)
    theta0_rad = np.asarray(theta0_rad)
    
    # T_impact = 2 * v0 * sin(theta) / g
    T_impact = 2 * v0 * np.sin(theta0_rad) / g
    
    # x_impact = v0 * cos(theta) * T_impact = v0^2 * sin(2*theta) / g
    x_impact = v0 * np.cos(theta0_rad) * T_impact
    
    return T_impact, x_impact

# %% Funções utilitárias
# ====== Funções utilitárias ======
def sample_inputs(n_samples):
    """Amostra V0 e theta0 uniformemente nos ranges definidos."""
    v0 = np.random.uniform(V0_min, V0_max, n_samples)
    theta0_deg = np.random.uniform(theta0_min, theta0_max, n_samples)
    theta0_rad = np.radians(theta0_deg)
    return v0, theta0_rad

def normalize_inputs(v0, theta0_rad):
    """Normaliza inputs para [-1, 1]."""
    v0_norm = 2 * (v0 - V0_min) / (V0_max - V0_min) - 1
    theta0_min_rad = np.radians(theta0_min)
    theta0_max_rad = np.radians(theta0_max)
    theta0_norm = 2 * (theta0_rad - theta0_min_rad) / (theta0_max_rad - theta0_min_rad) - 1
    return v0_norm, theta0_norm

# %% Definição da rede
# ====== MLP com integração numérica ======
class ImpactPINN(nn.Module):
    def __init__(self, layers, n_integration=100, lambda_y=0.1, lambda_x=0.1, lambda_vy=0.8, 
                 epsilon_vy=1.5, T_scale=50.0, X_scale=6000.0):
        super().__init__()
        dims = layers
        mods = []
        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)
        
        self.n_integration = n_integration
        self.g = g
        self.lambda_y = lambda_y
        self.lambda_x = lambda_x
        self.lambda_vy = lambda_vy
        self.epsilon_vy = epsilon_vy
        self.T_scale = T_scale
        self.X_scale = X_scale

        # Inicialização Xavier/Glorot para estabilidade
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
        inputs: tensor de shape (batch, 2) com [v0_norm, theta0_norm]
        Retorna: tensor de shape (batch, 2) com [T_impact, x_impact]
        
        A rede prediz valores normalizados [0, 1] que são escalados para os ranges físicos.
        Usamos sigmoid para garantir outputs em [0, 1], depois escalamos.
        """
        raw = self.net(inputs)
        
        # T_impact: sigmoid * T_scale (range: [0, T_scale])
        # Adicionamos pequeno offset para evitar T=0
        T_impact = torch.sigmoid(raw[:, 0:1]) * self.T_scale + 0.1
        
        # x_impact: sigmoid * X_scale (range: [0, X_scale])
        x_impact = torch.sigmoid(raw[:, 1:2]) * self.X_scale
        
        return torch.cat([T_impact, x_impact], dim=1)
    
    def compute_physics(self, v0, theta0_rad, T_impact):
        """
        Calcula y_impact e x_integrated usando integração numérica (trapézio).
        
        v0: tensor (batch,) - velocidade inicial
        theta0_rad: tensor (batch,) - ângulo em radianos
        T_impact: tensor (batch, 1) - tempo de impacto predito
        
        Retorna: y_integrated, x_integrated, vy_impact
        """
        batch_size = v0.shape[0]
        device = v0.device
        
        # Gera pontos de tempo para integração: t ∈ [0, T_impact]
        # Shape: (batch, n_integration)
        t_normalized = torch.linspace(0, 1, self.n_integration, device=device)
        t_normalized = t_normalized.unsqueeze(0).expand(batch_size, -1)  # (batch, n_integration)
        
        # t = t_normalized * T_impact para cada amostra
        t = t_normalized * T_impact  # (batch, n_integration)
        
        # Componentes de velocidade inicial
        vx0 = v0 * torch.cos(theta0_rad)  # (batch,)
        vy0 = v0 * torch.sin(theta0_rad)  # (batch,)
        
        # Velocidades ao longo do tempo
        # vx(t) = vx0 (constante no vácuo)
        # vy(t) = vy0 - g*t
        vx = vx0.unsqueeze(1).expand(-1, self.n_integration)  # (batch, n_integration)
        vy = vy0.unsqueeze(1) - self.g * t  # (batch, n_integration)
        
        # Integração por trapézio
        # x = ∫ vx dt
        x_integrated = torch.trapezoid(vx, t, dim=1)  # (batch,)
        
        # y = ∫ vy dt  
        y_integrated = torch.trapezoid(vy, t, dim=1)  # (batch,)
        
        # vy no momento do impacto
        vy_impact = vy0 - self.g * T_impact.squeeze(1)  # (batch,)
        
        return y_integrated, x_integrated, vy_impact
    
    def compute_loss(self, v0, theta0_rad, outputs):
        """
        Calcula a perda baseada nos contornos físicos.
        
        Contornos:
        1. y(T_impact) = 0 (projétil atinge o solo)
        2. x(T_impact) = x_impact (consistência com predição)
        3. vy(T_impact) < 0 (evita solução trivial)
        """
        T_impact = outputs[:, 0:1]  # (batch, 1)
        x_impact = outputs[:, 1:2]  # (batch, 1)
        
        # Calcula integrais
        y_integrated, x_integrated, vy_impact = self.compute_physics(
            v0, theta0_rad, T_impact
        )
        
        # Loss 1: y(T_impact) = 0 (normalizado por X_scale²)
        loss_y = ((y_integrated / self.X_scale) ** 2).mean()
        
        # Loss 2: x integrado = x_impact predito (normalizado por X_scale²)
        loss_x = (((x_integrated - x_impact.squeeze(1)) / self.X_scale) ** 2).mean()
        
        # Loss 3: vy_impact < 0 (penaliza se vy >= 0)
        # Usamos ReLU: max(0, vy_impact + epsilon_vy) para margem de segurança
        # Normalizado para escala similar (divide por velocidade típica)
        v_scale = self.X_scale / self.T_scale  # velocidade típica
        loss_vy = (torch.relu(vy_impact + self.epsilon_vy) / v_scale).mean()
        
        # Perda total com pesos configuráveis
        loss = self.lambda_y * loss_y + self.lambda_x * loss_x + self.lambda_vy * loss_vy
        
        return loss, loss_y, loss_x, loss_vy

# %% Criação do modelo
default_dtype = torch.get_default_dtype()
g_t = torch.tensor(g, device=device, dtype=default_dtype)

model = ImpactPINN(layers, n_integration=N_integration, lambda_y=lambda_y, lambda_x=lambda_x, 
                   lambda_vy=lambda_vy, epsilon_vy=epsilon_vy, T_scale=T_scale, X_scale=X_scale).to(device)

# %% Otimizadores
# ====== Otimizadores ======
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
loss_y_history = []
loss_x_history = []
loss_vy_history = []

# Rastreia melhor modelo
best_loss = float('inf')
best_model_state = None
best_step = -1

if lbfgs_steps > 0:
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps, line_search_fn='strong_wolfe')

# %% Diretórios
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_vacuo"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
images_dir = PROJECT_ROOT / "imagens" / "impact_vacuo"
images_dir.mkdir(parents=True, exist_ok=True)

# %% Treinamento com Adam
# ====== Loop de treino (Adam) ======
model.train()
print("\n" + "="*50)
print("Iniciando treinamento com Adam...")
print("="*50)

for step in range(adam_steps):
    adam.zero_grad()
    
    # Amostra batch de inputs
    v0_np, theta0_rad_np = sample_inputs(N_samples)
    v0_norm_np, theta0_norm_np = normalize_inputs(v0_np, theta0_rad_np)
    
    # Converte para torch
    v0 = torch.tensor(v0_np, device=device, dtype=default_dtype)
    theta0_rad = torch.tensor(theta0_rad_np, device=device, dtype=default_dtype)
    
    inputs_norm = torch.stack([
        torch.tensor(v0_norm_np, device=device, dtype=default_dtype),
        torch.tensor(theta0_norm_np, device=device, dtype=default_dtype)
    ], dim=1)  # (batch, 2)
    
    # Forward
    outputs = model(inputs_norm)
    
    # Loss
    loss, loss_y, loss_x, loss_vy = model.compute_loss(v0, theta0_rad, outputs)
    
    loss.backward()
    adam.step()
    
    loss_history.append(loss.item())
    loss_y_history.append(loss_y.item())
    loss_x_history.append(loss_x.item())
    loss_vy_history.append(loss_vy.item())
    
    # Rastreia melhor modelo
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_step = step
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if step % 500 == 0:
        print(f"[Adam] step={step:04d}  loss={loss.item():.6e}  L_y={loss_y.item():.3e}  L_x={loss_x.item():.3e}  L_vy={loss_vy.item():.3e}")

# %% (Opcional) Refinamento com L-BFGS
# ====== (Opcional) Refinamento com L-BFGS ======
lbfgs_losses = []
lbfgs_y_losses = []
lbfgs_x_losses = []
lbfgs_vy_losses = []

if lbfgs_steps > 0:
    print("\n" + "="*50)
    print(f"Iniciando refinamento com L-BFGS (max {lbfgs_steps} iterações)...")
    print("="*50)
    
    lbfgs_iter = [0]
    best_tracking = {'loss': best_loss, 'step': best_step, 'state': best_model_state}
    
    # Gera batch fixo para L-BFGS (estabilidade)
    v0_np_fixed, theta0_rad_np_fixed = sample_inputs(N_samples * 4)  # mais amostras
    v0_norm_np_fixed, theta0_norm_np_fixed = normalize_inputs(v0_np_fixed, theta0_rad_np_fixed)
    
    v0_fixed = torch.tensor(v0_np_fixed, device=device, dtype=default_dtype)
    theta0_rad_fixed = torch.tensor(theta0_rad_np_fixed, device=device, dtype=default_dtype)
    inputs_norm_fixed = torch.stack([
        torch.tensor(v0_norm_np_fixed, device=device, dtype=default_dtype),
        torch.tensor(theta0_norm_np_fixed, device=device, dtype=default_dtype)
    ], dim=1)
    
    def closure():
        lbfgs.zero_grad()
        
        outputs = model(inputs_norm_fixed)
        loss, loss_y, loss_x, loss_vy = model.compute_loss(v0_fixed, theta0_rad_fixed, outputs)
        loss.backward()
        
        lbfgs_losses.append(loss.item())
        lbfgs_y_losses.append(loss_y.item())
        lbfgs_x_losses.append(loss_x.item())
        lbfgs_vy_losses.append(loss_vy.item())
        lbfgs_iter[0] += 1
        
        if lbfgs_iter[0] % 100 == 0 or lbfgs_iter[0] == 1:
            print(f"[L-BFGS] iter={lbfgs_iter[0]:03d}  loss={loss.item():.6e}  L_y={loss_y.item():.3e}  L_x={loss_x.item():.3e}  L_vy={loss_vy.item():.3e}")
        
        if loss.item() < best_tracking['loss']:
            best_tracking['loss'] = loss.item()
            best_tracking['step'] = f"LBFGS-{lbfgs_iter[0]}"
            best_tracking['state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        return loss
    
    lbfgs.step(closure)
    
    best_loss = best_tracking['loss']
    best_step = best_tracking['step']
    best_model_state = best_tracking['state']
    
    print(f"L-BFGS finalizado após {lbfgs_iter[0]} iterações")
    print(f"Loss final L-BFGS: {lbfgs_losses[-1]:.6e}\n")

# %% Salvamento do checkpoint
# ====== Checkpoint ======
layer_str = "_".join(str(n) for n in layers)
last_step = adam_steps - 1

# Formata lambdas para o nome do arquivo
lambda_y_str = f"{lambda_y:.2f}".replace(".", "p").rstrip("0").rstrip("p")
lambda_x_str = f"{lambda_x:.2f}".replace(".", "p").rstrip("0").rstrip("p")
lambda_vy_str = f"{lambda_vy:.2f}".replace(".", "p").rstrip("0").rstrip("p")
if lambda_y_str == "":
    lambda_y_str = "0"
if lambda_x_str == "":
    lambda_x_str = "0"
if lambda_vy_str == "":
    lambda_vy_str = "0"

name_parts = [
    f"impact-{layer_str}",
    str(last_step + 1),
    f"nint{N_integration}",
    f"nsamp{N_samples}",
    f"eps{epsilon_vy:.1f}".replace(".", "p"),
    f"ly{lambda_y_str}",
    f"lx{lambda_x_str}",
    f"lvy{lambda_vy_str}",
]

if lbfgs_steps > 0:
    name_parts.append(f"lbfgs{lbfgs_steps}")

checkpoint_name = "-".join(name_parts)
checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pth"

checkpoint_payload = {
    "model_state": model.state_dict(),
    "best_model_state": best_model_state,
    "step": last_step,
    "best_loss": best_loss,
    "best_step": best_step,
    "config": {
        "layers": layers,
        "adam_steps": adam_steps,
        "learning_rate": learning_rate,
        "N_samples": N_samples,
        "N_integration": N_integration,
        "lbfgs_steps": lbfgs_steps,
        "V0_range": [V0_min, V0_max],
        "theta0_range": [theta0_min, theta0_max],
        "lambda_y": lambda_y,
        "lambda_x": lambda_x,
        "lambda_vy": lambda_vy,
        "epsilon_vy": epsilon_vy,
        "T_scale": T_scale,
        "X_scale": X_scale,
    },
}

torch.save(checkpoint_payload, checkpoint_path)
print(f"\nCheckpoint salvo em {checkpoint_path}")
if isinstance(best_step, str) and "LBFGS" in best_step:
    print(f"Melhor loss: {best_loss:.6e} (encontrado durante L-BFGS: {best_step})")
else:
    print(f"Melhor loss: {best_loss:.6e} no passo {best_step}")

print("\nTreino finalizado.")

# %% Carregamento do melhor modelo para avaliação
# ====== Usar melhor modelo ======
if best_model_state is not None:
    best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
    model.load_state_dict(best_model_state_device)
    print(f"Usando melhor modelo (passo {best_step}) para avaliação")
else:
    print("Usando modelo final para avaliação")

# %% Avaliação
# ====== Avaliação ======
model.eval()

# Gera grid de avaliação
n_eval = 50
v0_eval = np.linspace(V0_min, V0_max, n_eval)
theta0_eval_deg = np.linspace(theta0_min, theta0_max, n_eval)
V0_grid, Theta0_grid_deg = np.meshgrid(v0_eval, theta0_eval_deg)
V0_flat = V0_grid.flatten()
Theta0_flat_rad = np.radians(Theta0_grid_deg.flatten())

# Ground truth
T_true, X_true = impact_ground_truth(V0_flat, Theta0_flat_rad)

# Predição
v0_norm_eval, theta0_norm_eval = normalize_inputs(V0_flat, Theta0_flat_rad)
inputs_eval = torch.stack([
    torch.tensor(v0_norm_eval, device=device, dtype=default_dtype),
    torch.tensor(theta0_norm_eval, device=device, dtype=default_dtype)
], dim=1)

with torch.no_grad():
    outputs_eval = model(inputs_eval)
    T_pred = outputs_eval[:, 0].cpu().numpy()
    X_pred = outputs_eval[:, 1].cpu().numpy()

# RMSE
rmse_T = np.sqrt(np.mean((T_pred - T_true)**2))
rmse_X = np.sqrt(np.mean((X_pred - X_true)**2))

print(f"\nRMSE T_impact: {rmse_T:.4e} s")
print(f"RMSE x_impact: {rmse_X:.4e} m")

# Erro relativo médio
rel_err_T = np.mean(np.abs(T_pred - T_true) / T_true) * 100
rel_err_X = np.mean(np.abs(X_pred - X_true) / X_true) * 100
print(f"Erro relativo médio T: {rel_err_T:.2f}%")
print(f"Erro relativo médio X: {rel_err_X:.2f}%")

# %% Visualizações
# ====== Visualizações ======

# Reshape para grid
T_true_grid = T_true.reshape(n_eval, n_eval)
T_pred_grid = T_pred.reshape(n_eval, n_eval)
X_true_grid = X_true.reshape(n_eval, n_eval)
X_pred_grid = X_pred.reshape(n_eval, n_eval)

# Gráfico das perdas
steps = np.arange(1, len(loss_history) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, loss_history, label="Loss total (Adam)", color='tab:blue')
plt.plot(steps, loss_y_history, label="Loss y (Adam)", color='tab:orange')
plt.plot(steps, loss_x_history, label="Loss x (Adam)", color='tab:green')
plt.plot(steps, loss_vy_history, label="Loss vy (Adam)", color='tab:red')

if len(lbfgs_losses) > 0:
    adam_end = steps[-1]
    plt.axvline(x=adam_end, color='gray', linestyle='--', linewidth=1.5, label='Início L-BFGS')
    lbfgs_steps_arr = np.arange(adam_end + 1, adam_end + len(lbfgs_losses) + 1)
    plt.plot(lbfgs_steps_arr, lbfgs_losses, label="Loss total (L-BFGS)", color='tab:blue', linestyle=':')

plt.yscale("log")
plt.xlabel("Iteração")
plt.ylabel("Perda")
plt.title("Evolução das perdas")
plt.legend(fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
# Comparação T_impact para theta0 = 45 graus
idx_45 = n_eval // 2
plt.plot(v0_eval, T_true_grid[idx_45, :], 'o-', label='T_impact GT', markersize=3)
plt.plot(v0_eval, T_pred_grid[idx_45, :], 'x--', label='T_impact PINN', markersize=3)
plt.xlabel("V0 (m/s)")
plt.ylabel("T_impact (s)")
plt.title(f"T_impact vs V0 (θ ≈ {theta0_eval_deg[idx_45]:.0f}°)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_loss.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Mapa de calor de erro
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# T_impact
im1 = axes[0, 0].contourf(V0_grid, Theta0_grid_deg, T_true_grid, levels=20, cmap='viridis')
axes[0, 0].set_title('T_impact (Ground Truth)')
axes[0, 0].set_xlabel('V0 (m/s)')
axes[0, 0].set_ylabel('θ0 (graus)')
plt.colorbar(im1, ax=axes[0, 0], label='s')

im2 = axes[0, 1].contourf(V0_grid, Theta0_grid_deg, T_pred_grid, levels=20, cmap='viridis')
axes[0, 1].set_title('T_impact (PINN)')
axes[0, 1].set_xlabel('V0 (m/s)')
axes[0, 1].set_ylabel('θ0 (graus)')
plt.colorbar(im2, ax=axes[0, 1], label='s')

T_err = np.abs(T_pred_grid - T_true_grid) / T_true_grid * 100
im3 = axes[0, 2].contourf(V0_grid, Theta0_grid_deg, T_err, levels=20, cmap='Reds')
axes[0, 2].set_title('Erro relativo T_impact (%)')
axes[0, 2].set_xlabel('V0 (m/s)')
axes[0, 2].set_ylabel('θ0 (graus)')
plt.colorbar(im3, ax=axes[0, 2], label='%')

# x_impact
im4 = axes[1, 0].contourf(V0_grid, Theta0_grid_deg, X_true_grid, levels=20, cmap='plasma')
axes[1, 0].set_title('x_impact (Ground Truth)')
axes[1, 0].set_xlabel('V0 (m/s)')
axes[1, 0].set_ylabel('θ0 (graus)')
plt.colorbar(im4, ax=axes[1, 0], label='m')

im5 = axes[1, 1].contourf(V0_grid, Theta0_grid_deg, X_pred_grid, levels=20, cmap='plasma')
axes[1, 1].set_title('x_impact (PINN)')
axes[1, 1].set_xlabel('V0 (m/s)')
axes[1, 1].set_ylabel('θ0 (graus)')
plt.colorbar(im5, ax=axes[1, 1], label='m')

X_err = np.abs(X_pred_grid - X_true_grid) / X_true_grid * 100
im6 = axes[1, 2].contourf(V0_grid, Theta0_grid_deg, X_err, levels=20, cmap='Reds')
axes[1, 2].set_title('Erro relativo x_impact (%)')
axes[1, 2].set_xlabel('V0 (m/s)')
axes[1, 2].set_ylabel('θ0 (graus)')
plt.colorbar(im6, ax=axes[1, 2], label='%')

plt.suptitle(f'Comparação PINN vs Ground Truth\nRMSE T: {rmse_T:.4f}s | RMSE X: {rmse_X:.4f}m', fontsize=12)
plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Scatter plot de predição vs ground truth
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(T_true, T_pred, alpha=0.5, s=10)
axes[0].plot([T_true.min(), T_true.max()], [T_true.min(), T_true.max()], 'r--', label='y=x')
axes[0].set_xlabel('T_impact GT (s)')
axes[0].set_ylabel('T_impact PINN (s)')
axes[0].set_title(f'T_impact: RMSE = {rmse_T:.4f}s')
axes[0].legend()
axes[0].grid(True)
axes[0].set_aspect('equal')

axes[1].scatter(X_true, X_pred, alpha=0.5, s=10)
axes[1].plot([X_true.min(), X_true.max()], [X_true.min(), X_true.max()], 'r--', label='y=x')
axes[1].set_xlabel('x_impact GT (m)')
axes[1].set_ylabel('x_impact PINN (m)')
axes[1].set_title(f'x_impact: RMSE = {rmse_X:.4f}m')
axes[1].legend()
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_scatter.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print("\nVisualizações salvas em", images_dir)

# %%


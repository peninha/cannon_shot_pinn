# %% [markdown]
# # PINN para encontrar ponto de impacto de tiro de canhão com arrasto
#
# ---
#
# Rede mapeia `(V0, theta0) -> (T_impact, x_impact)`
#
# Usamos integração numérica (Euler) para impor a física com arrasto:
# >  vx' = -k |v| vx<br>
# >  vy' = -g - k |v| vy<br>
# >  x = ∫₀^T vx dt<br>
# >  y = ∫₀^T vy dt<br>
#
# onde `k = (ρ · Cd · A) / (2 · m)` e `|v| = sqrt(vx² + vy²)`
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

# Parâmetros de arrasto
m = 1.0                          # kg - massa do projétil
rho = 1.225                      # kg/m^3 - densidade do ar
densidade_chumbo = 11340         # kg/m^3 - densidade do chumbo
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)  # m - diâmetro do projétil
A = np.pi * (diametro / 2) ** 2  # m^2 - área de seção reta do projétil
Cd = 0.47                        # coeficiente de arrasto (esfera)

# Constante de arrasto k = (rho * Cd * A) / (2 * m)
k_drag = (rho * Cd * A) / (2 * m)

# Escalas esperadas para outputs (baseadas na física - aproximação vácuo)
# Com arrasto, os valores serão menores, mas usamos vácuo como limite superior
T_scale = 2 * V0_max * np.sin(np.radians(theta0_max)) / g  # ≈ 50s
X_scale = V0_max**2 / g  # ≈ 6370m

# %% Parâmetros de treino
# ====== Parâmetros de treino ======

# Pontos de amostragem
adam_steps = 5000     # número de passos do Adam
lbfgs_steps = 3000     # máximo de iterações do L-BFGS (0 = disabled)
N_samples = 256        # número de amostras (V0, theta0) por batch (menor pois Euler é mais lento)
N_integration = 1000    # pontos para integração Euler (100 é suficiente, mais = mais lento)

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
epsilon_vy = 0.0   # margem para penalizar vy_impact não negativo (m/s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Treinamento determinístico com seed {seed}")
else:
    print("Treinamento não determinístico")

# Use para retomar treinamento de um checkpoint ou None para iniciar do zero
resume_from = None
resume_from = "impact_arrasto-2_64_64_64_2-20000-nint1000-nsamp256-eps0p0-ly0p45-lx0p45-lvy0p1-Cd0p47-lbfgs3000"
# Exemplo: resume_from = "impact_arrasto-2_64_64_64_2-10000-nint500-nsamp256-..."

# %% Ground Truth numérico
# ====== Ground Truth numérico (integração com arrasto) ======
def impact_ground_truth(v0_arr, theta0_rad_arr):
    """
    Calcula T_impact e x_impact numericamente (com arrasto).
    v0_arr: array de velocidades iniciais (m/s)
    theta0_rad_arr: array de ângulos em radianos
    Retorna: T_impact, x_impact (arrays)
    """
    v0_arr = np.asarray(v0_arr)
    theta0_rad_arr = np.asarray(theta0_rad_arr)
    
    n_samples = v0_arr.size
    T_impact = np.zeros(n_samples)
    x_impact = np.zeros(n_samples)
    
    dt = 0.001  # passo de integração
    max_time = 100.0  # tempo máximo de simulação
    
    for i in range(n_samples):
        v0 = v0_arr.flat[i]
        theta0 = theta0_rad_arr.flat[i]
        
        # Condições iniciais
        x, y = 0.0, 0.0
        vx = v0 * np.cos(theta0)
        vy = v0 * np.sin(theta0)
        t = 0.0
        
        # Integração Euler
        while y >= 0.0 and t < max_time:
            speed = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * speed * vx
            ay = -g - k_drag * speed * vy
            
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            t += dt
        
        T_impact[i] = t
        x_impact[i] = x
    
    return T_impact.reshape(v0_arr.shape), x_impact.reshape(v0_arr.shape)

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
# ====== MLP com integração numérica (Euler para arrasto) e escala adaptativa ======
class ImpactPINN(nn.Module):
    def __init__(self, layers, n_integration=100, lambda_y=0.45, lambda_x=0.45, lambda_vy=0.1, 
                 epsilon_vy=0.5, T_scale=50.0, X_scale=6000.0, k_drag=0.0,
                 V0_min=10.0, V0_max=250.0, theta0_min_rad=0.174, theta0_max_rad=1.396):
        super().__init__()
        dims = layers
        mods = []
        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)
        
        self.n_integration = n_integration
        self.g = g
        self.k_drag = k_drag
        self.lambda_y = lambda_y
        self.lambda_x = lambda_x
        self.lambda_vy = lambda_vy
        self.epsilon_vy = epsilon_vy
        self.T_scale = T_scale
        self.X_scale = X_scale
        
        # Ranges para desnormalização (escala adaptativa)
        self.V0_min = V0_min
        self.V0_max = V0_max
        self.theta0_min_rad = theta0_min_rad
        self.theta0_max_rad = theta0_max_rad

        # Inicialização Xavier/Glorot para estabilidade
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        """
        inputs: tensor de shape (batch, 2) com [v0_norm, theta0_norm]
        Retorna: tensor de shape (batch, 2) com [T_impact, x_impact]
        
        Usa escala adaptativa baseada na física para cada amostra.
        A rede prediz um fator de ajuste, e a escala é baseada no valor esperado (vácuo como limite superior).
        """
        raw = self.net(inputs)
        
        # Desnormaliza inputs para calcular escala adaptativa
        v0 = (inputs[:, 0:1] + 1) / 2 * (self.V0_max - self.V0_min) + self.V0_min
        theta0 = (inputs[:, 1:2] + 1) / 2 * (self.theta0_max_rad - self.theta0_min_rad) + self.theta0_min_rad
        
        # Escala esperada baseada na física (vácuo como limite superior)
        # Com arrasto, os valores reais serão menores
        T_expected = 2 * v0 * torch.sin(theta0) / self.g
        X_expected = v0**2 * torch.sin(2 * theta0) / self.g
        
        # Rede prediz fator de ajuste: sigmoid dá [0, 1], multiplicamos por 2 para [0, 2x esperado]
        # Para arrasto, o valor real será < esperado (vácuo), então [0, 2x] dá margem
        T_impact = torch.sigmoid(raw[:, 0:1]) * 2 * T_expected
        x_impact = torch.sigmoid(raw[:, 1:2]) * 2 * X_expected
        
        return torch.cat([T_impact, x_impact], dim=1)
    
    def compute_physics(self, v0, theta0_rad, T_impact):
        """
        Calcula y_impact e x_integrated usando integração numérica (Euler) com arrasto.
        
        v0: tensor (batch,) - velocidade inicial
        theta0_rad: tensor (batch,) - ângulo em radianos
        T_impact: tensor (batch, 1) - tempo de impacto predito
        
        Retorna: y_final, x_final, vy_final
        """
        batch_size = v0.shape[0]
        device = v0.device
        
        # Condições iniciais
        vx = v0 * torch.cos(theta0_rad)  # (batch,)
        vy = v0 * torch.sin(theta0_rad)  # (batch,)
        x = torch.zeros(batch_size, device=device)
        y = torch.zeros(batch_size, device=device)
        
        # Passo de tempo
        dt = T_impact.squeeze(1) / self.n_integration  # (batch,)
        
        # Integração Euler
        for _ in range(self.n_integration):
            speed = torch.sqrt(vx**2 + vy**2 + 1e-8)  # evita divisão por zero
            
            # Acelerações com arrasto
            ax = -self.k_drag * speed * vx
            ay = -self.g - self.k_drag * speed * vy
            
            # Atualiza velocidades
            vx = vx + ax * dt
            vy = vy + ay * dt
            
            # Atualiza posições
            x = x + vx * dt
            y = y + vy * dt
        
        return y, x, vy
    
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
        
        # Calcula integrais com arrasto
        y_final, x_final, vy_final = self.compute_physics(
            v0, theta0_rad, T_impact
        )
        
        # Loss 1: y(T_impact) = 0 (normalizado por X_scale²)
        loss_y = ((y_final / self.X_scale) ** 2).mean()
        
        # Loss 2: x integrado = x_impact predito (normalizado por X_scale²)
        loss_x = (((x_final - x_impact.squeeze(1)) / self.X_scale) ** 2).mean()
        
        # Loss 3: vy_impact < 0 (penaliza se vy >= 0)
        # Usamos ReLU: max(0, vy_final + epsilon_vy) para margem de segurança
        # Normalizado para escala similar (divide por velocidade típica)
        v_scale = self.X_scale / self.T_scale  # velocidade típica
        loss_vy = (torch.relu(vy_final + self.epsilon_vy) / v_scale).mean()
        
        # Perda total com pesos configuráveis
        loss = self.lambda_y * loss_y + self.lambda_x * loss_x + self.lambda_vy * loss_vy
        
        return loss, loss_y, loss_x, loss_vy

# %% Criação do modelo
default_dtype = torch.get_default_dtype()
g_t = torch.tensor(g, device=device, dtype=default_dtype)
k_drag_t = torch.tensor(k_drag, device=device, dtype=default_dtype)

theta0_min_rad = np.radians(theta0_min)
theta0_max_rad = np.radians(theta0_max)

model = ImpactPINN(layers, n_integration=N_integration, lambda_y=lambda_y, lambda_x=lambda_x, 
                   lambda_vy=lambda_vy, epsilon_vy=epsilon_vy, T_scale=T_scale, X_scale=X_scale,
                   k_drag=k_drag, V0_min=V0_min, V0_max=V0_max, 
                   theta0_min_rad=theta0_min_rad, theta0_max_rad=theta0_max_rad).to(device)

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
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_arrasto"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
images_dir = PROJECT_ROOT / "imagens" / "impact_arrasto"
images_dir.mkdir(parents=True, exist_ok=True)

# %% Checagem de checkpoint para retomada
# ====== Carregamento de checkpoint ======
previous_steps = 0
if resume_from is not None:
    resume_path = checkpoint_dir / f"{resume_from}.pth"
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        # Verifica compatibilidade
        config = checkpoint.get("config", {})
        if config.get("layers") != layers:
            print(f"AVISO: Arquitetura incompatível! Checkpoint: {config.get('layers')}, Atual: {layers}")
            print("Iniciando do zero.")
        else:
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                adam.load_state_dict(checkpoint["optimizer_state"])
            previous_steps = int(checkpoint.get("step", -1)) + 1
            # Carrega rastreamento do melhor modelo
            best_loss = float(checkpoint.get("best_loss", float('inf')))
            best_model_state = checkpoint.get("best_model_state", None)
            best_step_raw = checkpoint.get("best_step", -1)
            try:
                best_step = int(best_step_raw) if best_step_raw is not None else -1
            except (ValueError, TypeError):
                best_step = -1  # Formato antigo (ex: "LBFGS-1098"), ignora
            print(f"Checkpoint carregado: {resume_from}")
            print(f"Passos anteriores: {previous_steps}")
            if best_step >= 0:
                print(f"Melhor loss até agora: {best_loss:.6e} no passo {best_step}")
            print(f"Treinando mais {adam_steps} passos...")
    else:
        print(f"AVISO: Checkpoint '{resume_from}' não encontrado em {checkpoint_dir}")
        print("Iniciando do zero.")

total_steps = previous_steps + adam_steps

# %% Treinamento com Adam
# ====== Loop de treino (Adam) ======
model.train()
print("\n" + "="*50)
print("Iniciando treinamento com Adam...")
print(f"Parâmetros de arrasto: Cd={Cd}, k={k_drag:.6f}")
print("="*50)

for step in range(previous_steps, total_steps):
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
last_step = total_steps - 1

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
    f"impact_arrasto-{layer_str}",
    str(last_step + 1),
    f"nint{N_integration}",
    f"nsamp{N_samples}",
    f"eps{epsilon_vy:.1f}".replace(".", "p"),
    f"ly{lambda_y_str}",
    f"lx{lambda_x_str}",
    f"lvy{lambda_vy_str}",
    f"Cd{Cd:.2f}".replace(".", "p"),
]

if lbfgs_steps > 0:
    name_parts.append(f"lbfgs{lbfgs_steps}")

checkpoint_name = "-".join(name_parts)
checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pth"

checkpoint_payload = {
    "model_state": model.state_dict(),
    "best_model_state": best_model_state,
    "optimizer_state": adam.state_dict(),
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
        "Cd": Cd,
        "k_drag": k_drag,
        "m": m,
        "rho": rho,
        "A": A,
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
n_eval = 30  # menos pontos pois ground truth é lento
v0_eval = np.linspace(V0_min, V0_max, n_eval)
theta0_eval_deg = np.linspace(theta0_min, theta0_max, n_eval)
V0_grid, Theta0_grid_deg = np.meshgrid(v0_eval, theta0_eval_deg)
V0_flat = V0_grid.flatten()
Theta0_flat_rad = np.radians(Theta0_grid_deg.flatten())

# Ground truth (pode demorar um pouco)
print("\nCalculando ground truth (pode demorar)...")
T_true, X_true = impact_ground_truth(V0_flat, Theta0_flat_rad)
print("Ground truth calculado.")

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
rel_err_T = np.mean(np.abs(T_pred - T_true) / (T_true + 1e-8)) * 100
rel_err_X = np.mean(np.abs(X_pred - X_true) / (X_true + 1e-8)) * 100
print(f"Erro relativo médio T: {rel_err_T:.2f}%")
print(f"Erro relativo médio X: {rel_err_X:.2f}%")

# %% Visualizações
# ====== Visualizações ======

# Reshape para grid
T_true_grid = T_true.reshape(n_eval, n_eval)
T_pred_grid = T_pred.reshape(n_eval, n_eval)
X_true_grid = X_true.reshape(n_eval, n_eval)
X_pred_grid = X_pred.reshape(n_eval, n_eval)

# Gráfico das perdas e comparações T_impact
steps = np.arange(1, len(loss_history) + 1)

# Índices para theta = 10°, 45°, 80°
idx_10 = 0  # primeiro índice (theta_min)
idx_45 = n_eval // 2  # índice do meio
idx_80 = n_eval - 1  # último índice (theta_max)

plt.figure(figsize=(12, 8))

# Subplot 1: Evolução das perdas
plt.subplot(2, 2, 1)
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
plt.legend(fontsize=7)
plt.grid(True)

# Subplot 2: T_impact vs V0 (theta = 10°)
plt.subplot(2, 2, 2)
plt.plot(v0_eval, T_true_grid[idx_10, :], 'o-', label='T_impact GT', markersize=3)
plt.plot(v0_eval, T_pred_grid[idx_10, :], 'x--', label='T_impact PINN', markersize=3)
plt.xlabel("V0 (m/s)")
plt.ylabel("T_impact (s)")
plt.title(f"T_impact vs V0 (θ = {theta0_eval_deg[idx_10]:.0f}°)")
plt.legend()
plt.grid(True)

# Subplot 3: T_impact vs V0 (theta = 45°)
plt.subplot(2, 2, 3)
plt.plot(v0_eval, T_true_grid[idx_45, :], 'o-', label='T_impact GT', markersize=3)
plt.plot(v0_eval, T_pred_grid[idx_45, :], 'x--', label='T_impact PINN', markersize=3)
plt.xlabel("V0 (m/s)")
plt.ylabel("T_impact (s)")
plt.title(f"T_impact vs V0 (θ ≈ {theta0_eval_deg[idx_45]:.0f}°)")
plt.legend()
plt.grid(True)

# Subplot 4: T_impact vs V0 (theta = 80°)
plt.subplot(2, 2, 4)
plt.plot(v0_eval, T_true_grid[idx_80, :], 'o-', label='T_impact GT', markersize=3)
plt.plot(v0_eval, T_pred_grid[idx_80, :], 'x--', label='T_impact PINN', markersize=3)
plt.xlabel("V0 (m/s)")
plt.ylabel("T_impact (s)")
plt.title(f"T_impact vs V0 (θ = {theta0_eval_deg[idx_80]:.0f}°)")
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

T_err = np.abs(T_pred_grid - T_true_grid) / (T_true_grid + 1e-8) * 100
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

X_err = np.abs(X_pred_grid - X_true_grid) / (X_true_grid + 1e-8) * 100
im6 = axes[1, 2].contourf(V0_grid, Theta0_grid_deg, X_err, levels=20, cmap='Reds')
axes[1, 2].set_title('Erro relativo x_impact (%)')
axes[1, 2].set_xlabel('V0 (m/s)')
axes[1, 2].set_ylabel('θ0 (graus)')
plt.colorbar(im6, ax=axes[1, 2], label='%')

plt.suptitle(f'Comparação PINN vs Ground Truth (com arrasto, Cd={Cd})\nRMSE T: {rmse_T:.4f}s | RMSE X: {rmse_X:.4f}m', fontsize=12)
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


# %% [markdown]
# # PINN para balística 2D sem arrasto
#
# ---
#
# Estado `s(t) = [x(t), y(t), vx(t), vy(t)]` como funções de `t`
#
# Rede mapeia `t -> s(t) = [x(t), y(t), vx(t), vy(t)]`
#
# Usamos autodiff para obter derivadas temporais e impor a física:
# >  x'  = vx<br>
# >  y'  = vy<br>
# >  vx' = 0<br>
# >  vy' = -g<br>
#
# E condições iniciais (IC) em t=0.
# >  x(0) = 0<br>
# >  y(0) = 0<br>
# >  vx(0) = v0 \* cos( theta0 )<br>
# >  vy(0) = v0 \* sin( theta0 )<br>
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
g = 9.81                         # m/s^2
v0 = 50.0                        # m/s
theta0_deg = 63.0                # graus
theta0 = np.radians(theta0_deg)  # radianos
y0 = 0.0                         # m
x0 = 0.0                         # m

def shot_flight_time(y0, v0, theta0, g):
    tf = 1/g * (v0*np.sin(theta0) + np.sqrt(v0**2*np.sin(theta0)**2 + 2*g*y0))
    return tf

T = shot_flight_time(y0, v0, theta0, g) # tempo de voo total

# %% Parâmetros de treino
# ====== Parâmetros de treino ======

# Pontos de amostragem
adam_steps = 5000   # número de passos do Adam
lbfgs_steps = 1000  # máximo de iterações do L-BFGS (0 = disabled)
N_phys = 500        # pontos para a física
N_ic = 1            # pontos para IC (usaremos t=0)
N_data = 4          # pontos para dados de treino
noise_level = 0     # nível de ruído nos dados (0 = sem ruído)

# Rede neural
layers = [1, 20, 20, 20, 4]   # PINN: 1D (t) -> 4D (x,y,vx,vy)
lambda_phys = 0.8             # peso da perda física
lambda_data = 1 - lambda_phys # peso da perda dados (inclui IC)
lambda_dynamic = 0            # taxa de ajuste de lambda_phys (0 = disabled)
learning_rate = 1e-3          # taxa de aprendizado do Adam
resample_phys_points = True   # se True, reamostra pontos de física a cada passo
deterministic = False         # se True, torna o treinamento determinístico
seed = 42                     # seed para reprodução
eval_samples = 1000           # pontos para avaliação
eval_time_range = 1           # intervalo de tempo para avaliação (1 = 100% do tempo de voo)

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
#resume_from = "vacuo-1_64_64_64_4-3000-500-0-lamb05-resamplephys"

#%% Preparação do ground truth em torch
# Preparando constantes em torch para o ground truth
default_dtype = torch.get_default_dtype()
vx0_t = torch.tensor(v0 * np.cos(theta0), device=device, dtype=default_dtype)
vy0_t = torch.tensor(v0 * np.sin(theta0), device=device, dtype=default_dtype)
x0_t = torch.tensor(x0, device=device, dtype=default_dtype)
y0_t = torch.tensor(y0, device=device, dtype=default_dtype)
g_t = torch.tensor(g, device=device, dtype=default_dtype)
T_t = torch.tensor(T, device=device, dtype=default_dtype)

# Função de Estado - Ground Truth em torch
def shot_state_ground_truth_torch(t):
    t_clamped = torch.minimum(t, T_t) # faz projétil colidir com o solo
    x = x0_t + vx0_t * t_clamped
    y = y0_t + vy0_t * t_clamped - 0.5 * g_t * (t_clamped ** 2)

    zeros = torch.zeros_like(t_clamped) # velocidades zero após colisão com o solo
    vy_air = vy0_t - g_t * t_clamped
    vx = torch.where(t_clamped >= T_t, zeros, vx0_t)
    vy = torch.where(t_clamped >= T_t, zeros, vy_air)
    return x, y, vx, vy

# %% Definição da rede
# ====== MLP simples com tanh ======
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        dims = layers
        mods = []
        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)

        # Inicialização Xavier/Glorot para estabilidade
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)
    
    def compute_loss(self, t_phys, t_ic, t_data, lambda_phys, lambda_data, 
                     x_data_true, y_data_true, vx_data_true, vy_data_true):
        """Calcula perda total (física + IC + dados)"""
        # ---- Perda física ----
        s_phys = self(t_phys)
        x_phys, y_phys, vx_phys, vy_phys = s_phys[:,0:1], s_phys[:,1:2], s_phys[:,2:3], s_phys[:,3:4]
        
        dx_dt = grad(x_phys, t_phys)
        dy_dt = grad(y_phys, t_phys)
        dvx_dt = grad(vx_phys, t_phys)
        dvy_dt = grad(vy_phys, t_phys)
        
        # Resíduos da física
        r1 = dx_dt - vx_phys
        r2 = dy_dt - vy_phys
        r3 = dvx_dt
        r4 = dvy_dt + g
        
        loss_phys = (r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean() + r4.pow(2).mean())
        
        # ---- Condições iniciais ----
        s_ic = self(t_ic)
        x_ic, y_ic, vx_ic, vy_ic = s_ic[:,0:1], s_ic[:,1:2], s_ic[:,2:3], s_ic[:,3:4]
        
        loss_ic = ((x_ic - x0_t)**2).mean() + ((y_ic - y0_t)**2).mean() \
                  + ((vx_ic - vx0_t)**2).mean() + ((vy_ic - vy0_t)**2).mean()
        
        # ---- Perda de dados ----
        loss_data = torch.tensor(0.0, device=device)
        if N_data > 0 and t_data is not None:
            s_data = self(t_data)
            x_data, y_data, vx_data, vy_data = s_data[:,0:1], s_data[:,1:2], s_data[:,2:3], s_data[:,3:4]
            
            # Usa dados pré-gerados (com ruído já aplicado, se especificado)
            loss_data = ((x_data - x_data_true)**2).mean() + ((y_data - y_data_true)**2).mean() \
                      + ((vx_data - vx_data_true)**2).mean() + ((vy_data - vy_data_true)**2).mean()
        
        # Perda total
        loss = lambda_data * (loss_ic + loss_data) + lambda_phys * loss_phys
        
        return loss, loss_ic, loss_data, loss_phys

model = PINN(layers).to(device)

# %% Funções utilitárias
# ====== Funções utilitárias ======
def grad(outputs, inputs):
    # gera a derivada de `outputs` em relação a `inputs`
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               retain_graph=True, create_graph=True)[0]

def amostrar_pontos_fisica():
    # Gera N_phys pontos aleatórios no intervalo [0, 0.95 * T]
    t_phys = torch.rand(N_phys, 1, device=device) * 0.95 * T
    t_phys.requires_grad_(True)
    return t_phys

def amostrar_pontos_dados():
    # Gera N_data+1 pontos igualmente espaçados no intervalo [0, 0.95 * T] e remove o primeiro (CI t=0)
    t_data = torch.linspace(0, 0.95 * T, N_data + 1, device=device, dtype=default_dtype).reshape(-1, 1)
    t_data = t_data[1:]  # Remove o primeiro ponto
    t_data.requires_grad_(True)
    return t_data

# %% Amostragem de pontos e geração de dados
# ====== Amostragem de pontos ======
# Pontos de amostragem na física no intervalo [0, T]
t_phys = amostrar_pontos_fisica()
# Ponto de condição inicial (t=0)
t_ic = torch.zeros(N_ic, 1, device=device, requires_grad=True)

# Pontos de amostragem nos dados no intervalo [0, T]
if N_data > 0:
    t_data = amostrar_pontos_dados()
    with torch.no_grad():
        x_data_true, y_data_true, vx_data_true, vy_data_true = shot_state_ground_truth_torch(t_data)
        if noise_level > 0:
            # Adiciona ruído aos dados de treino
            x_data_true = x_data_true + noise_level * torch.randn_like(x_data_true)
            y_data_true = y_data_true + noise_level * torch.randn_like(y_data_true)
            vx_data_true = vx_data_true + noise_level * torch.randn_like(vx_data_true)
            vy_data_true = vy_data_true + noise_level * torch.randn_like(vy_data_true)
else:
    t_data = None
    x_data_true = None
    y_data_true = None
    vx_data_true = None
    vy_data_true = None

# %% Otimizadores
# ====== Otimizadores ======
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
loss_ic_history = []
loss_phys_history = []

# Rastreia melhor modelo
best_loss = float('inf')
best_model_state = None
best_step = -1

if lbfgs_steps > 0:
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps, line_search_fn='strong_wolfe')

# %% Checagem de checkpoint para retomada
# ====== Carregamento de checkpoint ======
checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

previous_steps = 0
if resume_from is not None:
    resume_path = checkpoint_dir / f"{resume_from}.pth"
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Verifica compatibilidade
        config = checkpoint.get("config", {})
        if config.get("layers") != layers:
            print(f"AVISO: Arquitetura incompatível! Checkpoint: {config.get('layers')}, Atual: {layers}")
            print("Iniciando do zero.")
        else:
            model.load_state_dict(checkpoint["model_state"])
            adam.load_state_dict(checkpoint["optimizer_state"])
            previous_steps = checkpoint.get("step", -1) + 1
            # Carrega rastreamento do melhor modelo
            best_loss = checkpoint.get("best_loss", float('inf'))
            best_model_state = checkpoint.get("best_model_state", None)
            best_step = checkpoint.get("best_step", -1)
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
print("="*50)
lambda_phys_initial = lambda_phys  # Salva valor inicial para nomeação do checkpoint
step = previous_steps - 1
for step in range(previous_steps, total_steps):
    adam.zero_grad()

    if resample_phys_points:
        t_phys = amostrar_pontos_fisica()

    # Calcula perda usando método do modelo
    loss, loss_ic, loss_data, loss_phys = model.compute_loss(
        t_phys, t_ic, t_data, lambda_phys, lambda_data,
        x_data_true, y_data_true, vx_data_true, vy_data_true
    )
    loss.backward()
    adam.step()

    loss_history.append(loss.item())
    loss_ic_history.append((loss_ic + loss_data).item())
    loss_phys_history.append(loss_phys.item())
    
    # Ajuste dinâmico de lambda
    if lambda_dynamic != 0:
        loss_data_val = (loss_ic + loss_data).item()
        loss_phys_val = loss_phys.item()
        
        if loss_phys_val > loss_data_val:
            # Perda física é maior, aumenta lambda_phys
            lambda_phys = min(1.0, lambda_phys + lambda_dynamic)
        else:
            # Perda de dados é maior, diminui lambda_phys
            lambda_phys = max(0.0, lambda_phys - lambda_dynamic)
        
        lambda_data = 1 - lambda_phys
    
    # Rastreia melhor modelo
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_step = step
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if step % 500 == 0:
        print(f"[Adam] step={step:04d}  loss={loss.item():.6e}  L_data={(loss_ic + loss_data).item():.3e}  L_phys={loss_phys.item():.3e}  λ_phys={lambda_phys:.3f}")

# %% (Opcional) Refinamento com L-BFGS
# ====== (Opcional) Refinamento com L-BFGS ======
lbfgs_losses = []
lbfgs_ic_losses = []
lbfgs_phys_losses = []

if lbfgs_steps > 0:
    print("\n" + "="*50)
    print(f"Iniciando refinamento com L-BFGS (max {lbfgs_steps} iterações)...")
    print("="*50)
    
    lbfgs_iter = [0]  # Contador de iterações
    best_tracking = {'loss': best_loss, 'step': best_step, 'state': best_model_state}
    
    def closure():
        lbfgs.zero_grad()
        loss, loss_ic, loss_data, loss_phys = model.compute_loss(
            t_phys, t_ic, t_data, lambda_phys, lambda_data,
            x_data_true, y_data_true, vx_data_true, vy_data_true
        )
        loss.backward()
        
        lbfgs_losses.append(loss.item())
        lbfgs_ic_losses.append((loss_ic + loss_data).item())
        lbfgs_phys_losses.append(loss_phys.item())
        lbfgs_iter[0] += 1
        
        if lbfgs_iter[0] % 100 == 0 or lbfgs_iter[0] == 1:
            print(f"[L-BFGS] iter={lbfgs_iter[0]:03d}  loss={loss.item():.6e}  L_data={(loss_ic + loss_data).item():.3e}  L_phys={loss_phys.item():.3e}")
        
        # Rastreia melhor modelo durante L-BFGS
        if loss.item() < best_tracking['loss']:
            best_tracking['loss'] = loss.item()
            best_tracking['step'] = f"LBFGS-{lbfgs_iter[0]}"
            best_tracking['state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        return loss
    
    lbfgs.step(closure)
    
    # Atualiza melhor global se L-BFGS melhorou
    best_loss = best_tracking['loss']
    best_step = best_tracking['step']
    best_model_state = best_tracking['state']
    
    print(f"L-BFGS finalizado após {lbfgs_iter[0]} iterações")
    print(f"Loss final L-BFGS: {lbfgs_losses[-1]:.6e}\n")

# %% Salvamento do checkpoint
# ====== Checkpoint ======
images_dir = PROJECT_ROOT / "imagens"
images_dir.mkdir(exist_ok=True)

# Constrói nome do checkpoint com passos totais
layer_str = "_".join(str(n) for n in layers)
# Usa lambda_phys inicial para nomeação
lambda_str = f"{lambda_phys_initial:.2f}".replace(".", "").rstrip("0")
if lambda_str == "":
    lambda_str = "0"

last_step = step if step >= 0 else -1
name_parts = [
    f"vacuo-{layer_str}",
    str(last_step + 1),  # total de passos completados
    str(N_phys),
    str(N_data),
    f"lamb{lambda_str}",
]

if lambda_dynamic != 0:
    dyn_str = f"{lambda_dynamic:.0e}".replace("e-0", "").replace("e-", "m")
    name_parts.append(f"dyn{dyn_str}")
if lbfgs_steps > 0:
    name_parts.append(f"lbfgs{lbfgs_steps}")
if resample_phys_points:
    name_parts.append("resamplephys")
if noise_level > 0:
    noise_str = f"{noise_level:.2f}".replace(".", "p")
    name_parts.append(f"noise{noise_str}")

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
        "N_phys": N_phys,
        "N_data": N_data,
        "lambda_phys": lambda_phys,
        "lambda_phys_initial": lambda_phys_initial,
        "lambda_data": lambda_data,
        "lambda_dynamic": lambda_dynamic,
        "lbfgs_steps": lbfgs_steps,
        "resample_phys_points": resample_phys_points,
        "noise_level": noise_level,
    },
}

torch.save(checkpoint_payload, checkpoint_path)
print(f"\nCheckpoint salvo em {checkpoint_path}")
if isinstance(best_step, str) and "LBFGS" in best_step:
    print(f"Melhor loss: {best_loss:.6e} (encontrado durante L-BFGS: {best_step})")
else:
    print(f"Melhor loss: {best_loss:.6e} no passo {best_step}")
if lambda_dynamic != 0:
    print(f"Lambda final: λ_phys={lambda_phys:.3f}, λ_data={lambda_data:.3f}")

print("\nTreino finalizado.")

# %% Carregamento do melhor modelo para avaliação
# ====== Usar melhor modelo ======
if best_model_state is not None:
    # Move estado do melhor modelo de volta para o dispositivo e carrega
    best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
    model.load_state_dict(best_model_state_device)
    print(f"Usando melhor modelo (passo {best_step}) para avaliação")
else:
    print("Usando modelo final para avaliação")

# %% Avaliação
# ====== Avaliação ======
with torch.no_grad():
    t_eval = torch.linspace(0, eval_time_range * T, eval_samples, device=device, dtype=default_dtype).reshape(-1, 1)
    pred = model(t_eval)
    x_pred, y_pred, vx_pred, vy_pred = pred[:,0:1], pred[:,1:2], pred[:,2:3], pred[:,3:4]

    x_true, y_true, vx_true, vy_true = shot_state_ground_truth_torch(t_eval)

rmse_x = torch.sqrt(torch.mean((x_pred - x_true)**2)).item()
rmse_y = torch.sqrt(torch.mean((y_pred - y_true)**2)).item()
rmse_vx = torch.sqrt(torch.mean((vx_pred - vx_true)**2)).item()
rmse_vy = torch.sqrt(torch.mean((vy_pred - vy_true)**2)).item()
print(
    f"RMSE x: {rmse_x:.4e} | RMSE y: {rmse_y:.4e} | RMSE vx: {rmse_vx:.4e} | RMSE vy: {rmse_vy:.4e}"
)

# %% Visualizações
# ====== Visualizações ======
with torch.no_grad():
    pred_eval = model(t_eval)
    x_pred_eval = pred_eval[:,0].cpu().numpy()
    y_pred_eval = pred_eval[:,1].cpu().numpy()
    vx_pred_eval = pred_eval[:,2].cpu().numpy()
    vy_pred_eval = pred_eval[:,3].cpu().numpy()

t_eval_np = t_eval.cpu().numpy().flatten()
with torch.no_grad():
    x_true_eval_t, y_true_eval_t, vx_true_eval_t, vy_true_eval_t = shot_state_ground_truth_torch(t_eval)

x_true_eval = x_true_eval_t.cpu().numpy().flatten()
y_true_eval = y_true_eval_t.cpu().numpy().flatten()
vx_true_eval = vx_true_eval_t.cpu().numpy().flatten()
vy_true_eval = vy_true_eval_t.cpu().numpy().flatten()

steps = np.arange(previous_steps + 1, previous_steps + len(loss_history) + 1)

# Gráfico das perdas
plt.figure(figsize=(7, 5))
plt.plot(steps, loss_history, label="Perda total (Adam)", color='tab:blue')
plt.plot(steps, loss_ic_history, label="Perda dados (Adam)", color='tab:orange')
plt.plot(steps, loss_phys_history, label="Perda física (Adam)", color='tab:green')

# Adiciona perdas do L-BFGS se disponíveis
if len(lbfgs_losses) > 0:
    # Linha vertical separando Adam e L-BFGS
    adam_end = steps[-1]
    plt.axvline(x=adam_end, color='red', linestyle='--', linewidth=1.5, label='Início L-BFGS')
    
    # Passos do L-BFGS (continua do Adam)
    lbfgs_steps = np.arange(adam_end + 1, adam_end + len(lbfgs_losses) + 1)
    plt.plot(lbfgs_steps, lbfgs_losses, label="Perda total (L-BFGS)", color='tab:blue', linestyle=':')
    plt.plot(lbfgs_steps, lbfgs_ic_losses, label="Perda dados (L-BFGS)", color='tab:orange', linestyle=':')
    plt.plot(lbfgs_steps, lbfgs_phys_losses, label="Perda física (L-BFGS)", color='tab:green', linestyle=':')

plt.yscale("log")
plt.xlabel("Iteração")
plt.ylabel("Perda")
plt.title("Evolução das perdas durante o treinamento")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_loss.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Gráfico de trajetória com vetores de velocidade
arrow_stride = max(1, eval_samples // 40)
arrow_idx = np.arange(0, len(x_pred_eval), arrow_stride)
vx_arrows = vx_pred_eval[arrow_idx]
vy_arrows = vy_pred_eval[arrow_idx]
vx_true_arrows = vx_true_eval[arrow_idx]
vy_true_arrows = vy_true_eval[arrow_idx]

speed_pred = np.sqrt(vx_arrows**2 + vy_arrows**2)
speed_true = np.sqrt(vx_true_arrows**2 + vy_true_arrows**2)
max_speed = np.max(np.concatenate([speed_pred, speed_true])) if len(speed_pred) > 0 else 0.0
scale_factor = max_speed if max_speed > 0 else 1.0

vx_scaled = vx_arrows / scale_factor
vy_scaled = vy_arrows / scale_factor
vx_true_scaled = vx_true_arrows / scale_factor
vy_true_scaled = vy_true_arrows / scale_factor

# Obtém pontos de dados para visualização (usa dados reais com ruído, se aplicado)
if N_data > 0 and t_data is not None:
    x_data_np = x_data_true.cpu().numpy().flatten()
    y_data_np = y_data_true.cpu().numpy().flatten()
    t_data_np = t_data.detach().cpu().numpy().flatten()
    vx_data_np = vx_data_true.cpu().numpy().flatten()
    vy_data_np = vy_data_true.cpu().numpy().flatten()

# Obtém ponto da condição inicial
with torch.no_grad():
    x_ic_viz, y_ic_viz, vx_ic_viz, vy_ic_viz = shot_state_ground_truth_torch(t_ic)
x_ic_np = x_ic_viz.cpu().numpy().flatten()
y_ic_np = y_ic_viz.cpu().numpy().flatten()
t_ic_np = t_ic.detach().cpu().numpy().flatten()
vx_ic_np = vx_ic_viz.cpu().numpy().flatten()
vy_ic_np = vy_ic_viz.cpu().numpy().flatten()

plt.figure(figsize=(7, 5))
plt.plot(x_pred_eval, y_pred_eval, label=" PINN", linestyle="--", color="blue")
plt.plot(x_true_eval, y_true_eval, label="Ground truth", color="orange")
plt.scatter(x_ic_np, y_ic_np, c='tab:green', s=100, zorder=6, marker='*', label="Condição inicial", edgecolors='black', linewidths=0.8)
if N_data > 0 and t_data is not None:
    plt.scatter(x_data_np, y_data_np, c='tab:green', s=50, zorder=5, label="Dados de treino", edgecolors='black', linewidths=0.5)
plt.quiver(
    x_pred_eval[arrow_idx],
    y_pred_eval[arrow_idx],
    vx_scaled,
    vy_scaled,
    angles="xy",
    scale_units="xy",
    scale=0.2,
    color="blue",
    width=0.004,
)
plt.quiver(
    x_true_eval[arrow_idx],
    y_true_eval[arrow_idx],
    vx_true_scaled,
    vy_true_scaled,
    angles="xy",
    scale_units="xy",
    scale=0.2,
    color="orange",
    width=0.004,
)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajetória")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_traj.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Gráfico de velocidades
plt.figure(figsize=(7, 5))
plt.plot(t_eval_np, vx_pred_eval, label="vx PINN", linestyle="--", color="blue")
plt.plot(t_eval_np, vx_true_eval, label="vx GT", color="orange")
plt.plot(t_eval_np, vy_pred_eval, label="vy PINN", linestyle="--", color="darkblue")
plt.plot(t_eval_np, vy_true_eval, label="vy GT", color="darkgoldenrod")
plt.scatter(t_ic_np, vx_ic_np, c='tab:green', s=100, zorder=6, marker='*', edgecolors='black', linewidths=0.8)
plt.scatter(t_ic_np, vy_ic_np, c='tab:green', s=100, zorder=6, marker='*', label="Condição inicial", edgecolors='black', linewidths=0.8)
if N_data > 0 and t_data is not None:
    plt.scatter(t_data_np, vx_data_np, c='tab:green', s=50, zorder=5, marker='o', edgecolors='black', linewidths=0.5, label="Dados de treino vx")
    plt.scatter(t_data_np, vy_data_np, c='tab:green', s=50, zorder=5, marker='o', edgecolors='black', linewidths=0.5, label="Dados de treino vy")
plt.xlabel("t (s)")
plt.ylabel("Velocidade (m/s)")
plt.title("Velocidades")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(images_dir / f"{checkpoint_name}_vel.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# %%

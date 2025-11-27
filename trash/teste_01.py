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
import matplotlib.pyplot as plt

# Ensure working directory is project root
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# %% Parâmetros físicos
# ====== Parâmetros físicos ======
g = 9.81  # m/s^2
v0 = 50.0  # m/s
theta0_deg = 63.0
theta0 = np.radians(theta0_deg)
y0 = 0.0
x0 = 0.0

def shot_flight_time(y0, v0, theta0, g):
    tf = 1/g * (v0*np.sin(theta0) + np.sqrt(v0**2*np.sin(theta0)**2 + 2*g*y0))
    return tf

T = shot_flight_time(y0, v0, theta0, g) # tempo de voo total

# %% Parâmetros de treino
# ====== Parâmetros de treino ======
# Pontos de amostragem
layers = [1, 64, 64, 64, 4]  # PINN: 1D (t) -> 4D (x,y,vx,vy)
adam_steps = 5000
N_phys = 500    # pontos para a física
N_data = 0    # pontos para dados de treino
N_ic = 1    # pontos para IC (usaremos t=0)
add_noise = False # se True, adiciona ruído aos dados de treino
noise_level = 0.01 # nível de ruído
# Rede neural
lambda_phys = 0.5
lambda_data = 1 - lambda_phys
learning_rate = 1e-3
use_lbfgs = False  # opcional: refinar com L-BFGS
resample_phys_points = False  # se True, reamostra pontos de física a cada passo
deterministic = False # se True, torna o treinamento determinístico
seed = 42
eval_samples = 200
eval_time_range = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if deterministic:
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Treinamento determinístico com seed {seed}")
else:
    print("Treinamento não determinístico")

#%% Preparação do ground truth em torch
# Preparando constantes em torch para o ground truth - uma só vez para evitar recomputação a cada iteração
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

        # Inicialização Xavier para estabilidade
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

model = PINN(layers).to(device)

# %% Funções utilitárias
# ====== Funções utilitárias ======
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               retain_graph=True, create_graph=True)[0]

def amostrar_pontos_fisica():
    t_phys = torch.rand(N_phys, 1, device=device) * T
    t_phys.requires_grad_(True)
    return t_phys

def amostrar_pontos_dados():
    t_data = torch.rand(N_data, 1, device=device) * T
    t_data.requires_grad_(True)
    return t_data
# %% Amostragem de pontos
# ====== Amostragem de pontos ======
# Pontos de amostragem na física no intervalo [0, T]
t_phys = amostrar_pontos_fisica()
# Ponto de condição inicial (t=0)
t_ic = torch.zeros(N_ic, 1, device=device, requires_grad=True)
# Pontos de amostragem nos dados no intervalo [0, T]
if N_data > 0:
    t_data = amostrar_pontos_dados()
else:
    t_data = None

# %% Otimizadores
# ====== Otimizadores ======
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
loss_ic_history = []
loss_phys_history = []

if use_lbfgs:
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=500, line_search_fn='strong_wolfe')

# %% Treinamento com Adam
# ====== Loop de treino (Adam) ======
model.train()
step = -1  # ensures availability after loop
for step in range(adam_steps):
    adam.zero_grad()

    if resample_phys_points:
        t_phys = amostrar_pontos_fisica()

    # ---- Física nos pontos de amostragem ----
    s_phys = model(t_phys) # [N_phys, 4]
    x_phys, y_phys, vx_phys, vy_phys = s_phys[:,0:1], s_phys[:,1:2], s_phys[:,2:3], s_phys[:,3:4]

    dx_dt = grad(x_phys, t_phys)
    dy_dt = grad(y_phys, t_phys)
    dvx_dt = grad(vx_phys, t_phys)
    dvy_dt = grad(vy_phys, t_phys)

    # f(s,t;phi) do sistema sem arrasto
    #   x'  - vx = 0
    #   y'  - vy = 0
    #   vx' - 0  = 0
    #   vy' + g  = 0
    r1 = dx_dt - vx_phys
    r2 = dy_dt - vy_phys
    r3 = dvx_dt
    r4 = dvy_dt + g

    loss_phys = (r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean() + r4.pow(2).mean())

    # ---- Condições iniciais ----
    s_ic = model(t_ic)                    # [N_ic, 4]
    x_ic, y_ic, vx_ic, vy_ic = s_ic[:,0:1], s_ic[:,1:2], s_ic[:,2:3], s_ic[:,3:4]

    loss_ic = ((x_ic - x0_t)**2).mean() + ((y_ic - y0_t)**2).mean() \
              + ((vx_ic - vx0_t)**2).mean() + ((vy_ic - vy0_t)**2).mean()

    # ---- Dados ----
    loss_data = torch.tensor(0.0, device=device)
    if N_data > 0:
        s_data = model(t_data)
        x_data, y_data, vx_data, vy_data = s_data[:,0:1], s_data[:,1:2], s_data[:,2:3], s_data[:,3:4]

        x_data_true, y_data_true, vx_data_true, vy_data_true = shot_state_ground_truth_torch(t_data)
        if add_noise:
            x_data = x_data + noise_level * torch.randn_like(x_data)
            y_data = y_data + noise_level * torch.randn_like(y_data)
            vx_data = vx_data + noise_level * torch.randn_like(vx_data)
            vy_data = vy_data + noise_level * torch.randn_like(vy_data)
        loss_data = ((x_data - x_data_true)**2).mean() + ((y_data - y_data_true)**2).mean() \
                  + ((vx_data - vx_data_true)**2).mean() + ((vy_data - vy_data_true)**2).mean()

    loss = lambda_data*(loss_ic + loss_data) + lambda_phys*loss_phys
    loss.backward()
    adam.step()

    loss_history.append(loss.item())
    loss_ic_history.append((loss_ic + loss_data).item())
    loss_phys_history.append(loss_phys.item())

    if step % 500 == 0:
        print(f"[Adam] step={step:04d}  loss={loss.item():.6e}  L_data={(loss_ic + loss_data).item():.3e}  L_phys={loss_phys.item():.3e}")

# %% (Opcional) Refinamento com L-BFGS
# ====== (Opcional) Refinamento com L-BFGS ======
if use_lbfgs:
    def closure():
        lbfgs.zero_grad()
        s_phys = model(t_phys)
        x_phys, y_phys, vx_phys, vy_phys = s_phys[:,0:1], s_phys[:,1:2], s_phys[:,2:3], s_phys[:,3:4]
        dx_dt = grad(x_phys, t_phys)
        dy_dt = grad(y_phys, t_phys)
        dvx_dt = grad(vx_phys, t_phys)
        dvy_dt = grad(vy_phys, t_phys)
        r1 = dx_dt - vx_phys
        r2 = dy_dt - vy_phys
        r3 = dvx_dt
        r4 = dvy_dt + g
        loss_phys = (r1.pow(2).mean() + r2.pow(2).mean() + r3.pow(2).mean() + r4.pow(2).mean())

        s_ic = model(t_ic)
        x_ic, y_ic, vx_ic, vy_ic = s_ic[:,0:1], s_ic[:,1:2], s_ic[:,2:3], s_ic[:,3:4]
        loss_ic = ((x_ic - x0_t)**2).mean() + ((y_ic - y0_t)**2).mean() \
                  + ((vx_ic - vx0_t)**2).mean() + ((vy_ic - vy0_t)**2).mean()

        # ---- Dados ----
        loss_data = torch.tensor(0.0, device=device)
        if N_data > 0 and t_data is not None:
            s_data = model(t_data)
            x_data, y_data, vx_data, vy_data = s_data[:,0:1], s_data[:,1:2], s_data[:,2:3], s_data[:,3:4]

            x_data_true, y_data_true, vx_data_true, vy_data_true = shot_state_ground_truth_torch(t_data)

            if add_noise:
                x_data_true = x_data_true + noise_level * torch.randn_like(x_data_true)
                y_data_true = y_data_true + noise_level * torch.randn_like(y_data_true)
                vx_data_true = vx_data_true + noise_level * torch.randn_like(vx_data_true)
                vy_data_true = vy_data_true + noise_level * torch.randn_like(vy_data_true)

            loss_data = ((x_data - x_data_true)**2).mean() + ((y_data - y_data_true)**2).mean() \
                        + ((vx_data - vx_data_true)**2).mean() + ((vy_data - vy_data_true)**2).mean()

        loss = lambda_data*(loss_ic + loss_data) + lambda_phys*loss_phys
        loss.backward()
        return loss

    lbfgs.step(closure)

# %% Salvamento do checkpoint
# ====== Checkpoint ======
checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

layer_str = "_".join(str(n) for n in layers)
lambda_str = f"{lambda_phys:.2f}".replace(".", "").rstrip("0")
if lambda_str == "":
    lambda_str = "0"

name_parts = [
    f"vacuo-{layer_str}",
    str(adam_steps),
    str(N_phys),
    str(N_data),
    f"lamb{lambda_str}",
]

if use_lbfgs:
    name_parts.append("lbfgs")
if resample_phys_points:
    name_parts.append("resamplephys")
if add_noise:
    noise_str = f"{noise_level:.2f}"
    name_parts.append(f"noise{noise_str}")

checkpoint_name = "-".join(name_parts)
checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pth"

last_step = step if step >= 0 else -1
checkpoint_payload = {
    "model_state": model.state_dict(),
    "optimizer_state": adam.state_dict(),
    "step": last_step,
    "config": {
        "layers": layers,
        "adam_steps": adam_steps,
        "learning_rate": learning_rate,
        "N_phys": N_phys,
        "N_data": N_data,
        "lambda_phys": lambda_phys,
        "lambda_data": lambda_data,
        "use_lbfgs": use_lbfgs,
        "resample_phys_points": resample_phys_points,
        "add_noise": add_noise,
        "noise_level": noise_level,
    },
}

torch.save(checkpoint_payload, checkpoint_path)
print(f"Checkpoint salvo em {checkpoint_path}")

print("Treino finalizado.")

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
    t_samples = t_phys.detach()
    samples_pred = model(t_samples)
    x_samples = samples_pred[:,0].cpu().numpy()
    y_samples = samples_pred[:,1].cpu().numpy()

    pred_dense = model(t_eval)
    x_pred_dense = pred_dense[:,0].cpu().numpy()
    y_pred_dense = pred_dense[:,1].cpu().numpy()
    vx_pred_dense = pred_dense[:,2].cpu().numpy()
    vy_pred_dense = pred_dense[:,3].cpu().numpy()

t_eval_np = t_eval.cpu().numpy().flatten()
with torch.no_grad():
    x_true_dense_t, y_true_dense_t, vx_true_dense_t, vy_true_dense_t = shot_state_ground_truth_torch(t_eval)

x_true_dense = x_true_dense_t.cpu().numpy().flatten()
y_true_dense = y_true_dense_t.cpu().numpy().flatten()
vx_true_dense = vx_true_dense_t.cpu().numpy().flatten()
vy_true_dense = vy_true_dense_t.cpu().numpy().flatten()

steps = np.arange(1, len(loss_history)+1)

plt.figure(figsize=(7, 5))
plt.scatter(x_samples, y_samples, s=20, alpha=0.7)
plt.title("Trajetória dos pontos amostrados")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(steps, loss_history, label="Perda total")
plt.plot(steps, loss_ic_history, label="Perda observada (IC+dados)")
plt.plot(steps, loss_phys_history, label="Perda física")
plt.yscale("log")
plt.xlabel("Época (Adam)")
plt.ylabel("Perda")
plt.title("Evolução das perdas durante o treinamento")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(x_true_dense, y_true_dense, label="Trajetória analítica")
plt.plot(x_pred_dense, y_pred_dense, label="Trajetória PINN", linestyle="--")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Comparação de trajetórias")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(t_eval_np, vx_true_dense, label="vx analítico")
plt.plot(t_eval_np, vx_pred_dense, label="vx PINN", linestyle="--")
plt.plot(t_eval_np, vy_true_dense, label="vy analítico")
plt.plot(t_eval_np, vy_pred_dense, label="vy PINN", linestyle="--")
plt.xlabel("t (s)")
plt.ylabel("Velocidade (m/s)")
plt.title("Comparação de velocidades")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

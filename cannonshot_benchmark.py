# %% [markdown]
# # Benchmark: PINN vs Integradores Numéricos
#
# ---
#
# Compara o tempo de inferência da PINN treinada contra vários algoritmos
# de integração numérica para calcular (T_impact, x_impact) dado (V0, θ0).
#
# A hipótese é que a PINN é muito mais rápida, pois a inferência é apenas
# uma série de multiplicações de matrizes (forward pass), enquanto os
# integradores precisam fazer milhares de passos iterativos.
#

# %% Configuração
# ======================================================================
# PARÂMETROS DO TESTE
# ======================================================================

# Modo: True = arrasto, False = vácuo
USE_DRAG = True

# Parâmetros de entrada para o teste
V0_teste = 120.0      # m/s
Theta0_teste = 53.0   # graus

# Número de repetições para média do tempo
N_REPETICOES = 100

# ======================================================================

# %% Importações
import time
import inspect
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from scipy.integrate import solve_ivp, odeint

# Garante que o diretório de trabalho seja a raiz do projeto
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# %% Parâmetros físicos
g = 9.81  # m/s^2

# Ranges de entrada (padrão)
V0_min, V0_max = 10.0, 250.0
theta0_min, theta0_max = 10.0, 80.0

# Parâmetros de arrasto (padrão)
Cd = 0.47
k_drag = None

device = torch.device("cpu")  # CPU para benchmark justo
default_dtype = torch.get_default_dtype()

theta0_min_rad = np.radians(theta0_min)
theta0_max_rad = np.radians(theta0_max)

# %% Definição da rede PINN
class ImpactPINN(nn.Module):
    def __init__(self, layers, n_integration=100, T_scale=50.0, X_scale=6000.0,
                 V0_min=10.0, V0_max=250.0, theta0_min_rad=0.174, theta0_max_rad=1.396):
        super().__init__()
        dims = layers
        mods = []
        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)
        
        self.g = g
        self.T_scale = T_scale
        self.X_scale = X_scale
        self.V0_min = V0_min
        self.V0_max = V0_max
        self.theta0_min_rad = theta0_min_rad
        self.theta0_max_rad = theta0_max_rad

    def forward(self, inputs):
        raw = self.net(inputs)
        
        v0 = (inputs[:, 0:1] + 1) / 2 * (self.V0_max - self.V0_min) + self.V0_min
        theta0 = (inputs[:, 1:2] + 1) / 2 * (self.theta0_max_rad - self.theta0_min_rad) + self.theta0_min_rad
        
        T_expected = 2 * v0 * torch.sin(theta0) / self.g
        X_expected = v0**2 * torch.sin(2 * theta0) / self.g
        
        T_impact = torch.sigmoid(raw[:, 0:1]) * 2 * T_expected
        x_impact = torch.sigmoid(raw[:, 1:2]) * 2 * X_expected
        
        return torch.cat([T_impact, x_impact], dim=1)

# %% Carregamento do modelo
print("=" * 70)
print("BENCHMARK: PINN vs Integradores Numéricos")
print("=" * 70)

if USE_DRAG:
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_arrasto"
    print("\n🌬️  Modo: ARRASTO")
else:
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_vacuo"
    print("\n🚀 Modo: VÁCUO")

checkpoint_files = list(checkpoint_dir.glob("*.pth"))
if not checkpoint_files:
    raise FileNotFoundError(f"Nenhum checkpoint encontrado em {checkpoint_dir}")

checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
checkpoint_path = checkpoint_files[0]

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint.get("config", {})
layers = config.get("layers", [2, 64, 64, 64, 2])

# Atualiza parâmetros do checkpoint
V0_range = config.get("V0_range", [V0_min, V0_max])
V0_min, V0_max = V0_range[0], V0_range[1]

theta0_range = config.get("theta0_range", [theta0_min, theta0_max])
theta0_min, theta0_max = theta0_range[0], theta0_range[1]
theta0_min_rad = np.radians(theta0_min)
theta0_max_rad = np.radians(theta0_max)

T_scale = 2 * V0_max * np.sin(np.radians(theta0_max)) / g
X_scale = V0_max**2 / g

if USE_DRAG:
    Cd = config.get("Cd", 0.47)
    k_drag = config.get("k_drag", None)
    
    if k_drag is None:
        m = 1.0
        rho = 1.225
        densidade_chumbo = 11340
        diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)
        A = np.pi * (diametro / 2) ** 2
        k_drag = (rho * Cd * A) / (2 * m)
    
    print(f"   Cd = {Cd}, k_drag = {k_drag:.6f}")

print(f"   Checkpoint: {checkpoint_path.name}")

model = ImpactPINN(
    layers, 
    T_scale=T_scale, 
    X_scale=X_scale,
    V0_min=V0_min, 
    V0_max=V0_max, 
    theta0_min_rad=theta0_min_rad, 
    theta0_max_rad=theta0_max_rad
).to(device)

if "best_model_state" in checkpoint and checkpoint["best_model_state"] is not None:
    model.load_state_dict(checkpoint["best_model_state"])
else:
    model.load_state_dict(checkpoint["model_state"])

model.eval()

# %% Funções auxiliares
def normalize_inputs(v0, theta_deg):
    """Normaliza inputs para [-1, 1]."""
    theta_rad = np.radians(theta_deg)
    v0_norm = 2 * (v0 - V0_min) / (V0_max - V0_min) - 1
    theta_norm = 2 * (theta_rad - theta0_min_rad) / (theta0_max_rad - theta0_min_rad) - 1
    return v0_norm, theta_norm

# %% Definição dos métodos de integração

def pinn_inference(v0, theta_deg):
    """Inferência usando a PINN treinada."""
    v0_norm, theta_norm = normalize_inputs(v0, theta_deg)
    
    with torch.no_grad():
        inputs = torch.tensor([[v0_norm, theta_norm]], device=device, dtype=default_dtype)
        outputs = model(inputs)
        T_impact = outputs[0, 0].item()
        x_impact = outputs[0, 1].item()
    
    return T_impact, x_impact


def solve_ivp_method(v0, theta_deg, method='RK45'):
    """Integração usando scipy.integrate.solve_ivp."""
    theta_rad = np.radians(theta_deg)
    
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    y0 = [0.0, 0.0, vx0, vy0]
    
    def derivatives(t, state):
        x, y, vx, vy = state
        if USE_DRAG:
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
        else:
            ax = 0
            ay = -g
        return [vx, vy, ax, ay]
    
    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    sol = solve_ivp(derivatives, [0, 200], y0, method=method, events=hit_ground)
    
    if sol.t_events[0].size > 0:
        T_impact = sol.t_events[0][0]
        x_impact = sol.y_events[0][0][0]
    else:
        T_impact = sol.t[-1]
        x_impact = sol.y[0, -1]
    
    return T_impact, x_impact


def euler_method(v0, theta_deg, dt=0.001):
    """Integração usando método de Euler (manual)."""
    theta_rad = np.radians(theta_deg)
    
    x, y = 0.0, 0.0
    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)
    t = 0.0
    
    while y >= 0 or t == 0:
        if USE_DRAG:
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
        else:
            ax = 0
            ay = -g
        
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        
        if t > 200:
            break
    
    return t, x


def rk4_method(v0, theta_deg, dt=0.01):
    """Integração usando Runge-Kutta 4ª ordem (manual)."""
    theta_rad = np.radians(theta_deg)
    
    # Estado: [x, y, vx, vy]
    state = np.array([0.0, 0.0, v0 * np.cos(theta_rad), v0 * np.sin(theta_rad)])
    t = 0.0
    
    def deriv(s):
        x, y, vx, vy = s
        if USE_DRAG:
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
        else:
            ax = 0
            ay = -g
        return np.array([vx, vy, ax, ay])
    
    while state[1] >= 0 or t == 0:
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        
        state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        
        if t > 200:
            break
    
    return t, state[0]


def analytical_vacuum(v0, theta_deg):
    """Solução analítica (SÓ VÁCUO)."""
    theta_rad = np.radians(theta_deg)
    T_impact = 2 * v0 * np.sin(theta_rad) / g
    x_impact = v0**2 * np.sin(2 * theta_rad) / g
    return T_impact, x_impact


# %% Lista de métodos a testar
methods = [
    ("PINN (rede neural)", lambda v, t: pinn_inference(v, t)),
    ("solve_ivp RK45", lambda v, t: solve_ivp_method(v, t, 'RK45')),
    ("solve_ivp RK23", lambda v, t: solve_ivp_method(v, t, 'RK23')),
    ("solve_ivp DOP853", lambda v, t: solve_ivp_method(v, t, 'DOP853')),
    ("Euler (dt=0.001)", lambda v, t: euler_method(v, t, dt=0.001)),
    ("Euler (dt=0.01)", lambda v, t: euler_method(v, t, dt=0.01)),
    ("RK4 manual (dt=0.01)", lambda v, t: rk4_method(v, t, dt=0.01)),
    ("RK4 manual (dt=0.001)", lambda v, t: rk4_method(v, t, dt=0.001)),
]

if not USE_DRAG:
    methods.append(("Analítico (fórmula)", lambda v, t: analytical_vacuum(v, t)))

# %% Warm-up: carrega tudo na memória
print("\n⏳ Fazendo warm-up (carregando tudo na memória)...")

for name, func in methods:
    try:
        _ = func(V0_teste, Theta0_teste)
        print(f"   ✓ {name}")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

print("   Warm-up completo!\n")

# %% Benchmark
print(f"🏁 Iniciando benchmark com {N_REPETICOES} repetições...")
print(f"   V0 = {V0_teste} m/s, θ0 = {Theta0_teste}°\n")

results = []

for name, func in methods:
    # Executa N repetições e mede o tempo
    times = []
    T_impact = None
    x_impact = None
    
    for i in range(N_REPETICOES):
        start = time.perf_counter()
        T_impact, x_impact = func(V0_teste, Theta0_teste)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times) * 1000  # em ms
    std_time = np.std(times) * 1000
    
    results.append({
        'name': name,
        'T_impact': T_impact,
        'x_impact': x_impact,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'times': times
    })

# %% Resultados
print("=" * 100)
print("RESULTADOS")
print("=" * 100)

# Encontra o tempo de referência (PINN)
pinn_time = results[0]['avg_time_ms']

print(f"\n{'Método':<25} │ {'T_impact (s)':>12} {'x_impact (m)':>14} │ {'Tempo (ms)':>12} {'± σ':>10} │ {'Speedup':>10}")
print("-" * 100)

for r in results:
    speedup = r['avg_time_ms'] / pinn_time if r['name'] != "PINN (rede neural)" else 1.0
    speedup_str = f"{speedup:.1f}x" if r['name'] != "PINN (rede neural)" else "REF"
    
    print(f"{r['name']:<25} │ {r['T_impact']:12.4f} {r['x_impact']:14.2f} │ "
          f"{r['avg_time_ms']:12.4f} {r['std_time_ms']:10.4f} │ {speedup_str:>10}")

print("-" * 100)

# %% Análise
print("\n📊 ANÁLISE:")

# Encontra o mais rápido
fastest = min(results, key=lambda x: x['avg_time_ms'])
slowest = max(results, key=lambda x: x['avg_time_ms'])

print(f"\n   🥇 Mais rápido: {fastest['name']} ({fastest['avg_time_ms']:.4f} ms)")
print(f"   🐢 Mais lento:  {slowest['name']} ({slowest['avg_time_ms']:.4f} ms)")
print(f"   📈 Diferença:   {slowest['avg_time_ms'] / fastest['avg_time_ms']:.1f}x")

# Compara PINN com integradores
pinn_result = results[0]
integrators = [r for r in results if "PINN" not in r['name'] and "Analítico" not in r['name']]

if integrators:
    fastest_integrator = min(integrators, key=lambda x: x['avg_time_ms'])
    slowest_integrator = max(integrators, key=lambda x: x['avg_time_ms'])
    
    print(f"\n   🔬 PINN vs Integradores:")
    print(f"      vs mais rápido ({fastest_integrator['name']}): "
          f"{fastest_integrator['avg_time_ms'] / pinn_result['avg_time_ms']:.1f}x mais lento")
    print(f"      vs mais lento ({slowest_integrator['name']}): "
          f"{slowest_integrator['avg_time_ms'] / pinn_result['avg_time_ms']:.1f}x mais lento")

# Verifica precisão
print(f"\n   🎯 Precisão (comparando com solve_ivp RK45 como referência):")
ref = next(r for r in results if "RK45" in r['name'])

for r in results:
    if r['name'] != ref['name']:
        erro_T = abs(r['T_impact'] - ref['T_impact']) / ref['T_impact'] * 100
        erro_x = abs(r['x_impact'] - ref['x_impact']) / ref['x_impact'] * 100
        print(f"      {r['name']:<25}: ΔT = {erro_T:.3f}%, Δx = {erro_x:.3f}%")

# %% Conclusão
print("\n" + "=" * 100)
print("CONCLUSÃO")
print("=" * 100)

if pinn_result['avg_time_ms'] < fastest_integrator['avg_time_ms']:
    speedup = fastest_integrator['avg_time_ms'] / pinn_result['avg_time_ms']
    print(f"""
🏆 A PINN É MAIS RÁPIDA!

   A rede neural é {speedup:.1f}x mais rápida que o integrador mais rápido.
   
   Isso faz sentido porque:
   - A inferência da PINN é apenas uma série de multiplicações de matrizes
   - Os integradores precisam fazer centenas/milhares de passos iterativos
   
   Para aplicações em tempo real ou otimização (onde milhares de avaliações
   são necessárias), a PINN oferece uma vantagem significativa!
""")
else:
    print(f"""
🤔 Resultado surpreendente!

   O integrador {fastest_integrator['name']} foi mais rápido que a PINN.
   Isso pode acontecer para redes muito grandes ou problemas muito simples.
""")

# %%


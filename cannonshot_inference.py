# %% [markdown]
# # Inferência do Modelo de Balística Inversa
#
# Carrega um checkpoint treinado e compara:
# - Predições da rede neural (instantâneas)
# - Solução numérica com scipy (iterativa)
#
# Gera tabelas comparativas e visualização de trajetórias.

# %% Importações
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from tabulate import tabulate

# %% Configuração
# ======================================================================
# CHECKPOINT
# ======================================================================
CHECKPOINT_FILE = "inverse_pinn_latest.pth"

# ======================================================================
# ALVOS E ÂNGULOS PARA TESTE
# ======================================================================
X_IMPACTS = [400, 800, 1200, 1500]  # metros
THETAS = [15.0, 45.0, 65.0, 75.0]       # graus

# ======================================================================
# CORES PARA CADA ALVO (degradê será gerado automaticamente)
# ======================================================================
TARGET_COLORS = [
    'Blues',      # Azul
    'Reds',       # Vermelho
    'Greens',     # Verde
    'Purples',    # Roxo
    'Oranges',    # Laranja
    'YlOrBr',     # Amarelo-Marrom
    'PuRd',       # Rosa
    'BuGn',       # Azul-Verde
]

# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
device = torch.device("cpu")  # Inferência em CPU é mais rápida para batches pequenos

print(f"Device: {device}")

# %% Carrega checkpoint e extrai configuração
checkpoint_path = PROJECT_ROOT / "checkpoints" / "inverse_pinn" / CHECKPOINT_FILE
print(f"Carregando checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']

# Extrai parâmetros físicos e de normalização
X_IMPACT_MIN = config['X_IMPACT_MIN']
X_IMPACT_MAX = config['X_IMPACT_MAX']
V0_MIN = config['V0_MIN']
V0_MAX = config['V0_MAX']
THETA_MIN = config['THETA_MIN']
THETA_MAX = config['THETA_MAX']
k_drag = config['k_drag']
HIDDEN_LAYERS = config['hidden_layers']

# Constantes físicas
g = 9.81

print(f"Configuração do modelo:")
print(f"  X_IMPACT: [{X_IMPACT_MIN:.1f}, {X_IMPACT_MAX:.1f}] m")
print(f"  V0: [{V0_MIN:.1f}, {V0_MAX:.1f}] m/s")
print(f"  THETA: [{THETA_MIN:.1f}, {THETA_MAX:.1f}]°")
print(f"  k_drag: {k_drag:.6f}")
print(f"  Arquitetura: {HIDDEN_LAYERS}")

# %% Funções de normalização (DEVEM ser idênticas ao treino!)
def normalize_x(x):
    """Normaliza x_impact para [-1, 1] (igual ao treino!)."""
    return 2 * (x - X_IMPACT_MIN) / (X_IMPACT_MAX - X_IMPACT_MIN) - 1

def normalize_theta(theta_deg):
    """Normaliza θ para [0, 1]."""
    return (theta_deg - THETA_MIN) / (THETA_MAX - THETA_MIN)

# %% Definição da Rede Neural (DEVE ser idêntica ao treino!)
class InverseBallisticPINN(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64]):
        super().__init__()
        
        self.V0_min = V0_MIN
        self.V0_max = V0_MAX
        self.theta_min_rad = np.radians(THETA_MIN)
        self.theta_max_rad = np.radians(THETA_MAX)
        
        # IMPORTANTE: T_max deve ser IDÊNTICO ao treino!
        self.T_max = 2 * V0_MAX / g * 1.5  # margem de segurança (igual ao treino)
        
        # Build layers dynamically
        layers = []
        in_features = 2  # x_impact_norm, theta_norm
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.Tanh())
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, 2))  # Output: V0, T
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_impact_norm, theta_norm):
        # Stack inputs
        inputs = torch.stack([x_impact_norm, theta_norm], dim=1)
        
        # Forward
        raw = self.net(inputs)
        
        # V0: softplus → [V0_min, +∞)
        v0 = self.V0_min + torch.nn.functional.softplus(raw[:, 0])
        
        # T: softplus para garantir T > 0, escalado
        T = torch.nn.functional.softplus(raw[:, 1]) * self.T_max / 5 + 0.2
        
        # Converte theta_norm para radianos
        theta_rad = theta_norm * (self.theta_max_rad - self.theta_min_rad) + self.theta_min_rad
        
        return v0, theta_rad, T

# %% Instancia e carrega modelo
model = InverseBallisticPINN(hidden_layers=HIDDEN_LAYERS).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"\n✅ Modelo carregado com {sum(p.numel() for p in model.parameters())} parâmetros")
print(f"   Melhor loss do treino: {checkpoint.get('best_loss', 'N/A'):.6f}")

# %% Solver numérico (scipy) para comparação
def solve_ballistic_scipy(x_target, theta_deg, tol=1e-6, max_iter=50):
    """
    Resolve o problema balístico inverso usando scipy.
    Encontra V0 tal que o projétil atinja x_target com ângulo theta_deg.
    
    Returns:
        v0, T, tempo_processamento, sucesso
    """
    theta_rad = np.radians(theta_deg)
    
    def dynamics(t, state):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        return [vx, vy, -k_drag * v_mag * vx, -g - k_drag * v_mag * vy]
    
    def hit_ground(t, state):
        return state[1]  # y = 0
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    def shoot(v0):
        """Dispara com v0 e retorna x no impacto."""
        vx0 = v0 * np.cos(theta_rad)
        vy0 = v0 * np.sin(theta_rad)
        t_max = 3 * v0 * np.sin(theta_rad) / g + 1
        
        sol = solve_ivp(dynamics, [0, t_max], [0, 0, vx0, vy0],
                        events=hit_ground, dense_output=True, max_step=0.1)
        
        if sol.t_events[0].size > 0:
            t_impact = sol.t_events[0][0]
            x_impact = sol.sol(t_impact)[0]
            return x_impact, t_impact, sol
        return sol.y[0, -1], sol.t[-1], sol
    
    start_time = time.perf_counter()
    
    # Busca binária para encontrar V0
    v0_low, v0_high = V0_MIN, V0_MAX * 2
    
    # Verifica se a solução existe no range
    x_low, _, _ = shoot(v0_low)
    x_high, _, _ = shoot(v0_high)
    
    if x_target < x_low or x_target > x_high:
        elapsed = time.perf_counter() - start_time
        return None, None, elapsed * 1000, False
    
    # Brentq para encontrar raiz
    try:
        def error(v0):
            x_impact, _, _ = shoot(v0)
            return x_impact - x_target
        
        v0_solution = brentq(error, v0_low, v0_high, xtol=tol)
        _, T_solution, _ = shoot(v0_solution)
        
        elapsed = time.perf_counter() - start_time
        return v0_solution, T_solution, elapsed * 1000, True
    except:
        elapsed = time.perf_counter() - start_time
        return None, None, elapsed * 1000, False

# %% Simula trajetória para plotagem
def simulate_trajectory(v0, theta_rad, T):
    """Simula trajetória completa usando scipy."""
    def dynamics(t, state):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        return [vx, vy, -k_drag * v_mag * vx, -g - k_drag * v_mag * vy]
    
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    
    t_eval = np.linspace(0, T, 200)
    sol = solve_ivp(dynamics, [0, T], [0, 0, vx0, vy0], t_eval=t_eval, method='RK45')
    
    return sol.y[0], sol.y[1], sol.y[2], sol.y[3]

# %% Executa inferência
print("\n" + "="*100)
print("COMPARAÇÃO: REDE NEURAL vs SCIPY SOLVER")
print("="*100)

results = []

for x_target in X_IMPACTS:
    for theta_deg in THETAS:
        # --- Rede Neural ---
        start_nn = time.perf_counter()
        with torch.no_grad():
            x_norm = torch.tensor([normalize_x(x_target)], device=device, dtype=torch.float32)
            theta_norm = torch.tensor([normalize_theta(theta_deg)], device=device, dtype=torch.float32)
            v0_nn, theta_rad_nn, T_nn = model(x_norm, theta_norm)
            v0_nn = v0_nn.item()
            theta_rad_nn = theta_rad_nn.item()
            T_nn = T_nn.item()
        time_nn = (time.perf_counter() - start_nn) * 1000  # ms
        
        # Simula para verificar
        x_traj, y_traj, vx_traj, vy_traj = simulate_trajectory(v0_nn, theta_rad_nn, T_nn)
        x_final_nn = x_traj[-1]
        y_final_nn = y_traj[-1]
        
        # --- Scipy Solver ---
        v0_scipy, T_scipy, time_scipy, success_scipy = solve_ballistic_scipy(x_target, theta_deg)
        
        if success_scipy:
            x_traj_sp, y_traj_sp, _, _ = simulate_trajectory(v0_scipy, np.radians(theta_deg), T_scipy)
            x_final_scipy = x_traj_sp[-1]
        else:
            x_final_scipy = None
        
        # Erro
        erro_nn = abs(x_final_nn - x_target) / x_target * 100 if x_target > 0 else 0
        
        results.append({
            'x_target': x_target,
            'theta': theta_deg,
            'v0_nn': v0_nn,
            'T_nn': T_nn,
            'x_final_nn': x_final_nn,
            'y_final_nn': y_final_nn,
            'erro_nn': erro_nn,
            'time_nn': time_nn,
            'v0_scipy': v0_scipy,
            'T_scipy': T_scipy,
            'x_final_scipy': x_final_scipy,
            'time_scipy': time_scipy,
            'success_scipy': success_scipy
        })

# %% Tabela de resultados
print("\n" + "-"*100)
print("RESULTADOS DETALHADOS")
print("-"*100)

table_data = []
for r in results:
    v0_scipy_str = f"{r['v0_scipy']:.1f}" if r['v0_scipy'] else "N/A"
    T_scipy_str = f"{r['T_scipy']:.2f}" if r['T_scipy'] else "N/A"
    
    status = "✓" if r['erro_nn'] < 1.0 else "✗"
    if r['v0_nn'] > V0_MAX:
        status = "⚠ SAT"  # Saturado (V0 > V0_MAX)
    
    table_data.append([
        r['x_target'],
        r['theta'],
        f"{r['v0_nn']:.1f}",
        f"{r['T_nn']:.2f}",
        f"{r['x_final_nn']:.1f}",
        f"{r['erro_nn']:.2f}%",
        f"{r['time_nn']:.3f}",
        v0_scipy_str,
        T_scipy_str,
        f"{r['time_scipy']:.1f}",
        status
    ])

headers = ['x_alvo', 'θ(°)', 'V0_NN', 'T_NN', 'x_sim', 'Erro', 't_NN(ms)', 
           'V0_scipy', 'T_scipy', 't_scipy(ms)', 'Status']

print(tabulate(table_data, headers=headers, tablefmt='simple', stralign='right'))

# %% Estatísticas
print("\n" + "-"*100)
print("ESTATÍSTICAS")
print("-"*100)

erros = [r['erro_nn'] for r in results]
times_nn = [r['time_nn'] for r in results]
times_scipy = [r['time_scipy'] for r in results]
saturados = sum(1 for r in results if r['v0_nn'] > V0_MAX)

print(f"Erro médio NN:     {np.mean(erros):.3f}%")
print(f"Erro máximo NN:    {np.max(erros):.3f}%")
print(f"Casos < 1% erro:   {sum(1 for e in erros if e < 1.0)}/{len(erros)}")
print(f"Casos saturados:   {saturados}/{len(results)} (V0 > {V0_MAX} m/s)")
print(f"Tempo médio NN:    {np.mean(times_nn):.3f} ms")
print(f"Tempo médio scipy: {np.mean(times_scipy):.1f} ms")
print(f"Speedup:           {np.mean(times_scipy)/np.mean(times_nn):.0f}x")

# %% Visualização - Todas as trajetórias
print("\n" + "="*100)
print("GERANDO VISUALIZAÇÃO")
print("="*100)

fig, ax = plt.subplots(figsize=(16, 10))

# Para cada alvo, usa um colormap diferente
n_thetas = len(THETAS)

for i, x_target in enumerate(X_IMPACTS):
    # Seleciona colormap para este alvo
    cmap_name = TARGET_COLORS[i % len(TARGET_COLORS)]
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Plota alvo
    ax.scatter([x_target], [0], c=[cmap(0.7)], s=150, marker='o', 
               zorder=10, edgecolors='black', linewidths=1.5)
    
    # Trajetórias para cada ângulo
    for j, theta_deg in enumerate(THETAS):
        # Encontra resultado correspondente
        r = next(r for r in results if r['x_target'] == x_target and r['theta'] == theta_deg)
        
        # Cor no degradê (mais escuro para ângulos menores)
        color_intensity = 0.3 + 0.6 * (j / (n_thetas - 1)) if n_thetas > 1 else 0.6
        color = cmap(color_intensity)
        
        # Simula trajetória
        x_traj, y_traj, _, _ = simulate_trajectory(r['v0_nn'], np.radians(theta_deg), r['T_nn'])
        
        # Estilo baseado no erro
        linestyle = '-' if r['erro_nn'] < 1.0 else '--'
        alpha = 0.9 if r['erro_nn'] < 1.0 else 0.5
        linewidth = 1.5 if r['erro_nn'] < 1.0 else 1.0
        
        # Label apenas para o primeiro ângulo de cada alvo
        label = f'Alvo {x_target}m' if j == 0 else None
        
        ax.plot(x_traj, y_traj, linestyle=linestyle, color=color, 
                alpha=alpha, linewidth=linewidth, label=label)

# Chão
ax.axhline(y=0, color='saddlebrown', linewidth=3, alpha=0.7, zorder=1)

# Canhão
ax.scatter([0], [0], c='black', s=300, marker='^', zorder=15, label='Canhão')

# Configurações do gráfico
ax.set_xlabel('Distância (m)', fontsize=12)
ax.set_ylabel('Altura (m)', fontsize=12)
ax.set_title(f'Trajetórias Balísticas - Modelo Neural\n'
             f'Alvos: {X_IMPACTS} | Ângulos: {THETAS}°', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-50, max(X_IMPACTS) * 1.1)
ax.set_ylim(-50, None)

plt.tight_layout()

# Salva figura
images_dir = PROJECT_ROOT / "imagens" / "inverse_pinn"
images_dir.mkdir(parents=True, exist_ok=True)
output_file = images_dir / "inferencia_trajetorias.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")
print(f"\n✅ Figura salva em: {output_file}")

plt.show()

# %% Gráfico de erro por alvo/ângulo
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Erro por alvo
for x_target in X_IMPACTS:
    erros_alvo = [r['erro_nn'] for r in results if r['x_target'] == x_target]
    ax1.plot(THETAS, erros_alvo, 'o-', label=f'{x_target}m', linewidth=2, markersize=8)

ax1.set_xlabel('Ângulo (°)', fontsize=12)
ax1.set_ylabel('Erro em x (%)', fontsize=12)
ax1.set_title('Erro por Ângulo para cada Alvo', fontsize=14)
ax1.legend(title='Alvo')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1% erro')

# V0 predito vs theta
for x_target in X_IMPACTS:
    v0s = [r['v0_nn'] for r in results if r['x_target'] == x_target]
    ax2.plot(THETAS, v0s, 'o-', label=f'{x_target}m', linewidth=2, markersize=8)

ax2.axhline(y=V0_MAX, color='red', linestyle='--', linewidth=2, label=f'V0_MAX ({V0_MAX} m/s)')
ax2.set_xlabel('Ângulo (°)', fontsize=12)
ax2.set_ylabel('V₀ predito (m/s)', fontsize=12)
ax2.set_title('Velocidade Inicial Predita', fontsize=14)
ax2.legend(title='Alvo')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(images_dir / "inferencia_analise.png", dpi=200, bbox_inches="tight")
print(f"✅ Análise salva em: {images_dir / 'inferencia_analise.png'}")
plt.show()

print("\n" + "="*100)
print("INFERÊNCIA CONCLUÍDA")
print("="*100)


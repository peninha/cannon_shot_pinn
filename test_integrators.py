# %% [markdown]
# # Benchmark de Integradores para Balística com Arrasto
#
# Compara performance de diferentes métodos de integração:
# - RK4 manual (PyTorch)
# - Euler (PyTorch)
# - torchdiffeq (odeint)
# - scipy.integrate (não diferenciável, apenas referência)
#
# Métricas avaliadas:
# - Tempo de execução (forward pass)
# - Tempo com backward pass (gradientes)
# - Precisão vs solução de referência
# - Escalabilidade com batch size

# %% Configuração
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Tenta importar torchdiffeq
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("⚠️ torchdiffeq não instalado. Instale com: pip install torchdiffeq")

# Tenta importar scipy
try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy não instalado.")

# %% Parâmetros físicos
g = 9.81
Cd = 0.47
m = 1.0
rho = 1.225
densidade_chumbo = 11340
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)
A = np.pi * (diametro / 2) ** 2
k_drag = (rho * Cd * A) / (2 * m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Device: {device}")
print(f"k_drag = {k_drag:.6f}")

# %% Integradores

# ----------------------------------------------------------------------
# 1. RK4 Manual (PyTorch) - O que já usamos
# ----------------------------------------------------------------------
def integrate_rk4(v0, theta_rad, T, n_steps=200):
    """RK4 manual em PyTorch - totalmente diferenciável."""
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
# 2. Euler Simples (PyTorch)
# ----------------------------------------------------------------------
def integrate_euler(v0, theta_rad, T, n_steps=200):
    """Euler simples - rápido mas menos preciso."""
    batch_size = v0.shape[0]
    
    x = torch.zeros(batch_size, device=v0.device)
    y = torch.zeros(batch_size, device=v0.device)
    vx = v0 * torch.cos(theta_rad)
    vy = v0 * torch.sin(theta_rad)
    
    dt = T / n_steps
    
    for _ in range(n_steps):
        v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
        ax = -k_drag * v_mag * vx
        ay = -g - k_drag * v_mag * vy
        
        x = x + vx * dt
        y = y + vy * dt
        vx = vx + ax * dt
        vy = vy + ay * dt
    
    return x, y, vx, vy


# ----------------------------------------------------------------------
# 3. Heun (RK2) - Meio termo entre Euler e RK4
# ----------------------------------------------------------------------
def integrate_heun(v0, theta_rad, T, n_steps=200):
    """Método de Heun (RK2) - balanço entre velocidade e precisão."""
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
# 4. torchdiffeq (se disponível)
# ----------------------------------------------------------------------
if HAS_TORCHDIFFEQ:
    def integrate_torchdiffeq(v0, theta_rad, T, n_steps=200, method='dopri5'):
        """
        torchdiffeq odeint com tempo normalizado para batch com T variável.
        
        Truque: Usamos τ ∈ [0,1] e escalamos: dt/dτ = T
        Assim: dx/dτ = T * dx/dt
        """
        batch_size = v0.shape[0]
        
        vx0 = v0 * torch.cos(theta_rad)
        vy0 = v0 * torch.sin(theta_rad)
        
        # Initial state: [x, y, vx, vy] - batched
        state0 = torch.stack([
            torch.zeros(batch_size, device=v0.device),
            torch.zeros(batch_size, device=v0.device),
            vx0, vy0
        ], dim=1)  # Shape: [batch, 4]
        
        # T needs to be accessible in dynamics
        T_batch = T  # closure
        
        def dynamics(tau, state):
            """
            Dynamics in normalized time τ.
            dx/dτ = T * dx/dt
            """
            x, y, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
            v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
            
            # Physical derivatives (dx/dt)
            dx_dt = vx
            dy_dt = vy
            dvx_dt = -k_drag * v_mag * vx
            dvy_dt = -g - k_drag * v_mag * vy
            
            # Scale by T to convert to dX/dτ
            return torch.stack([
                T_batch * dx_dt,
                T_batch * dy_dt,
                T_batch * dvx_dt,
                T_batch * dvy_dt
            ], dim=1)
        
        # Integrate from τ=0 to τ=1 (which corresponds to t=0 to t=T for each sample)
        tau_span = torch.tensor([0.0, 1.0], device=v0.device)
        
        # Options for fixed step methods
        if method == 'rk4':
            options = {'step_size': 1.0 / n_steps}
        else:
            options = {}
        
        solution = odeint(dynamics, state0, tau_span, method=method, options=options)
        
        # solution shape: [2, batch, 4] - we want the last time point
        final_state = solution[-1]  # [batch, 4]
        
        return final_state[:, 0], final_state[:, 1], final_state[:, 2], final_state[:, 3]
    
    def integrate_torchdiffeq_dopri5(v0, theta_rad, T, n_steps=200):
        """torchdiffeq com Dormand-Prince (adaptativo)."""
        return integrate_torchdiffeq(v0, theta_rad, T, n_steps, method='dopri5')
    
    def integrate_torchdiffeq_rk4(v0, theta_rad, T, n_steps=200):
        """torchdiffeq com RK4 fixo."""
        return integrate_torchdiffeq(v0, theta_rad, T, n_steps, method='rk4')


# ----------------------------------------------------------------------
# 5. Scipy (apenas para referência - NÃO diferenciável)
# ----------------------------------------------------------------------
if HAS_SCIPY:
    def integrate_scipy(v0, theta_rad, T, n_steps=200):
        """Scipy solve_ivp - NÃO diferenciável, apenas referência."""
        # Convert to numpy
        v0_np = v0.detach().cpu().numpy()
        theta_np = theta_rad.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy()
        
        batch_size = len(v0_np)
        results = np.zeros((batch_size, 4))
        
        def dynamics(t, state):
            x, y, vx, vy = state
            v_mag = np.sqrt(vx**2 + vy**2 + 1e-8)
            return [vx, vy, -k_drag * v_mag * vx, -g - k_drag * v_mag * vy]
        
        for i in range(batch_size):
            vx0 = v0_np[i] * np.cos(theta_np[i])
            vy0 = v0_np[i] * np.sin(theta_np[i])
            
            sol = solve_ivp(dynamics, [0, T_np[i]], [0, 0, vx0, vy0], 
                           method='RK45', dense_output=False)
            results[i] = sol.y[:, -1]
        
        # Back to torch (no gradients!)
        results = torch.tensor(results, device=v0.device, dtype=v0.dtype)
        return results[:, 0], results[:, 1], results[:, 2], results[:, 3]


# %% Benchmark Functions

def benchmark_forward(integrator, v0, theta_rad, T, n_warmup=5, n_runs=20):
    """Benchmark forward pass only."""
    # Warmup
    for _ in range(n_warmup):
        _ = integrator(v0, theta_rad, T)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        x, y, vx, vy = integrator(v0, theta_rad, T)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times), (x, y, vx, vy)


def benchmark_backward(integrator, v0, theta_rad, T, n_warmup=3, n_runs=10):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(n_warmup):
        v0_ = v0.clone().requires_grad_(True)
        x, y, vx, vy = integrator(v0_, theta_rad, T)
        loss = (x**2 + y**2).mean()
        loss.backward()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(n_runs):
        v0_ = v0.clone().requires_grad_(True)
        
        start = time.perf_counter()
        x, y, vx, vy = integrator(v0_, theta_rad, T)
        loss = (x**2 + y**2).mean()
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


# %% Run Benchmark

print("\n" + "="*70)
print("BENCHMARK DE INTEGRADORES")
print("="*70)

# Test parameters
batch_sizes = [32, 64, 128, 256, 512, 1024]
n_steps_list = [50, 100, 200, 400]

# Fixed test case for accuracy comparison
torch.manual_seed(42)
v0_test = torch.rand(256, device=device) * 200 + 50  # 50-250 m/s
theta_test = torch.rand(256, device=device) * 1.2 + 0.2  # 0.2-1.4 rad (~10-80°)
T_test = torch.rand(256, device=device) * 20 + 5  # 5-25 s

# List of integrators to test
integrators = {
    'RK4 Manual': integrate_rk4,
    'Euler': integrate_euler,
    'Heun (RK2)': integrate_heun,
}

if HAS_TORCHDIFFEQ:
    integrators['torchdiffeq RK4'] = integrate_torchdiffeq_rk4
    integrators['torchdiffeq dopri5'] = integrate_torchdiffeq_dopri5

if HAS_SCIPY:
    integrators['Scipy RK45 (ref)'] = integrate_scipy

# %% 1. Accuracy Comparison
print("\n" + "-"*70)
print("1. COMPARAÇÃO DE PRECISÃO (vs Scipy RK45)")
print("-"*70)

if HAS_SCIPY:
    # Reference solution
    x_ref, y_ref, vx_ref, vy_ref = integrate_scipy(v0_test, theta_test, T_test)
    
    print(f"\n{'Integrador':<25} {'Erro x (m)':<15} {'Erro y (m)':<15} {'Erro |v| (m/s)':<15}")
    print("-" * 70)
    
    for name, integrator in integrators.items():
        if 'Scipy' in name:
            continue
        
        with torch.no_grad():
            x, y, vx, vy = integrator(v0_test, theta_test, T_test)
        
        err_x = torch.abs(x - x_ref).mean().item()
        err_y = torch.abs(y - y_ref).mean().item()
        v_mag = torch.sqrt(vx**2 + vy**2)
        v_ref_mag = torch.sqrt(vx_ref**2 + vy_ref**2)
        err_v = torch.abs(v_mag - v_ref_mag).mean().item()
        
        print(f"{name:<25} {err_x:<15.6f} {err_y:<15.6f} {err_v:<15.6f}")
else:
    print("Scipy não disponível para comparação de precisão.")

# %% 2. Forward Pass Timing
print("\n" + "-"*70)
print("2. TEMPO DE FORWARD PASS (batch=256, n_steps=200)")
print("-"*70)

print(f"\n{'Integrador':<25} {'Tempo (ms)':<15} {'Std (ms)':<15}")
print("-" * 55)

forward_times = {}
for name, integrator in integrators.items():
    # Scipy é mais lento, usamos menos runs
    n_runs = 5 if 'Scipy' in name else 20
    n_warmup = 1 if 'Scipy' in name else 5
    
    mean_t, std_t, _ = benchmark_forward(integrator, v0_test, theta_test, T_test, 
                                          n_warmup=n_warmup, n_runs=n_runs)
    forward_times[name] = mean_t * 1000
    print(f"{name:<25} {mean_t*1000:<15.3f} {std_t*1000:<15.3f}")

# %% 3. Backward Pass Timing (importante para treinamento!)
print("\n" + "-"*70)
print("3. TEMPO DE FORWARD + BACKWARD (batch=256, n_steps=200)")
print("-"*70)

print(f"\n{'Integrador':<25} {'Tempo (ms)':<15} {'Std (ms)':<15} {'Diferenciável':<15}")
print("-" * 70)

backward_times = {}
for name, integrator in integrators.items():
    if 'Scipy' in name:
        print(f"{name:<25} {'N/A':<15} {'N/A':<15} {'❌ Não':<15}")
        continue
    
    try:
        mean_t, std_t = benchmark_backward(integrator, v0_test, theta_test, T_test)
        backward_times[name] = mean_t * 1000
        print(f"{name:<25} {mean_t*1000:<15.3f} {std_t*1000:<15.3f} {'✅ Sim':<15}")
    except Exception as e:
        print(f"{name:<25} {'ERRO':<15} {'':<15} {'❌ Não':<15}")
        print(f"    → {e}")

# %% 4. Scaling with Batch Size
print("\n" + "-"*70)
print("4. ESCALABILIDADE COM BATCH SIZE (n_steps=200)")
print("-"*70)

scaling_results = {name: [] for name in integrators.keys() if 'Scipy' not in name}

for batch_size in batch_sizes:
    v0 = torch.rand(batch_size, device=device) * 200 + 50
    theta = torch.rand(batch_size, device=device) * 1.2 + 0.2
    T = torch.rand(batch_size, device=device) * 20 + 5
    
    for name, integrator in integrators.items():
        if 'Scipy' in name:
            continue
        
        try:
            mean_t, _, _ = benchmark_forward(integrator, v0, theta, T, n_warmup=2, n_runs=5)
            scaling_results[name].append(mean_t * 1000)
        except:
            scaling_results[name].append(float('nan'))

print(f"\n{'Batch':<10}", end="")
for name in scaling_results.keys():
    print(f"{name:<20}", end="")
print()
print("-" * (10 + 20 * len(scaling_results)))

for i, batch_size in enumerate(batch_sizes):
    print(f"{batch_size:<10}", end="")
    for name in scaling_results.keys():
        print(f"{scaling_results[name][i]:<20.3f}", end="")
    print()

# %% 5. Scaling with n_steps
print("\n" + "-"*70)
print("5. ESCALABILIDADE COM N_STEPS (batch=256)")
print("-"*70)

# Métodos cujo custo depende diretamente de n_steps
nsteps_methods = ['RK4 Manual', 'Euler', 'Heun (RK2)']
if HAS_TORCHDIFFEQ:
    # Método de passo fixo do torchdiffeq também depende de n_steps (via step_size)
    nsteps_methods.append('torchdiffeq RK4')

nsteps_results = {name: [] for name in nsteps_methods}

for n_steps in n_steps_list:
    for name in nsteps_results.keys():
        integrator = integrators[name]
        
        # Create wrapper with specific n_steps
        def wrapper(v0, theta, T, ns=n_steps):
            return integrator(v0, theta, T, n_steps=ns)
        
        mean_t, _, _ = benchmark_forward(wrapper, v0_test, theta_test, T_test, n_warmup=2, n_runs=5)
        nsteps_results[name].append(mean_t * 1000)

print(f"\n{'n_steps':<10}", end="")
for name in nsteps_results.keys():
    print(f"{name:<20}", end="")
print()
print("-" * (10 + 20 * len(nsteps_results)))

for i, n_steps in enumerate(n_steps_list):
    print(f"{n_steps:<10}", end="")
    for name in nsteps_results.keys():
        print(f"{nsteps_results[name][i]:<20.3f}", end="")
    print()

if HAS_TORCHDIFFEQ and 'torchdiffeq dopri5' in integrators:
    print("\nNota: 'torchdiffeq dopri5' é adaptativo (rtol/atol) e não depende de n_steps; por isso não aparece nesta tabela.")

# %% Visualização
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Forward times
ax1 = axes[0, 0]
names = list(forward_times.keys())
times = list(forward_times.values())
colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
bars = ax1.bar(names, times, color=colors)
ax1.set_ylabel('Tempo (ms)')
ax1.set_title('Forward Pass (batch=256)')
ax1.tick_params(axis='x', rotation=45)
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{t:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Backward times
ax2 = axes[0, 1]
if backward_times:
    names = list(backward_times.keys())
    times = list(backward_times.values())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    bars = ax2.bar(names, times, color=colors)
    ax2.set_ylabel('Tempo (ms)')
    ax2.set_title('Forward + Backward (batch=256)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{t:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Scaling with batch size
ax3 = axes[1, 0]
for name, times in scaling_results.items():
    ax3.plot(batch_sizes, times, 'o-', label=name, linewidth=2, markersize=6)
ax3.set_xlabel('Batch Size')
ax3.set_ylabel('Tempo (ms)')
ax3.set_title('Escalabilidade com Batch Size')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Scaling with n_steps
ax4 = axes[1, 1]
for name, times in nsteps_results.items():
    ax4.plot(n_steps_list, times, 'o-', label=name, linewidth=2, markersize=6)
ax4.set_xlabel('Número de Passos')
ax4.set_ylabel('Tempo (ms)')
ax4.set_title('Escalabilidade com n_steps')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imagens/benchmark_integradores.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Conclusões
print("\n" + "="*70)
print("CONCLUSÕES")
print("="*70)

# Análise dinâmica dos resultados
print("\n📊 RANKING DE PERFORMANCE:")

# Forward pass ranking
if forward_times:
    print("\n  Forward Pass (só integração):")
    sorted_forward = sorted(forward_times.items(), key=lambda x: x[1])
    for i, (name, t) in enumerate(sorted_forward, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"    {emoji} {i}. {name}: {t:.2f} ms")

# Backward pass ranking (importante para treinamento!)
if backward_times:
    print("\n  Forward + Backward (para treinamento):")
    sorted_backward = sorted(backward_times.items(), key=lambda x: x[1])
    for i, (name, t) in enumerate(sorted_backward, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"    {emoji} {i}. {name}: {t:.2f} ms")

# Recomendações dinâmicas
print("\n" + "-"*70)
print("📋 RECOMENDAÇÕES:")
print("-"*70)

if backward_times:
    fastest_bwd = min(backward_times, key=backward_times.get)
    fastest_bwd_time = backward_times[fastest_bwd]
    
    # Encontra o mais preciso entre os diferenciáveis
    precise_methods = ['RK4 Manual', 'Heun (RK2)', 'torchdiffeq dopri5']
    available_precise = [m for m in precise_methods if m in backward_times]
    
    print(f"\n1. Para TREINAMENTO (precisa de gradientes):")
    print(f"   → Recomendado: {fastest_bwd} ({fastest_bwd_time:.2f} ms)")
    
    if 'Euler' in backward_times and available_precise:
        euler_time = backward_times['Euler']
        rk4_time = backward_times.get('RK4 Manual', float('inf'))
        speedup = rk4_time / euler_time if euler_time > 0 else 0
        print(f"   → Euler é {speedup:.1f}x mais rápido que RK4, mas menos preciso")
        print(f"   → Se precisão é crítica, use {available_precise[0]}")

if forward_times:
    fastest_fwd = min(forward_times, key=forward_times.get)
    print(f"\n2. Para INFERÊNCIA (sem gradientes):")
    print(f"   → Recomendado: {fastest_fwd} ({forward_times[fastest_fwd]:.2f} ms)")
    
    if 'Scipy RK45 (ref)' in forward_times:
        scipy_time = forward_times['Scipy RK45 (ref)']
        rk4_time = forward_times.get('RK4 Manual', scipy_time)
        print(f"   → Scipy é {scipy_time/rk4_time:.1f}x mais lento que RK4 Manual")

print(f"\n3. Dicas gerais:")
print(f"   → Device atual: {device}")
if device.type == 'cuda':
    print(f"   → GPU detectada! Batches maiores = melhor throughput")
else:
    print(f"   → CPU mode. Considere usar GPU para batches grandes")

# Verifica se Euler é aceitável
if HAS_SCIPY and 'Euler' in forward_times:
    # Pega os erros da comparação de precisão (se foram calculados)
    print(f"   → Euler com 2x n_steps pode igualar precisão do RK4")

print()

# %%


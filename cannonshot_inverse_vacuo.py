# %% [markdown]
# # Problema Inverso: Dado um alvo x_impact, encontrar (V0, θ0, T_impact)
#
# ---
#
# Usa a PINN treinada como modelo diferenciável e otimiza (V0, θ0) para atingir o alvo.
#
# Suporta dois modos:
# - Vácuo: soluções analíticas conhecidas (θ=45° minimiza V0)
# - Arrasto: soluções numéricas por integração (θ ótimo < 45°)
#

# %% Configuração
# ======================================================================
# MUDE AQUI PARA ALTERNAR ENTRE VÁCUO E ARRASTO
# ======================================================================
USE_DRAG = False  # True = arrasto, False = vácuo
# ======================================================================

# %% Importações
import inspect
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Garante que o diretório de trabalho seja a raiz do projeto
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path(inspect.getfile(inspect.currentframe())).resolve()
PROJECT_ROOT = CURRENT_FILE.parent
os.chdir(PROJECT_ROOT)

# %% Parâmetros físicos (devem ser iguais ao treinamento)
g = 9.81  # m/s^2

# Ranges de entrada
V0_min, V0_max = 10.0, 250.0        # m/s
theta0_min, theta0_max = 10.0, 80.0  # graus

# Parâmetros de arrasto (só usados se USE_DRAG=True)
m = 1.0                          # kg - massa do projétil
rho = 1.225                      # kg/m^3 - densidade do ar
densidade_chumbo = 11340         # kg/m^3 - densidade do chumbo
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)  # m
A = np.pi * (diametro / 2) ** 2  # m^2 - área de seção reta
Cd = 0.47                        # coeficiente de arrasto (esfera)
k_drag = (rho * Cd * A) / (2 * m)  # constante de arrasto

# Escalas esperadas
T_scale = 2 * V0_max * np.sin(np.radians(theta0_max)) / g
X_scale = V0_max**2 / g

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
default_dtype = torch.get_default_dtype()

theta0_min_rad = np.radians(theta0_min)
theta0_max_rad = np.radians(theta0_max)

# Mensagem de modo
if USE_DRAG:
    print(f"🌬️  Modo: ARRASTO (k_drag = {k_drag:.6f}, Cd = {Cd})")
else:
    print("🚀 Modo: VÁCUO (sem arrasto)")

# %% Definição da rede (mesma arquitetura do treinamento)
class ImpactPINN(nn.Module):
    def __init__(self, layers, n_integration=100, lambda_y=0.1, lambda_x=0.1, lambda_vy=0.8, 
                 epsilon_vy=1.5, T_scale=50.0, X_scale=6000.0,
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
        self.lambda_y = lambda_y
        self.lambda_x = lambda_x
        self.lambda_vy = lambda_vy
        self.epsilon_vy = epsilon_vy
        self.T_scale = T_scale
        self.X_scale = X_scale
        
        self.V0_min = V0_min
        self.V0_max = V0_max
        self.theta0_min_rad = theta0_min_rad
        self.theta0_max_rad = theta0_max_rad

    def forward(self, inputs):
        raw = self.net(inputs)
        
        # Desnormaliza inputs para calcular escala adaptativa
        v0 = (inputs[:, 0:1] + 1) / 2 * (self.V0_max - self.V0_min) + self.V0_min
        theta0 = (inputs[:, 1:2] + 1) / 2 * (self.theta0_max_rad - self.theta0_min_rad) + self.theta0_min_rad
        
        # Escala esperada baseada na física (vácuo)
        T_expected = 2 * v0 * torch.sin(theta0) / self.g
        X_expected = v0**2 * torch.sin(2 * theta0) / self.g
        
        T_impact = torch.sigmoid(raw[:, 0:1]) * 2 * T_expected
        x_impact = torch.sigmoid(raw[:, 1:2]) * 2 * X_expected
        
        return torch.cat([T_impact, x_impact], dim=1)

# %% Carregamento do modelo
if USE_DRAG:
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_arrasto"
else:
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "impact_vacuo"

# Encontra o checkpoint mais recente
checkpoint_files = list(checkpoint_dir.glob("*.pth"))
if not checkpoint_files:
    raise FileNotFoundError(f"Nenhum checkpoint encontrado em {checkpoint_dir}")

# Ordena por data de modificação (mais recente primeiro)
checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
checkpoint_path = checkpoint_files[0]
print(f"Carregando checkpoint: {checkpoint_path.name}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint.get("config", {})
layers = config.get("layers", [2, 64, 64, 64, 2])

model = ImpactPINN(
    layers, 
    n_integration=config.get("N_integration", 100),
    T_scale=T_scale, 
    X_scale=X_scale,
    V0_min=V0_min, 
    V0_max=V0_max, 
    theta0_min_rad=theta0_min_rad, 
    theta0_max_rad=theta0_max_rad
).to(device)

# Usa o melhor modelo se disponível
if "best_model_state" in checkpoint and checkpoint["best_model_state"] is not None:
    model.load_state_dict(checkpoint["best_model_state"])
    print("Usando melhor modelo do checkpoint")
else:
    model.load_state_dict(checkpoint["model_state"])
    print("Usando modelo final do checkpoint")

model.eval()

# %% Funções auxiliares
from scipy.optimize import brentq, minimize_scalar

def normalize_inputs_np(v0, theta0_rad):
    """Normaliza inputs para [-1, 1] (numpy)."""
    v0_norm = 2 * (v0 - V0_min) / (V0_max - V0_min) - 1
    theta0_norm = 2 * (theta0_rad - theta0_min_rad) / (theta0_max_rad - theta0_min_rad) - 1
    return v0_norm, theta0_norm

def denormalize_inputs_np(v0_norm, theta0_norm):
    """Desnormaliza inputs de [-1, 1] (numpy)."""
    v0 = (v0_norm + 1) / 2 * (V0_max - V0_min) + V0_min
    theta0_rad = (theta0_norm + 1) / 2 * (theta0_max_rad - theta0_min_rad) + theta0_min_rad
    return v0, theta0_rad

def pinn_predict(v0, theta_deg):
    """
    Usa a PINN para predizer (T_impact, x_impact) dado (V0, θ).
    
    Esta é a função que vamos usar como "caixa preta" para encontrar raízes.
    """
    theta_rad = np.radians(theta_deg)
    v0_norm, theta_norm = normalize_inputs_np(np.array([v0]), np.array([theta_rad]))
    
    with torch.no_grad():
        inputs = torch.tensor([[v0_norm[0], theta_norm[0]]], device=device, dtype=default_dtype)
        outputs = model(inputs)
        T_impact = outputs[0, 0].item()
        x_impact = outputs[0, 1].item()
    
    return T_impact, x_impact

def compute_x_real(v0, theta_deg):
    """
    Calcula o x_impact REAL (ground truth) para dados V0 e θ.
    
    - Vácuo: fórmula analítica
    - Arrasto: simulação numérica
    """
    if USE_DRAG:
        _, x_real = simulate_trajectory(v0, theta_deg, use_drag=True)
    else:
        theta_rad = np.radians(theta_deg)
        x_real = v0**2 * np.sin(2 * theta_rad) / g
    return x_real

# %% Ground Truth numérico (para arrasto)
def simulate_trajectory(v0, theta_deg, use_drag=USE_DRAG, dt=0.001, max_time=100):
    """
    Simula a trajetória do projétil usando integração numérica (Euler).
    
    Funciona tanto para vácuo quanto para arrasto.
    
    Returns: T_impact, x_impact, y_final (para verificação)
    """
    theta_rad = np.radians(theta_deg)
    
    # Condições iniciais
    x, y = 0.0, 0.0
    vx = v0 * np.cos(theta_rad)
    vy = v0 * np.sin(theta_rad)
    
    t = 0.0
    
    while t < max_time:
        # Verifica se atingiu o solo
        if y < 0 and t > 0:
            # Interpolação para encontrar o momento exato do impacto
            # y_prev + vy_prev * dt_final = 0
            # dt_final = -y_prev / vy
            # Voltamos um passo e refinamos
            break
        
        # Salva estado anterior
        x_prev, y_prev = x, y
        vx_prev, vy_prev = vx, vy
        
        if use_drag:
            # Com arrasto: F_drag = -k * |v| * v
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
        else:
            # Vácuo
            ax = 0
            ay = -g
        
        # Euler
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
    
    # Refinamento: interpolação linear para encontrar T exato
    if y_prev > 0 and vy_prev < 0:
        dt_final = -y_prev / vy_prev
        T_impact = t - dt + dt_final
        x_impact = x_prev + vx_prev * dt_final
    else:
        T_impact = t
        x_impact = x
    
    return T_impact, x_impact

def ground_truth_given_theta(x_target, theta_deg, use_drag=USE_DRAG):
    """
    Encontra V0 para atingir x_target com θ fixo (numérico).
    
    Usa busca binária (bisseção) para encontrar V0.
    """
    def f(v0):
        _, x = simulate_trajectory(v0, theta_deg, use_drag)
        return x - x_target
    
    # Verifica se há solução no intervalo
    f_min = f(V0_min)
    f_max = f(V0_max)
    
    if f_min * f_max > 0:
        return None  # Sem solução no intervalo
    
    try:
        v0_solution = brentq(f, V0_min, V0_max, xtol=1e-4)
        T_impact, x_achieved = simulate_trajectory(v0_solution, theta_deg, use_drag)
        
        return {
            'v0': v0_solution,
            'theta0_deg': theta_deg,
            'T_impact': T_impact,
            'x_target': x_target,
            'x_achieved': x_achieved
        }
    except ValueError:
        return None

def ground_truth_min_v0(x_target, n_angles=30, use_drag=USE_DRAG):
    """
    Encontra a solução que minimiza V0 para atingir x_target (numérico).
    
    Testa vários ângulos e retorna o que dá menor V0.
    Com arrasto, o ângulo ótimo é < 45°.
    """
    best_solution = None
    best_v0 = float('inf')
    
    theta_range = np.linspace(theta0_min + 1, theta0_max - 1, n_angles)
    
    for theta_deg in theta_range:
        sol = ground_truth_given_theta(x_target, theta_deg, use_drag)
        if sol and sol['v0'] < best_v0:
            best_v0 = sol['v0']
            best_solution = sol
    
    return best_solution

def ground_truth_all_solutions(x_target, n_points=20, use_drag=USE_DRAG):
    """
    Retorna todas as soluções numéricas para x_target.
    """
    solutions = []
    
    for theta_deg in np.linspace(theta0_min, theta0_max, n_points):
        sol = ground_truth_given_theta(x_target, theta_deg, use_drag)
        if sol:
            solutions.append(sol)
    
    return solutions

# %% Soluções analíticas (Ground Truth para VÁCUO)
def analytic_min_v0(x_target):
    """
    Solução analítica para MINIMIZAR V0 dado x_target (SÓ VÁCUO).
    
    O ângulo que minimiza V0 é SEMPRE 45°, pois:
    - x = v0² × sin(2θ) / g
    - v0 = sqrt(x × g / sin(2θ))
    - v0 mínimo quando sin(2θ) máximo → θ = 45°
    """
    if USE_DRAG:
        # Com arrasto, usa solução numérica
        return ground_truth_min_v0(x_target)
    
    theta_rad = np.pi / 4  # 45°
    v0 = np.sqrt(x_target * g)
    T = 2 * v0 * np.sin(theta_rad) / g
    
    return {
        'v0': v0,
        'theta0_deg': 45.0,
        'T_impact': T,
        'x_target': x_target
    }

def analytic_given_v0(x_target, v0):
    """
    Solução analítica dado x_target e V0 fixo (SÓ VÁCUO).
    """
    if USE_DRAG:
        # Com arrasto, não temos solução analítica
        return None
    
    sin_2theta = x_target * g / (v0**2)
    
    if sin_2theta > 1:
        return None
    
    theta_low_rad = np.arcsin(sin_2theta) / 2
    theta_high_rad = np.pi/2 - theta_low_rad
    
    solutions = []
    for theta_rad in [theta_low_rad, theta_high_rad]:
        theta_deg = np.degrees(theta_rad)
        if theta0_min <= theta_deg <= theta0_max:
            T = 2 * v0 * np.sin(theta_rad) / g
            solutions.append({
                'v0': v0,
                'theta0_deg': theta_deg,
                'T_impact': T,
                'x_target': x_target
            })
    
    return solutions if solutions else None

def analytic_given_theta(x_target, theta_deg):
    """
    Solução dado x_target e θ fixo.
    
    Usa fórmula analítica para vácuo, numérica para arrasto.
    """
    if USE_DRAG:
        return ground_truth_given_theta(x_target, theta_deg)
    
    theta_rad = np.radians(theta_deg)
    sin_2theta = np.sin(2 * theta_rad)
    
    if sin_2theta <= 0:
        return None
    
    v0 = np.sqrt(x_target * g / sin_2theta)
    
    if v0 < V0_min or v0 > V0_max:
        return None
    
    T = 2 * v0 * np.sin(theta_rad) / g
    
    return {
        'v0': v0,
        'theta0_deg': theta_deg,
        'T_impact': T,
        'x_target': x_target
    }

def analytic_all_solutions(x_target, n_points=20):
    """
    Retorna todas as soluções para x_target.
    """
    if USE_DRAG:
        return ground_truth_all_solutions(x_target, n_points)
    
    solutions = []
    for theta_deg in np.linspace(theta0_min, theta0_max, n_points):
        sol = analytic_given_theta(x_target, theta_deg)
        if sol:
            solutions.append(sol)
    
    return solutions

# %% Classe para resolver o problema inverso usando a PINN
class InverseProblemSolver:
    """
    Resolve o problema inverso: dado x_target, encontra (V0, θ0, T_impact).
    
    Usa método de BRENT (encontrar raízes) com a PINN como função.
    Muito mais simples e robusto que otimização por gradiente!
    """
    
    def __init__(self, model):
        self.model = model
    
    def solve_given_theta(self, x_target, theta_deg):
        """
        Encontra V0 para atingir x_target com ângulo fixo.
        
        Usa método de Brent para encontrar a raiz de:
            f(V0) = PINN_x(V0, θ) - x_target = 0
        """
        # Função para encontrar raiz
        def f(v0):
            _, x = pinn_predict(v0, theta_deg)
            return x - x_target
        
        # Verifica se há raiz no intervalo [V0_min, V0_max]
        f_min = f(V0_min)
        f_max = f(V0_max)
        
        # Se não há mudança de sinal, não há raiz no intervalo
        if f_min * f_max > 0:
            return None  # Impossível atingir x_target com esse ângulo
        
        try:
            # Método de Brent: encontra raiz garantidamente
            v0_solution = brentq(f, V0_min, V0_max, xtol=1e-6)
            
            # Avalia solução
            T_impact, x_achieved = pinn_predict(v0_solution, theta_deg)
            
            # Calcula x_real (ground truth) para verificação
            x_real = compute_x_real(v0_solution, theta_deg)
            
            return {
                'v0': v0_solution,
                'theta0_deg': theta_deg,
                'T_impact': T_impact,
                'x_achieved': x_achieved,  # O que a PINN acha
                'x_real': x_real,           # O que realmente acontece
                'error': abs(x_achieved - x_target),
                'rel_error_pct': abs(x_achieved - x_target) / x_target * 100
            }
        except ValueError:
            return None
    
    def solve_given_v0(self, x_target, v0):
        """
        Encontra θ para atingir x_target com V0 fixo.
        
        Usa método de Brent. Podem existir 2 soluções (ângulo baixo e alto).
        """
        def f(theta_deg):
            _, x = pinn_predict(v0, theta_deg)
            return x - x_target
        
        solutions = []
        
        # Procura raiz no intervalo baixo (theta_min até 45°)
        try:
            f_low = f(theta0_min)
            f_mid = f(45.0)
            
            if f_low * f_mid < 0:
                theta_solution = brentq(f, theta0_min, 45.0, xtol=1e-4)
                T, x = pinn_predict(v0, theta_solution)
                x_real = compute_x_real(v0, theta_solution)
                solutions.append({
                    'v0': v0,
                    'theta0_deg': theta_solution,
                    'T_impact': T,
                    'x_achieved': x,
                    'x_real': x_real,
                    'error': abs(x - x_target),
                    'rel_error_pct': abs(x - x_target) / x_target * 100
                })
        except ValueError:
            pass
        
        # Procura raiz no intervalo alto (45° até theta_max)
        try:
            f_mid = f(45.0)
            f_high = f(theta0_max)
            
            if f_mid * f_high < 0:
                theta_solution = brentq(f, 45.0, theta0_max, xtol=1e-4)
                T, x = pinn_predict(v0, theta_solution)
                x_real = compute_x_real(v0, theta_solution)
                solutions.append({
                    'v0': v0,
                    'theta0_deg': theta_solution,
                    'T_impact': T,
                    'x_achieved': x,
                    'x_real': x_real,
                    'error': abs(x - x_target),
                    'rel_error_pct': abs(x - x_target) / x_target * 100
                })
        except ValueError:
            pass
        
        return solutions if solutions else None
    
    def solve_min_v0(self, x_target, n_angles=30):
        """
        Encontra a solução que minimiza V0 para atingir x_target.
        
        Estratégia: testa vários ângulos e escolhe o com menor V0.
        """
        best_solution = None
        best_v0 = float('inf')
        
        theta_range = np.linspace(theta0_min + 1, theta0_max - 1, n_angles)
        
        for theta_deg in theta_range:
            sol = self.solve_given_theta(x_target, theta_deg)
            
            if sol and sol['v0'] < best_v0:
                best_v0 = sol['v0']
                best_solution = sol
        
        return best_solution
    
    def find_all_solutions(self, x_target, n_thetas=15, verbose=True):
        """
        Encontra soluções para vários ângulos, mostrando a curva completa.
        """
        solutions = []
        theta_range = np.linspace(theta0_min, theta0_max, n_thetas)
        
        if verbose:
            print(f"\nBuscando soluções para x_target = {x_target:.1f} m")
            print("-" * 90)
            print(f"{'θ (°)':>8} {'V0 (m/s)':>10} {'T (s)':>8} {'x_PINN':>10} {'x_real':>10} {'Erro real%':>10}")
            print("-" * 90)
        
        for theta_deg in theta_range:
            sol = self.solve_given_theta(x_target, theta_deg)
            
            if sol:
                solutions.append(sol)
                if verbose:
                    erro_real = abs(sol['x_real'] - x_target) / x_target * 100
                    print(f"{theta_deg:8.1f} {sol['v0']:10.1f} {sol['T_impact']:8.2f} "
                          f"{sol['x_achieved']:10.1f} {sol['x_real']:10.1f} {erro_real:10.2f}")
        
        return solutions

# %% Instancia o solver
solver = InverseProblemSolver(model)

# %% Testes: Minimizar V0
print("\n" + "="*100)
print("PROBLEMA INVERSO: Dado x_target, encontrar (V0, θ0, T_impact)")
print("="*100)

print("\n⚠️  IMPORTANTE: A qualidade do solver depende da qualidade da PINN!")
print("    Se a PINN foi mal treinada (erro alto), as soluções inversas serão incorretas.")
print("    A coluna 'x_real' mostra o alcance ANALÍTICO com os parâmetros encontrados.")
print("    Isso expõe se a PINN está errada.")

x_targets = [500, 1000, 2000, 3000, 5000]

if USE_DRAG:
    print("\n📍 Objetivo: Minimizar V0 (com arrasto, θ ótimo < 45°)")
else:
    print("\n📍 Objetivo: Minimizar V0 (vácuo, sempre θ = 45°)")
print("-" * 120)
print(f"{'x_target':>8} │ {'V0':>8} {'θ':>7} │ {'x_PINN':>8} {'x_real':>8} │ {'V0 Anal.':>8} {'θ Anal.':>7} │ {'Status':>15}")
print("-" * 120)

for x_target in x_targets:
    sol_pinn = solver.solve_min_v0(x_target, n_angles=20)
    sol_ana = analytic_min_v0(x_target)
    
    if sol_pinn:
        # x_PINN: o que a PINN acha que vai atingir
        x_pinn = sol_pinn['x_achieved']
        
        # x_real: o que realmente vai atingir (ground truth)
        x_real = compute_x_real(sol_pinn['v0'], sol_pinn['theta0_deg'])
        
        erro_pinn = abs(x_pinn - x_target) / x_target * 100
        erro_real = abs(x_real - x_target) / x_target * 100
        
        if erro_real < 2:
            status = "✓ OK"
        else:
            status = f"✗ Real:{erro_real:.1f}%"
        
        print(f"{x_target:8.0f} │ {sol_pinn['v0']:8.2f} {sol_pinn['theta0_deg']:7.2f}° │ "
              f"{x_pinn:8.1f} {x_real:8.1f} │ "
              f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}° │ {status:>15}")
    else:
        print(f"{x_target:8.0f} │ {'Não encontrado':>18} │ {'---':>8} {'---':>8} │ "
              f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}° │")

# Verifica se algum V0 analítico está fora do range
print("\n⚠️  Nota: V0_min = {:.0f} m/s, V0_max = {:.0f} m/s".format(V0_min, V0_max))
print("    x_min com θ=45° = V0_min² / g = {:.0f} m".format(V0_min**2 / g))
print("    x_max com θ=45° = V0_max² / g = {:.0f} m".format(V0_max**2 / g))

# %% Testes: Ângulo fixo
print("\n📍 Objetivo: Ângulo fixo de 45°")
print("-" * 120)
print(f"{'x_target':>8} │ {'V0':>8} {'θ':>7} │ {'x_PINN':>8} {'x_real':>8} │ {'V0 Anal.':>8} {'θ Anal.':>7} │ {'Status':>15}")
print("-" * 120)

for x_target in x_targets:
    sol_pinn = solver.solve_given_theta(x_target, 45.0)
    sol_ana = analytic_given_theta(x_target, 45.0)
    
    if sol_pinn and sol_ana:
        x_pinn = sol_pinn['x_achieved']
        x_real = compute_x_real(sol_pinn['v0'], sol_pinn['theta0_deg'])
        
        erro_pinn = abs(x_pinn - x_target) / x_target * 100
        erro_real = abs(x_real - x_target) / x_target * 100
        
        if erro_real < 2:
            status = "✓ OK"
        else:
            status = f"✗ Real:{erro_real:.1f}%"
        
        print(f"{x_target:8.0f} │ {sol_pinn['v0']:8.2f} {sol_pinn['theta0_deg']:7.2f}° │ "
              f"{x_pinn:8.1f} {x_real:8.1f} │ "
              f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}° │ {status:>15}")
    elif sol_pinn:
        x_pinn = sol_pinn['x_achieved']
        x_real = compute_x_real(sol_pinn['v0'], sol_pinn['theta0_deg'])
        print(f"{x_target:8.0f} │ {sol_pinn['v0']:8.2f} {sol_pinn['theta0_deg']:7.2f}° │ "
              f"{x_pinn:8.1f} {x_real:8.1f} │ {'Fora do range':>16} │")
    else:
        status_ana = f"{sol_ana['v0']:.2f}" if sol_ana else "Impossível"
        print(f"{x_target:8.0f} │ {'Não encontrado':>18} │ {'---':>8} {'---':>8} │ {status_ana:>16} │")

# %% Testes: Todas as soluções para x_target fixo
print("\n📍 Todas as soluções para x_target = 2000m (PINN vs Analítico)")
print("-" * 70)

sol_pinn = solver.find_all_solutions(2000, n_thetas=15, verbose=False)
sol_ana = analytic_all_solutions(2000, n_points=15)

print(f"{'θ (°)':>8} │ {'V0 PINN':>10} {'T PINN':>8} │ {'V0 Anal.':>10} {'T Anal.':>8} │ {'Erro V0 %':>10}")
print("-" * 70)

for sp, sa in zip(sol_pinn, sol_ana):
    erro_v0 = abs(sp['v0'] - sa['v0']) / sa['v0'] * 100
    print(f"{sp['theta0_deg']:8.1f} │ {sp['v0']:10.1f} {sp['T_impact']:8.2f} │ "
          f"{sa['v0']:10.1f} {sa['T_impact']:8.2f} │ {erro_v0:10.2f}")

# %% Visualização (RÁPIDA - usa dados analíticos + poucos pontos PINN)
print("\n📊 Gerando visualização...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Cores para diferentes alvos
x_targets_viz = [500, 1000, 2000, 3000, 4000]
colors = plt.cm.viridis(np.linspace(0, 1, len(x_targets_viz)))

# PRÉ-CALCULA dados PINN uma só vez (5 alvos × 5 ângulos = 25 chamadas apenas)
print("  Calculando pontos PINN...")
pinn_data = {}
for x_target in x_targets_viz:
    sols = solver.find_all_solutions(x_target, n_thetas=5, verbose=False)
    pinn_data[x_target] = sols

# Subplot 1: V0 vs θ
ax1 = axes[0, 0]
for x_target, color in zip(x_targets_viz, colors):
    # Curva analítica (instantâneo)
    sols_ana = analytic_all_solutions(x_target, n_points=50)
    if sols_ana:
        ax1.plot([s['theta0_deg'] for s in sols_ana], [s['v0'] for s in sols_ana], 
                 '-', color=color, alpha=0.7, linewidth=2)
    
    # Pontos PINN (já calculados)
    sols_pinn = pinn_data[x_target]
    if sols_pinn:
        ax1.scatter([s['theta0_deg'] for s in sols_pinn], [s['v0'] for s in sols_pinn], 
                   c=[color], s=50, marker='o', edgecolors='black', linewidths=0.5, 
                   label=f'x={x_target}m')

ax1.set_xlabel('θ0 (graus)')
ax1.set_ylabel('V0 (m/s)')
gt_label = "numérico" if USE_DRAG else "analítico"
ax1.set_title(f'V0 vs θ para atingir diferentes alvos\n(linhas: {gt_label}, pontos: PINN)')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Subplot 2: T_impact vs θ
ax2 = axes[0, 1]
for x_target, color in zip(x_targets_viz, colors):
    sols_ana = analytic_all_solutions(x_target, n_points=50)
    if sols_ana:
        ax2.plot([s['theta0_deg'] for s in sols_ana], [s['T_impact'] for s in sols_ana],
                 '-', color=color, alpha=0.7, linewidth=2)
    
    sols_pinn = pinn_data[x_target]
    if sols_pinn:
        ax2.scatter([s['theta0_deg'] for s in sols_pinn], [s['T_impact'] for s in sols_pinn],
                   c=[color], s=50, marker='o', edgecolors='black', linewidths=0.5,
                   label=f'x={x_target}m')

ax2.set_xlabel('θ0 (graus)')
ax2.set_ylabel('T_impact (s)')
ax2.set_title(f'Tempo de voo vs θ para diferentes alvos\n(linhas: {gt_label}, pontos: PINN)')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Subplot 3: V0 mínimo vs x_target
ax3 = axes[1, 0]

print("  Calculando V0 mínimo...")
x_test = [500, 1000, 2000, 3000, 5000]

# Ground truth (analítico para vácuo, numérico para arrasto)
v0_min_gt = []
theta_min_gt = []
for x in x_test:
    sol = analytic_min_v0(x)  # Já adaptado para arrasto
    if sol:
        v0_min_gt.append(sol['v0'])
        theta_min_gt.append(sol['theta0_deg'])
    else:
        v0_min_gt.append(np.nan)
        theta_min_gt.append(np.nan)

# PINN: usa solve_min_v0 que testa vários ângulos
v0_min_pinn = []
theta_min_pinn = []
for x in x_test:
    sol = solver.solve_min_v0(x, n_angles=15)
    if sol:
        v0_min_pinn.append(sol['v0'])
        theta_min_pinn.append(sol['theta0_deg'])
    else:
        v0_min_pinn.append(np.nan)
        theta_min_pinn.append(np.nan)

# Plota
ax3.plot(x_test, v0_min_gt, 'b-o', linewidth=2, markersize=6, label='Ground Truth')
ax3.scatter(x_test, v0_min_pinn, c='red', s=80, marker='s', 
           edgecolors='black', linewidths=1, label='PINN', zorder=5)

ax3.axhline(y=V0_min, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(y=V0_max, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('x_target (m)')
ax3.set_ylabel('V0 mínimo (m/s)')
title_mode = "arrasto" if USE_DRAG else "vácuo"
ax3.set_title(f'V0 mínimo para atingir cada alvo ({title_mode})')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# Subplot 4: Ângulo ótimo e Erro
ax4 = axes[1, 1]

# Mostra o ângulo ótimo encontrado
width = 0.35
x_pos = np.arange(len(x_test))

bars1 = ax4.bar(x_pos - width/2, theta_min_gt, width, label='θ Ground Truth', color='steelblue')
bars2 = ax4.bar(x_pos + width/2, theta_min_pinn, width, label='θ PINN', color='coral')

ax4.axhline(y=45, color='gray', linestyle='--', alpha=0.7, label='45° (ótimo vácuo)')
ax4.set_xlabel('x_target (m)')
ax4.set_ylabel('θ ótimo (graus)')
ax4.set_title(f'Ângulo que minimiza V0 ({title_mode})')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([str(x) for x in x_test])
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# Calcula erros para referência
erros_v0 = []
for v0_gt, v0_pinn in zip(v0_min_gt, v0_min_pinn):
    if not np.isnan(v0_gt) and not np.isnan(v0_pinn):
        erros_v0.append(abs(v0_pinn - v0_gt) / v0_gt * 100)
    else:
        erros_v0.append(np.nan)

ax4.bar(range(len(x_test)), erros_v0, tick_label=[str(x) for x in x_test], color='steelblue')
ax4.set_xlabel('x_target (m)')
ax4.set_ylabel('Erro V0 (%)')
ax4.set_title('Erro da PINN vs Analítico (θ=45°)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Salva figura
if USE_DRAG:
    images_dir = PROJECT_ROOT / "imagens" / "inverse_arrasto"
else:
    images_dir = PROJECT_ROOT / "imagens" / "inverse_vacuo"
images_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(images_dir / "inverse_solutions.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print(f"\n✅ Visualização salva em {images_dir / 'inverse_solutions.png'}")

# %% Função utilitária para uso externo
def find_cannon_parameters(x_target, objective='min_v0', **kwargs):
    """
    Função principal para resolver o problema inverso.
    
    Args:
        x_target: Distância alvo em metros
        objective: Critério de otimização
            - 'min_v0': Minimiza velocidade inicial (θ ≈ 45°)
            - 'given_theta': Usa ângulo específico (requer theta=...)
            - 'given_v0': Usa velocidade específica (requer v0=...)
    
    Returns:
        dict com v0, theta0_deg, T_impact, x_achieved, error
        ou None se não encontrar solução
    
    Exemplo:
        >>> result = find_cannon_parameters(1500, objective='min_v0')
        >>> print(f"V0 = {result['v0']:.1f} m/s, θ = {result['theta0_deg']:.1f}°")
        
        >>> result = find_cannon_parameters(2000, objective='given_theta', theta=30)
        >>> print(f"V0 = {result['v0']:.1f} m/s")
    """
    if objective == 'min_v0':
        return solver.solve_min_v0(x_target, **kwargs)
    elif objective == 'given_theta':
        theta = kwargs.pop('theta', 45)
        return solver.solve_given_theta(x_target, theta, **kwargs)
    elif objective == 'given_v0':
        v0 = kwargs.pop('v0', 100)
        return solver.solve_given_v0(x_target, v0, **kwargs)
    else:
        raise ValueError(f"Objetivo desconhecido: {objective}")

# %% Demo
print("\n" + "="*70)
print("💡 Como usar:")
print("="*70)
print("""
# Encontrar parâmetros para atingir 1500m com mínimo V0:
result = find_cannon_parameters(1500, objective='min_v0')
print(f"V0 = {result['v0']:.1f} m/s, θ = {result['theta0_deg']:.1f}°")

# Encontrar V0 para atingir 2000m com θ = 30°:
result = find_cannon_parameters(2000, objective='given_theta', theta=30)

# Encontrar θ para atingir 3000m com V0 = 200 m/s:
solutions = find_cannon_parameters(3000, objective='given_v0', v0=200)
# Nota: pode retornar 2 soluções (ângulo baixo e alto)!

# Listar todas as soluções para um alvo:
solutions = solver.find_all_solutions(2500, n_thetas=20)
""")

# %%

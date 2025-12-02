# %% [markdown]
# # Problema Inverso: Dado um alvo x_impact, encontrar (V0, Theta0, T_impact)
#
# ---
#
# Usa a PINN treinada como modelo diferenciável e otimiza (V0, Theta0) para atingir o alvo.
#
# Suporta dois modos:
# - Vácuo: soluções analíticas conhecidas (Theta0=45° minimiza V0)
# - Arrasto: soluções numéricas por integração (Theta0 ótimo < 45°)
#
# #### Como usar: ####
# Encontrar parâmetros para atingir 1500m com mínimo V0:
#
#    result = find_cannon_parameters(1500, objective='min_v0')
#
# Encontrar V0 para atingir 2000m com Theta0 = 30°:
#
#    result = find_cannon_parameters(2000, objective='given_theta', theta=30)
#
# Encontrar Theta0 para atingir 3000m com V0 = 200 m/s:
#
#    solutions = find_cannon_parameters(3000, objective='given_v0', v0=200)
#
# *Nota: pode retornar 2 soluções (ângulo baixo e alto)!*
#
# Listar todas as soluções para um alvo:
#
#    solutions = solver.find_all_solutions(2500, n_thetas=20)
#

# %% Configuração
# ======================================================================
# CONFIGURAÇÕES - MUDE AQUI
# ======================================================================

# Modo: True = arrasto, False = vácuo
USE_DRAG = True

# Alvos a testar (metros)
x_targets = [100, 500, 1000, 1400, 1800]

# Ângulo fixo para teste (graus)
theta_fixo = 54.0

# x_target fixo para listar todas as soluções (metros)
x_target_fixo = 800

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

# %% Parâmetros físicos (valores padrão, serão atualizados do checkpoint se disponível)
g = 9.81  # m/s^2

# Ranges de entrada (padrão)
V0_min, V0_max = 10.0, 250.0        # m/s
theta0_min, theta0_max = 10.0, 80.0  # graus

# Parâmetros de arrasto (padrão - serão atualizados do checkpoint)
Cd = None                       # será atualizado do checkpoint
k_drag = None                   # será atualizado do checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
default_dtype = torch.get_default_dtype()

theta0_min_rad = np.radians(theta0_min)
theta0_max_rad = np.radians(theta0_max)

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

# Atualiza escalas
T_scale = 2 * V0_max * np.sin(np.radians(theta0_max)) / g
X_scale = V0_max**2 / g

# Parâmetros de arrasto do checkpoint (se disponível)
if USE_DRAG:
    Cd = config.get("Cd", 0.47)
    k_drag = config.get("k_drag", None)
    
    # Se k_drag não está no checkpoint, calcula
    if k_drag is None:
        m = 1.0
        rho = 1.225
        densidade_chumbo = 11340
        diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)
        A = np.pi * (diametro / 2) ** 2
        k_drag = (rho * Cd * A) / (2 * m)
    
    print(f"🌬️  Modo: ARRASTO (Cd={Cd}, k_drag={k_drag:.6f})")
else:
    print("🚀 Modo: VÁCUO (sem arrasto)")

print(f"Carregando checkpoint: {checkpoint_path.name}")

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
def simulate_trajectory(v0, theta_deg, use_drag=USE_DRAG):
    """
    Simula a trajetória do projétil usando solve_ivp (muito mais rápido que Euler).
    
    Usa evento para detectar quando y=0 (impacto).
    
    Returns: T_impact, x_impact
    """
    theta_rad = np.radians(theta_deg)
    
    # Condições iniciais: [x, y, vx, vy]
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)
    y0 = [0.0, 0.0, vx0, vy0]
    
    def derivatives(t, state):
        x, y, vx, vy = state
        if use_drag:
            v_mag = np.sqrt(vx**2 + vy**2)
            ax = -k_drag * v_mag * vx
            ay = -g - k_drag * v_mag * vy
        else:
            ax = 0
            ay = -g
        return [vx, vy, ax, ay]
    
    def hit_ground(t, state):
        return state[1]  # y = 0
    hit_ground.terminal = True
    hit_ground.direction = -1  # Só quando y está diminuindo
    
    # Integra até o impacto (evento y=0)
    sol = solve_ivp(
        derivatives, 
        [0, 200],  # t_span (máximo 200s)
        y0,
        events=hit_ground,
        dense_output=False,
        max_step=1.0  # Passo máximo para precisão
    )
    
    if sol.t_events[0].size > 0:
        T_impact = sol.t_events[0][0]
        # Pega o estado no momento do impacto
        x_impact = sol.y_events[0][0][0]
    else:
        # Não atingiu o solo (algo errado)
        T_impact = sol.t[-1]
        x_impact = sol.y[0, -1]
    
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

def ground_truth_min_v0(x_target, use_drag=USE_DRAG):
    """
    Encontra a solução que minimiza V0 para atingir x_target (numérico).
    
    Usa scipy.optimize.minimize_scalar para encontrar o ângulo ótimo.
    Com arrasto, o ângulo ótimo é < 45°.
    """
    from scipy.optimize import minimize_scalar
    
    def v0_for_theta(theta_deg):
        sol = ground_truth_given_theta(x_target, theta_deg, use_drag)
        if sol:
            return sol['v0']
        else:
            return 1e10  # Valor grande mas finito (evita warnings do scipy)
    
    result = minimize_scalar(
        v0_for_theta,
        bounds=(theta0_min + 0.1, theta0_max - 0.1),
        method='bounded',
        options={'xatol': 0.01}
    )
    
    if result.success or result.fun < 1e9:  # 1e9 < 1e10 (valor de "sem solução")
        theta_opt = result.x
        return ground_truth_given_theta(x_target, theta_opt, use_drag)
    
    return None

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
    
    def solve_min_v0(self, x_target):
        """
        Encontra a solução que minimiza V0 para atingir x_target.
        
        Usa scipy.optimize.minimize_scalar para encontrar o ângulo ótimo.
        Para cada ângulo, usa brentq para encontrar o V0 correspondente.
        """
        from scipy.optimize import minimize_scalar
        
        def v0_for_theta(theta_deg):
            """Retorna V0 necessário para atingir x_target com ângulo theta_deg."""
            sol = self.solve_given_theta(x_target, theta_deg)
            if sol:
                return sol['v0']
            else:
                return 1e10  # Valor grande mas finito (evita warnings do scipy)
        
        # Minimiza V0 em função do ângulo
        result = minimize_scalar(
            v0_for_theta,
            bounds=(theta0_min + 0.1, theta0_max - 0.1),
            method='bounded',
            options={'xatol': 0.01}  # Precisão de 0.01°
        )
        
        if result.success or result.fun < 1e9:  # 1e9 < 1e10 (valor de "sem solução")
            theta_opt = result.x
            return self.solve_given_theta(x_target, theta_opt)
        
        return None
    
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
    
    def solve_given_T(self, x_target, T_desired):
        """
        Encontra (V0, θ) para atingir x_target com tempo de voo T_desired.
        
        Usa otimização para encontrar o ângulo que resulta no tempo desejado.
        """
        from scipy.optimize import minimize_scalar
        
        def T_error(theta_deg):
            """Retorna erro |T_achieved - T_desired| para dado ângulo."""
            sol = self.solve_given_theta(x_target, theta_deg)
            if sol:
                return abs(sol['T_impact'] - T_desired)
            else:
                return 1e10
        
        # Encontra ângulo que minimiza erro no tempo
        result = minimize_scalar(
            T_error,
            bounds=(theta0_min + 0.1, theta0_max - 0.1),
            method='bounded',
            options={'xatol': 0.01}
        )
        
        if result.success or result.fun < 1.0:  # Aceita erro < 1s
            theta_opt = result.x
            sol = self.solve_given_theta(x_target, theta_opt)
            if sol:
                sol['T_desired'] = T_desired
                sol['T_error'] = abs(sol['T_impact'] - T_desired)
            return sol
        
        return None
    
    def bateria_tiros(self, x_target, n_tiros=5, delta_T=3.0):
        """
        Calcula uma bateria de tiros que impactam simultaneamente no alvo.
        
        O primeiro tiro é o mais lento (maior T), os seguintes são mais rápidos.
        Se disparados com intervalo delta_T entre cada, todos chegam juntos.
        
        Args:
            x_target: Distância do alvo (m)
            n_tiros: Número de tiros na bateria
            delta_T: Intervalo entre disparos (s)
        
        Returns:
            Lista de soluções com 'disparo_em' indicando quando disparar
        """
        # Primeiro encontra os limites de T possíveis
        all_sols = self.find_all_solutions(x_target, n_thetas=30, verbose=False)
        
        if not all_sols:
            return None
        
        T_min = min(s['T_impact'] for s in all_sols)
        T_max = max(s['T_impact'] for s in all_sols)
        
        # Tempos de voo desejados (do mais lento ao mais rápido)
        T_range_needed = (n_tiros - 1) * delta_T
        
        if T_max - T_min < T_range_needed:
            print(f"⚠️  Aviso: Range de T disponível ({T_max-T_min:.1f}s) < necessário ({T_range_needed:.1f}s)")
            # Ajusta para o range disponível
            T_start = T_max
            actual_delta = (T_max - T_min) / (n_tiros - 1) if n_tiros > 1 else 0
        else:
            T_start = T_max
            actual_delta = delta_T
        
        # Calcula cada tiro
        bateria = []
        for i in range(n_tiros):
            T_desired = T_start - i * actual_delta
            
            if T_desired < T_min:
                continue
                
            sol = self.solve_given_T(x_target, T_desired)
            
            if sol:
                sol['tiro_num'] = i + 1
                sol['disparo_em'] = i * actual_delta  # Quando disparar (relativo ao 1º)
                sol['delta_T_usado'] = actual_delta
                bateria.append(sol)
        
        return bateria

# %% Instancia o solver
solver = InverseProblemSolver(model)

# %% Testes: Minimizar V0
print("\n" + "="*100)
print("PROBLEMA INVERSO: Dado x_target, encontrar (V0, θ0, T_impact)")
print("="*100)

if USE_DRAG:
    print("\n Objetivo: Minimizar V0 (com arrasto, θ ótimo < 45°)")
else:
    print("\n Objetivo: Minimizar V0 (vácuo, sempre θ = 45°)")
print("-" * 120)
print(f"{'x_target':>8} │ {'V0':>8} {'θ':>7} │ {'x_PINN':>8} {'x_real':>8} │ {'V0 Anal.':>8} {'θ Anal.':>7} │ {'Status':>15}")
print("-" * 120)

for x_target in x_targets:
    sol_pinn = solver.solve_min_v0(x_target)
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
        
        # Ground truth
        if sol_ana:
            gt_str = f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}°"
        else:
            gt_str = f"{'---':>8} {'---':>8}"
        
        print(f"{x_target:8.0f} │ {sol_pinn['v0']:8.2f} {sol_pinn['theta0_deg']:7.2f}° │ "
              f"{x_pinn:8.1f} {x_real:8.1f} │ {gt_str} │ {status:>15}")
    else:
        if sol_ana:
            gt_str = f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}°"
        else:
            gt_str = f"{'---':>8} {'---':>8}"
        print(f"{x_target:8.0f} │ {'Não encontrado':>18} │ {'---':>8} {'---':>8} │ {gt_str} │")


# %% Testes: Ângulo fixo
print(f"\n Objetivo: Ângulo fixo de {theta_fixo}°")
print("-" * 120)
print(f"{'x_target':>8} │ {'V0':>8} {'θ':>7} │ {'x_PINN':>8} {'x_real':>8} │ {'V0 Anal.':>8} {'θ Anal.':>7} │ {'Status':>15}")
print("-" * 120)

for x_target in x_targets:
    sol_pinn = solver.solve_given_theta(x_target, theta_fixo)
    sol_ana = analytic_given_theta(x_target, theta_fixo)
    
    if sol_pinn:
        x_pinn = sol_pinn['x_achieved']
        x_real = compute_x_real(sol_pinn['v0'], sol_pinn['theta0_deg'])
        
        erro_pinn = abs(x_pinn - x_target) / x_target * 100
        erro_real = abs(x_real - x_target) / x_target * 100
        
        if erro_real < 2:
            status = "✓ OK"
        else:
            status = f"✗ Real:{erro_real:.1f}%"
        
        if sol_ana:
            gt_str = f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}°"
        else:
            gt_str = f"{'---':>8} {'---':>8}"
        
        print(f"{x_target:8.0f} │ {sol_pinn['v0']:8.2f} {sol_pinn['theta0_deg']:7.2f}° │ "
              f"{x_pinn:8.1f} {x_real:8.1f} │ {gt_str} │ {status:>15}")
    else:
        if sol_ana:
            gt_str = f"{sol_ana['v0']:8.2f} {sol_ana['theta0_deg']:7.1f}°"
        else:
            gt_str = "Impossível"
        print(f"{x_target:8.0f} │ {'Não encontrado':>18} │ {'---':>8} {'---':>8} │ {gt_str:>16} │")

# %% Testes: Todas as soluções para x_target fixo
print(f"\n Todas as soluções para x_target = {x_target_fixo}m (PINN vs Ground Truth)")
print("-" * 70)

sol_pinn = solver.find_all_solutions(x_target_fixo, n_thetas=15, verbose=False)
sol_ana = analytic_all_solutions(x_target_fixo, n_points=15)

print(f"{'θ (°)':>8} │ {'V0 PINN':>10} {'T PINN':>8} │ {'V0 GT':>10} {'T GT':>8} │ {'Erro V0 %':>10}")
print("-" * 70)

if sol_pinn and sol_ana:
    # Cria dicionário por ângulo para matching
    ana_by_theta = {round(s['theta0_deg'], 1): s for s in sol_ana}
    
    for sp in sol_pinn:
        theta_key = round(sp['theta0_deg'], 1)
        sa = ana_by_theta.get(theta_key)
        
        if sa:
            erro_v0 = abs(sp['v0'] - sa['v0']) / sa['v0'] * 100
            print(f"{sp['theta0_deg']:8.1f} │ {sp['v0']:10.1f} {sp['T_impact']:8.2f} │ "
                  f"{sa['v0']:10.1f} {sa['T_impact']:8.2f} │ {erro_v0:10.2f}")
        else:
            print(f"{sp['theta0_deg']:8.1f} │ {sp['v0']:10.1f} {sp['T_impact']:8.2f} │ "
                  f"{'---':>10} {'---':>8} │ {'---':>10}")
elif sol_pinn:
    for sp in sol_pinn:
        print(f"{sp['theta0_deg']:8.1f} │ {sp['v0']:10.1f} {sp['T_impact']:8.2f} │ "
              f"{'---':>10} {'---':>8} │ {'---':>10}")
else:
    print("Nenhuma solução encontrada.")

# %% Bateria de Tiros - Impacto Simultâneo
# Cria diretório de imagens (usado aqui e na visualização geral)
images_dir = PROJECT_ROOT / "imagens" / "inverse_solver"
images_dir.mkdir(parents=True, exist_ok=True)

print(f"\n BATERIA DE TIROS - Impacto Simultâneo em x = {x_target_fixo}m")
print("=" * 85)
print("Tiros disparados em sequência que impactam todos ao mesmo tempo no alvo.")
print("-" * 85)

# Parâmetros da bateria
n_tiros_bateria = 5
delta_T_bateria = 3.0  # segundos entre disparos

bateria = solver.bateria_tiros(x_target_fixo, n_tiros=n_tiros_bateria, delta_T=delta_T_bateria)

if bateria:
    print(f"{'Tiro':>6} │ {'Dispara em':>10} │ {'T_voo':>8} {'V0':>10} {'θ':>8} │ {'Impacta em':>12}")
    print("-" * 85)
    
    # Tempo de impacto = tempo de disparo + tempo de voo
    # Para impacto simultâneo, todos devem impactar no mesmo instante
    T_impacto_comum = bateria[0]['disparo_em'] + bateria[0]['T_impact']
    
    for tiro in bateria:
        t_disparo = tiro['disparo_em']
        t_impacto = t_disparo + tiro['T_impact']
        print(f"{tiro['tiro_num']:6d} │ {t_disparo:10.2f}s │ {tiro['T_impact']:8.2f}s "
              f"{tiro['v0']:10.1f} {tiro['theta0_deg']:8.2f}° │ {t_impacto:12.2f}s")
    
    print("-" * 85)
    print(f"Todos os {len(bateria)} tiros impactam em t ≈ {T_impacto_comum:.2f}s")
    
    # Visualização da Bateria de Tiros
    print("\n📊 Gerando visualização da Bateria de Tiros...")
    
    def get_full_trajectory(v0, theta_deg, n_points=100):
        """Retorna trajetória completa (x, y) do disparo."""
        theta_rad = np.radians(theta_deg)
        
        if USE_DRAG:
            # Integra numericamente
            vx0 = v0 * np.cos(theta_rad)
            vy0 = v0 * np.sin(theta_rad)
            y0 = [0.0, 0.0, vx0, vy0]
            
            def derivatives(t, state):
                x, y, vx, vy = state
                v_mag = np.sqrt(vx**2 + vy**2)
                ax = -k_drag * v_mag * vx
                ay = -g - k_drag * v_mag * vy
                return [vx, vy, ax, ay]
            
            def hit_ground(t, state):
                return state[1]
            hit_ground.terminal = True
            hit_ground.direction = -1
            
            sol = solve_ivp(derivatives, [0, 200], y0, events=hit_ground, 
                           dense_output=True, max_step=0.5)
            
            t_eval = np.linspace(0, sol.t[-1], n_points)
            states = sol.sol(t_eval)
            return states[0], states[1], t_eval  # x, y, t
        else:
            # Analítico (vácuo)
            T = 2 * v0 * np.sin(theta_rad) / g
            t = np.linspace(0, T, n_points)
            x = v0 * np.cos(theta_rad) * t
            y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
            return x, y, t
    
    fig_bateria, ax_bat = plt.subplots(figsize=(12, 7))
    
    # Cores em degradê (do primeiro ao último tiro)
    cores_bateria = plt.cm.plasma(np.linspace(0.2, 0.9, len(bateria)))
    
    y_max = 0
    for i, tiro in enumerate(bateria):
        x_traj, y_traj, t_traj = get_full_trajectory(tiro['v0'], tiro['theta0_deg'])
        
        # Plota trajetória tracejada
        label = f"Tiro {tiro['tiro_num']}: V0={tiro['v0']:.0f}m/s, θ={tiro['theta0_deg']:.1f}°"
        ax_bat.plot(x_traj, y_traj, '--', color=cores_bateria[i], linewidth=2, 
                   alpha=0.8, label=label)
        
        y_max = max(y_max, max(y_traj))
    
    # Marca o alvo com bolinha vermelha
    ax_bat.scatter([x_target_fixo], [0], c='red', s=200, marker='o', 
                  edgecolors='black', linewidths=2, zorder=10, label=f'Alvo: {x_target_fixo}m')
    
    # Adiciona ícone do canhão na origem
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    cannon_path = PROJECT_ROOT / "apresentacao" / "canhao_para_grafico.png"
    if cannon_path.exists():
        cannon_img = plt.imread(cannon_path)
        # Ajusta o tamanho do canhão proporcionalmente ao gráfico
        zoom_factor = 0.3  # Ajuste conforme necessário
        imagebox = OffsetImage(cannon_img, zoom=zoom_factor)
        # Posiciona ligeiramente acima da origem para não cobrir o chão
        ab = AnnotationBbox(imagebox, (20, 50), frameon=False, zorder=15)
        ax_bat.add_artist(ab)
    
    # Linha do chão
    ax_bat.axhline(y=0, color='saddlebrown', linewidth=2, alpha=0.5)
    ax_bat.fill_between([0, x_target_fixo * 1.1], [0, 0], [-y_max*0.1, -y_max*0.1], 
                       color='saddlebrown', alpha=0.2)
    
    ax_bat.set_xlabel('Distância (m)', fontsize=12)
    ax_bat.set_ylabel('Altura (m)', fontsize=12)
    mode_str = "com arrasto" if USE_DRAG else "no vácuo"
    ax_bat.set_title(f'BATERIA DE TIROS - Impacto Simultâneo em x={x_target_fixo}m ({mode_str})\n'
                    f'{len(bateria)} tiros com Δt={bateria[0]["delta_T_usado"]:.1f}s entre disparos', 
                    fontsize=14)
    ax_bat.legend(loc='upper right', fontsize=9)
    ax_bat.grid(True, alpha=0.3)
    ax_bat.set_xlim(-x_target_fixo*0.05, x_target_fixo*1.1)
    ax_bat.set_ylim(-y_max*0.1, y_max*1.15)
    
    plt.tight_layout()
    
    # Salva figura da bateria
    mode_prefix = "arrasto" if USE_DRAG else "vacuo"
    filename_bateria = f"{mode_prefix}_bateria_xfix{x_target_fixo}.png"
    plt.savefig(images_dir / filename_bateria, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"✅ Visualização da bateria salva em {images_dir / filename_bateria}")
else:
    print("Não foi possível calcular a bateria de tiros.")

# %% Visualização (RÁPIDA - usa dados analíticos + poucos pontos PINN)
print("\n📊 Gerando visualização...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Usa os alvos definidos no setup
x_targets_viz = x_targets
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
x_test = x_targets  # Usa os alvos definidos no setup

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
    sol = solver.solve_min_v0(x)
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

plt.tight_layout()

# Nome do arquivo com parâmetros (images_dir já definido acima)
mode_prefix = "arrasto" if USE_DRAG else "vacuo"
x_min, x_max = min(x_targets), max(x_targets)
if USE_DRAG:
    filename = f"{mode_prefix}_x{x_min}-{x_max}_theta{theta_fixo:.0f}_xfix{x_target_fixo}_Cd{Cd}.png"
else:
    filename = f"{mode_prefix}_x{x_min}-{x_max}_theta{theta_fixo:.0f}_xfix{x_target_fixo}.png"

plt.savefig(images_dir / filename, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print(f"\n✅ Visualização salva em {images_dir / filename}")

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

# %%

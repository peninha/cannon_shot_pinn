"""
Script para gerar gráfico limpo apenas do ground truth da trajetória balística com arrasto
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters (from cannonshot_PINN_arrasto.py)
g = 9.81                         # m/s^2
v0 = 200.0                       # m/s
theta0_deg = 18.0                # degrees
theta0 = np.radians(theta0_deg)  # radians
y0 = 0.0                         # m
x0 = 0.0                         # m

# Drag parameters
m = 1.0                          # kg - projectile mass
rho = 1.225                      # kg/m^3 - air density
densidade_chumbo = 11340         # kg/m^3 - lead density
diametro = (6 * m / np.pi / densidade_chumbo) ** (1/3)  # m - projectile diameter
A = np.pi * (diametro / 2) ** 2  # m^2 - frontal area

# True drag coefficient
Cd_true = 2.0

def shot_flight_time(y0, v0, theta0, g):
    """Calculate approximate flight time (vacuum approximation)"""
    tf = 1/g * (v0*np.sin(theta0) + np.sqrt(v0**2*np.sin(theta0)**2 + 2*g*y0))
    return tf

def shot_state_ground_truth(t):
    """
    Calculate projectile state at time t (with quadratic drag).
    Uses numerical integration of the drag equations.
    Returns: x, y, vx, vy
    """
    t = np.asarray(t)
    t_flat = t.flatten()
    
    # Sort times and keep index for reordering at the end
    sorted_idx = np.argsort(t_flat)
    sorted_vals = t_flat[sorted_idx]
    max_t = float(sorted_vals[-1]) if sorted_vals.size > 0 else 0.0
    
    # Adaptive integration step
    dt = float(min(1e-3, max(max_t/5000.0, 1e-4)))

    # Constants
    k_true = (rho * Cd_true * A) / (2.0 * m)
    vx0 = v0 * np.cos(theta0)
    vy0 = v0 * np.sin(theta0)
    
    # Scalar states for integration
    x, y = x0, y0
    vx, vy = vx0, vy0
    t_curr = 0.0
    hit_ground = False

    # Helper function to advance one small step
    def step_state(vx_f, vy_f, x_f, y_f, h):
        speed = (vx_f * vx_f + vy_f * vy_f) ** 0.5
        ax = -k_true * speed * vx_f
        ay = -g - k_true * speed * vy_f
        vx_new = vx_f + h * ax
        vy_new = vy_f + h * ay
        x_new = x_f + h * vx_f
        y_new = y_f + h * vy_f
        return vx_new, vy_new, x_new, y_new

    # Output arrays
    out_x = np.empty_like(sorted_vals)
    out_y = np.empty_like(sorted_vals)
    out_vx = np.empty_like(sorted_vals)
    out_vy = np.empty_like(sorted_vals)

    j = 0
    for tau in sorted_vals:
        tau_f = float(tau)
        # Advance from t_curr to tau_f
        while t_curr + 1e-12 < tau_f:
            h = min(dt, tau_f - t_curr)
            if not hit_ground:
                vx, vy, x, y = step_state(vx, vy, x, y, h)
                if y <= 0.0 and vy < 0.0:
                    # Ground collision: fix state from here
                    y = 0.0
                    vx = 0.0
                    vy = 0.0
                    hit_ground = True
            t_curr += h

        # Record state at tau
        out_x[j] = x
        out_y[j] = y
        out_vx[j] = vx
        out_vy[j] = vy
        j += 1

    # Reorder to original format
    inv_idx = np.empty_like(sorted_idx)
    inv_idx[sorted_idx] = np.arange(sorted_idx.size)
    x_out = out_x[inv_idx].reshape(t.shape)
    y_out = out_y[inv_idx].reshape(t.shape)
    vx_out = out_vx[inv_idx].reshape(t.shape)
    vy_out = out_vy[inv_idx].reshape(t.shape)
    return x_out, y_out, vx_out, vy_out

# Generate trajectory
T = shot_flight_time(y0, v0, theta0, g)
t_eval = np.linspace(0, T, 2000)  # More points for smooth curve with drag
x, y, vx, vy = shot_state_ground_truth(t_eval)

# Find actual flight time (when projectile hits ground)
ground_idx = np.where(y <= 0)[0]
if len(ground_idx) > 1:
    actual_flight_time = t_eval[ground_idx[1]]
    actual_range = x[ground_idx[1]]
else:
    actual_flight_time = T
    actual_range = x[-1]

# Plot clean trajectory
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('trajetoria_ground_truth_arrasto.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráfico salvo como 'trajetoria_ground_truth_arrasto.png'")
print(f"Alcance: {actual_range:.2f} m")
print(f"Altura máxima: {y.max():.2f} m")
print(f"Tempo de voo: {actual_flight_time:.2f} s")
print(f"Coeficiente de arrasto (Cd): {Cd_true}")
print(f"Velocidade inicial: {v0:.1f} m/s")
print(f"Ângulo de lançamento: {theta0_deg:.1f}°")


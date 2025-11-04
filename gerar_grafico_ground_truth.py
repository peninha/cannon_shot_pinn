"""
Script para gerar gráfico limpo apenas do ground truth da trajetória balística
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
g = 9.81                         # m/s^2
v0 = 50.0                        # m/s
theta0_deg = 63.0                # degrees
theta0 = np.radians(theta0_deg)  # radians
y0 = 0.0                         # m
x0 = 0.0                         # m

def shot_flight_time(y0, v0, theta0, g):
    """Calculate total flight time"""
    tf = 1/g * (v0*np.sin(theta0) + np.sqrt(v0**2*np.sin(theta0)**2 + 2*g*y0))
    return tf

def shot_state_ground_truth(t):
    """
    Calculate projectile state at time t (no drag).
    Returns: x, y, vx, vy
    """
    t = np.asarray(t)
    vx0 = v0 * np.cos(theta0)
    vy0 = v0 * np.sin(theta0)
    
    T = shot_flight_time(y0, v0, theta0, g)
    t_clamped = np.minimum(t, T)
    
    x = x0 + vx0 * t_clamped
    y = y0 + vy0 * t_clamped - 0.5 * g * (t_clamped ** 2)
    
    return x, y

# Generate trajectory
T = shot_flight_time(y0, v0, theta0, g)
t_eval = np.linspace(0, T, 1000)
x, y = shot_state_ground_truth(t_eval)

# Plot clean trajectory
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('trajetoria_ground_truth.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Gráfico salvo como 'trajetoria_ground_truth.png'")
print(f"Alcance: {x[-1]:.2f} m")
print(f"Altura máxima: {y.max():.2f} m")
print(f"Tempo de voo: {T:.2f} s")


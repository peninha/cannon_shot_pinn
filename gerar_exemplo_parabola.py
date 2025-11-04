# Adjust the position of the velocity vector label and make the arrow larger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Parameters
g = 9.81
v0 = 22.0
theta0_deg = 52.0
theta0 = np.radians(theta0_deg)

# Time and trajectory
t_flight = 2 * v0 * np.sin(theta0) / g
t = np.linspace(0, t_flight, 300)
x = v0 * np.cos(theta0) * t
y = v0 * np.sin(theta0) * t - 0.5 * g * t**2

# Plot setup
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, y, label="Trajetória (sem arrasto)")
ax.axhline(0, linewidth=1)
ax.scatter([0], [0], s=120, zorder=5)

# Larger and repositioned velocity vector
arrow_scale = 0.2  # increased arrow length scale
vx0 = v0 * np.cos(theta0)
vy0 = v0 * np.sin(theta0)
ax.arrow(0, 0, vx0*arrow_scale, vy0*arrow_scale, head_width=0.6, head_length=0.8, color="black", length_includes_head=True)

# Shifted label for v0
ax.text(vx0*arrow_scale*0.8, vy0*arrow_scale*1.15, r"$\vec{v}_0$", ha="right", va="bottom", fontsize=12)

# Angle arc
arc_radius = max(1.0, 0.08 * x.max())
arc = Arc((0, 0), width=2*arc_radius, height=2*arc_radius, angle=0, theta1=0, theta2=theta0_deg, linewidth=1.5)
ax.add_patch(arc)
ax.text(arc_radius*0.8*np.cos(theta0/2), arc_radius*0.8*np.sin(theta0/2), r"$\theta_0$", ha="center", va="center")

# Formatting
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.5, x.max()*1.05)
ax.set_ylim(-0.5, max(y)*1.15)
ax.legend(loc="upper right", frameon=False)
ax.set_title("Trajetória balística com $\\vec{v}_0$ e $\\theta_0$")

# Save figure
output_path = "trajetoria_balistico_vetor_ajustado.png"
plt.tight_layout()
plt.savefig(output_path, dpi=200)
output_path

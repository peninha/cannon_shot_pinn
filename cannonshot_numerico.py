#%%
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
# # Caso simples - Tiro de canhão no vácuo
#
# Neste caso, desprezamos a resistência do ar e consideramos que o canhão está no vácuo.
#
# A equação diferencial que descreve o problema é:
#
# d2y/dt2 = -g ; d2x/dt2 = 0
#
# com as condições iniciais:
#
# y(0) = 0  ;  x(0) = 0
#
# dy/dt(0) = v0*sin(theta0)  ;  dx/dt(0) = v0*cos(theta0)
#
# A solução analítica para este problema é:
#
# x(t) = v0*cos(theta0)*t  ;  y(t) = v0*sin(theta0)*t - 0.5*g*t**2

#%%
x0 = 0
y0 = 0
v0 = 50
theta0 = 45
g = 9.81

#%%
def shot_trajectory(t,v0=10,theta0=4,x0=0,y0=0,g=9.81):
    theta0_rad = theta0*np.pi/180
    tf = 1/g * (v0*np.sin(theta0_rad) + np.sqrt(v0**2*np.sin(theta0_rad)**2 + 2*g*y0))
    t = np.minimum(t, tf)
    x = x0 + v0*np.cos(theta0_rad)*t
    y = y0 + v0*np.sin(theta0_rad)*t - 0.5*g*t**2
    return x,y

#%%
dt = 0.001
tf = 15
t = np.arange(0,tf,dt)
x, y = shot_trajectory(t, v0, theta0, x0, y0, g)

plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajetória: y vs x")
plt.axis('equal')  # mesma escala nos eixos X e Y

plt.subplot(1,2,2)
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Altura: y vs t")

plt.tight_layout()
plt.show()

#%%
# Numericamente, usando o método de Euler
def cannon_shot_euler(t,v0,theta0,g):
    theta0_rad = theta0*np.pi/180
    dxdt = v0*np.cos(theta0_rad)
    dydt = v0*np.sin(theta0_rad) - g*t
    return dxdt, dydt
#%%

dt = 0.001
tf = 15
nt = int(tf/dt)
x = np.zeros(nt)
y = np.zeros(nt)
t = np.arange(0,tf,dt)

for i in range(1,nt):
    dxdt, dydt = cannon_shot_euler(t[i],v0,theta0,g)
    y[i] = y[i-1] + dt*dydt
    x[i] = x[i-1] + dt*dxdt
    if y[i] < 0:
        x[i] = x[i-1]
        y[i] = y[i-1]

#%%
# Plota os gráficos para a solução obtida pelo método de Euler

plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajetória numérica: y vs x (Euler)")
plt.axis('equal')  # mesma escala nos eixos X e Y

plt.subplot(1,2,2)
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Altura numérica: y vs t (Euler)")

plt.tight_layout()
plt.show()
#%% [markdown]
# # Physics-Informed Neural Network

#%% [markdown]
# ## Importar bibliotecas
#
# Nesse caso, usaremos o PyTorch para poder utilizar a funcionalidade de diferenciação automática

#%%
import torch
import torch.nn as nn
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

np.random.seed(42)

#%% [markdown]
# # Geração do domínio
#
# Nesta seção, geraremos os pontos que a rede usará como referência para encontrar a solução
#
# Estes pontos estão uniformemente distribuídos ao longo do domínio.

#%%
def gerar_pontos_equacao(pontos_no_dominio, tempo_final):
  t_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=tempo_final)
  return t_dominio

#%% [markdown]
# ## Gerar pontos e plotar
#
# Agora, vamos usar as funções que montamos acima para gerar os pontos

#%%
def shot_trajectory(t,v0=10,theta0=4,x0=0,y0=0,g=9.81):
    theta0_rad = theta0*np.pi/180
    tf = 1/g * (v0*np.sin(theta0_rad) + np.sqrt(v0**2*np.sin(theta0_rad)**2 + 2*g*y0))
    t = np.minimum(t, tf)
    x = x0 + v0*np.cos(theta0_rad)*t
    y = y0 + v0*np.sin(theta0_rad)*t - 0.5*g*t**2
    return x,y

#%%
tempo_final = 15

x0 = 0
y0 = 0
v0 = 50
theta0 = 63
g = 9.81

pontos_no_dominio = 100

t_dominio = gerar_pontos_equacao(pontos_no_dominio,tempo_final)

#%% [markdown]
# Plotamos uma vista da trajetória do projétil no domínio para ver se os pontos estão nos lugares corretos

#%%
# Vista da trajetória do projétil no domínio
from plotly.subplots import make_subplots

x, y = shot_trajectory(t_dominio, v0, theta0, x0, y0, g)

# Create subplots with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Trajetória: y vs x', 'Altura: y vs t'))

# Add trajectory plot (x vs y)
fig.add_trace(go.Scatter(x=x.flatten(), y=y.flatten(), 
                         mode='markers', 
                         marker=dict(color='red'),
                         name='Pontos na trajetória'),
              row=1, col=1)

# Add height vs time plot (t vs y)
fig.add_trace(go.Scatter(x=t_dominio.flatten(), y=y.flatten(), 
                         mode='markers', 
                         marker=dict(color='blue'),
                         name='Altura vs tempo'),
              row=1, col=2)

# Update axes labels
fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_yaxes(title_text="y", row=1, col=1)
fig.update_xaxes(title_text="t", row=1, col=2)
fig.update_yaxes(title_text="y", row=1, col=2)

# Update layout
fig.update_layout(height=400, width=1000, showlegend=True)
fig.show()

#%% [markdown]
# # Definição da Rede Neural
#%%
class PINN(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, num_layers=5, hidden_dim=20, learn_params=True, g=9.81, v0=50, theta0=63, x0=0, y0=0):
        super(PINN, self).__init__()
        self.learn_params = learn_params
        self.g = g
        self.v0 = v0
        self.theta0 = theta0
        self.x0 = x0
        self.y0 = y0

        # Neural network for displacement x(t)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        # Output layer (displacement)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # Predict displacement x(t) and y(t)
        # Returns tensor of shape (N, 2) where [:, 0] is x and [:, 1] is y
        output = self.net(t)
        return output

    def physics_informed_loss(self, t):
        # Ensure t requires gradients
        t.requires_grad_(True)

        # Get predicted displacement r(t) = [x(t), y(t)]
        r = self(t)

        # Physics-informed loss: compare predicted position with physics equation
        physics_r = shot_trajectory(t, self.v0, self.theta0, self.x0, self.y0, self.g)
        
        # Calculate Euclidean distance between predicted and physics position
        pde_residual = torch.sum((physics_r - r)**2, dim=1, keepdim=True)
        
        # Mean squared error for the PDE residual
        loss_pde = torch.mean(pde_residual)

        return loss_pde

    def data_loss(self, t_data, x_data):
        # Predict displacement at data points
        x_pred = self(t_data)

        # Mean squared error for the data points
        loss_data = torch.mean((x_pred - x_data)**2)

        return loss_data



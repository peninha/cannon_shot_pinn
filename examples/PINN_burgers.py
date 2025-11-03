#%% [markdown]
# # Physics-Informed Neural Network
#
# Neste notebook resolveremos a equação de Burgers em duas dimensões usando uma rede neural informada por física
#
# Consulte os slides da aula 7 para a teoria e a definição do problema em mais detalhes

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

np.random.seed(1)

#%% [markdown]
# # Geração do domínio
#
# Nesta seção, geraremos os pontos que a rede usará como referência para encontrar a solução

#%% [markdown]
# ## Criar função que gera pontos no contorno do domínio
#
# Primeiro, vamos gerar pontos ao longo da condição de contorno. Nestes pontos, sabemos qual deve ser o valor de z. Eles servem para "ancorar" o resto da superfície da solução.
#
# O domínio tem 4 lados, cada um deles terá 1/4 do número total de pontos.

#%%
def gerar_pontos_contorno(pontos_no_contorno,comprimento_x,tempo_final):
  pontos_por_lado = pontos_no_contorno//4

  # Lado 1 (x = 0, qualquer t)
  x_lado1 = 0 * np.ones((pontos_por_lado,1))
  t_lado1 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

  u_lado1 = 0 * np.ones((pontos_por_lado,1))

  # Lado 2 (x = comprimento_x, qualquer t)
  x_lado2 = comprimento_x * np.ones((pontos_por_lado,1))
  t_lado2 = np.random.uniform(size=(pontos_por_lado,1),low=0,high=tempo_final)

  u_lado2 = 0 * np.ones((pontos_por_lado,1))

  # Condicao inicial (x = qualquer, t = 0)
  x_inicial = np.random.uniform(size=(2*pontos_por_lado,1),low=0,high=comprimento_x)
  t_inicial = 0 * np.ones((2*pontos_por_lado,1))

  u_inicial = np.sin(2*np.pi*x_inicial/comprimento_x)
  #u_inicial = (1-x_inicial)*x_inicial

  # Juntar todos os lados
  x_todos = np.vstack((x_lado1,x_lado2,x_inicial))
  t_todos = np.vstack((t_lado1,t_lado2,t_inicial))
  u_todos = np.vstack((u_lado1,u_lado2,u_inicial))

  # Criar arrays X e Y
  X_contorno = np.hstack((x_todos,t_todos))
  Y_contorno = u_todos

  return X_contorno, Y_contorno

#%% [markdown]
# ## Criar função que gera pontos de avaliação da equação
#
# Esta é a segunda classe de pontos que usaremos. Neles, não sabemos a solução. Mas sabemos qual equação eles devem obedecer.
#
# Estes pontos estão uniformemente distribuídos ao longo do domínio.

#%%
def gerar_pontos_equacao(pontos_no_dominio,comprimento_x,tempo_final):
  x_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=comprimento_x)
  t_dominio = np.random.uniform(size=(pontos_no_dominio,1),low=0,high=tempo_final)

  X_equacao = np.hstack((x_dominio,t_dominio))

  return X_equacao

#%% [markdown]
# ## Gerar pontos e plotar
#
# Agora, vamos usar as funções que montamos acima para gerar os pontos

#%%
comprimento_x = 1
tempo_final = 1

pontos_no_contorno = 600
pontos_no_dominio = 1000

X_contorno, Y_contorno = gerar_pontos_contorno(pontos_no_contorno,comprimento_x,tempo_final)
X_equacao = gerar_pontos_equacao(pontos_no_dominio,comprimento_x,tempo_final)

#%% [markdown]
# Plotamos uma vista superior do domínio para ver se os pontos estão nos lugares corretos

#%%
# Vista superior
#fig = plt.figure()
#ax = fig.add_subplot()
scatter_contorno = px.scatter(x=X_contorno[:,0],y=X_contorno[:,1])
scatter_equacao = px.scatter(x=X_equacao[:,0],y=X_equacao[:,1], color_discrete_sequence=['red'])
fig = go.Figure(data=scatter_contorno.data+scatter_equacao.data)
fig.update_layout(xaxis_title='x',yaxis_title='t')
fig.show()

#%% [markdown]
# E uma vista em perspectiva apenas dos pontos do contorno, para ver se parece correto

#%%
# Vista em perspectiva
scatter_3d = px.scatter_3d(x=X_contorno[:,0].flatten(),y=X_contorno[:,1].flatten(),z=Y_contorno.flatten())
fig = go.Figure(scatter_3d)
fig.update_layout(scene=dict(aspectratio=dict(x=1.5, y=1.5, z=0.5)))
fig.show()

#%% [markdown]
# # Definição da Rede Neural
#
# Neste seção, vamos definir a rede neural. Primeiro, montamos sua estrutura, depois definimos a função de perda e o otimizador.

#%% [markdown]
# ## Estrutura da rede neural
#
# Vamos criar uma função que monta uma rede neural totalmente conectada baseada em uma lista com o número de neurônios em cada camada.
#
# Note que removemos a função de ativação da última camada pois estamos fazendo um ajuste de função

#%%
def criar_rede_neural(numero_de_neuronios):

  # Criar uma lista de todas as camadas
  camadas = []

  # Para cada camada, adicionar as conexões e a função de ativação
  for i in range(len(numero_de_neuronios)-1):
    camadas.append(nn.Linear(numero_de_neuronios[i],numero_de_neuronios[i+1]))
    camadas.append(nn.Tanh())

  # Remover a última camada, pois é a função de ativação
  camadas.pop()
  #camadas.pop()

  # Criar rede
  return nn.Sequential(*camadas)

#%% [markdown]
# Agora, definimos o número de neurônios por camada e chamamos a função para inicializar a rede.
#
# Note que a primeira camada deve ter dois neurônios, pois a função tem duas entradas.
#
# De forma similar, a última camada deve ter apenas um neurônio, pois a função tem apenas uma saída.

#%%
numero_de_neuronios = [2, 20, 20, 20, 1]

rna = criar_rede_neural(numero_de_neuronios)

print(rna)

#%% [markdown]
# ## Definição das funções de perda
#
# Nesta forma de implementar um PINN, precisamos de duas funções de perda diferentes. Uma será responsável pelas condições de contorno. A outra, pela observância das equações.

#%% [markdown]
# A perda responsável pelas condições de contorno funciona de maneira bem similar a uma rede neural convencional:
# Comparamos a previsão da rede em cada ponto com o valor que temos de referência para estes pontos.

#%%
def calc_perda_contorno(rna,X_contorno,Y_contorno):
  Y_predito = rna(X_contorno)
  return nn.functional.mse_loss(Y_predito, Y_contorno)

#%% [markdown]
# Já a perda responsável pela observância das equações funciona de forma bem diferente:
#
# Usaremos a funcionalidade de derivadas automáticas (autograd) para obter as derivadas parciais das saídas da rede em função das entradas. Essas derivadas parciais serão usadas para se calcular o resíduo da equação em cada um dos pontos.

#%%
def calc_residuo(rna,X_equacao):
  x = X_equacao[:,0].reshape(-1, 1)
  t = X_equacao[:,1].reshape(-1, 1)

  u = rna(torch.hstack((x,t)))

  u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]
  u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),retain_graph=True, create_graph=True)[0]

  return u_t + u*u_x - 0.01/np.pi*u_xx

#%%
def calc_perda_equacao(rna,X_equacao):

  residuo = calc_residuo(rna,X_equacao)

  return torch.mean(torch.square(residuo))

#%%
def calc_perda(rna,X_contorno,Y_contorno,X_equacao,alpha=0.2):

  perda_contorno = calc_perda_contorno(rna,X_contorno,Y_contorno)
  perda_equacao = calc_perda_equacao(rna,X_equacao)

  perda = (1-alpha)*perda_contorno + alpha*perda_equacao

  return perda, perda_contorno, perda_equacao

#%% [markdown]
# ## Definir otimizador
#
# No PyTorch, uma das entradas para o otimizador são os parâmetros da rede. É a partir daqui que o otimizador "sabe" quais variáveis ele pode alterar.
#
# O Agendador serve para alterar parâmetros do otimizador ao longo da execução. Neste exemplo, iremos diminuir a taxa de aprendizado. Ela será multiplicada por 0.9 a cada 1000 épocas.
#
# O valor de alpha serve para equilibrar as perdas de contorno com as perdas de equação.

#%%
otimizador = torch.optim.Adam(rna.parameters(),lr=0.01)
agendador = torch.optim.lr_scheduler.StepLR(otimizador, step_size=1000, gamma=0.9)
alpha = 0.1

#%% [markdown]
# # Criar tensores e transferir para GPU
#
# No PyTorch, as variáveis devem ser armazenadas em tensores.
#
# Além disso, se formos rodar em GPU, precisamos manualmente carregar a rede a as variáveis na memória do GPU.
#
# Note a opção "requires_grad" que está ativa na variável X_equação, isso sinaliza que o PyTorch deverá manter a trilha de todas as operações feitas a partir desta variável para, depois, conseguir usar a regra da cadeia e calcular as derivadas.

#%%
X_equacao = torch.tensor(X_equacao,requires_grad=True,dtype=torch.float)
X_contorno = torch.tensor(X_contorno,dtype=torch.float)
Y_contorno = torch.tensor(Y_contorno,dtype=torch.float)

device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
X_equacao = X_equacao.to(device)
X_contorno = X_contorno.to(device)
Y_contorno = Y_contorno.to(device)
rna = rna.to(device)

#%% [markdown]
# # Testar modelo
#
# Neste seção, vamos varificar se o modelo foi construido corretamente e preparar funções para ver os resulados

#%% [markdown]
# ## Rodar alguns passos de otimização
#
# Primeiro, vamos rodar 10 épocas

#%%
# Colocar rede em modo de treinamento
rna.train()

# FAZER ITERAÇÃO
for epoca in range(10):

  # Inicializar gradientes
  otimizador.zero_grad()

  # Calcular perdas
  perda, perda_contorno, perda_equacao = calc_perda(rna,X_contorno,Y_contorno,X_equacao,alpha=alpha)

  # Backpropagation
  perda.backward()

  # Passo do otimizador
  otimizador.step()
  agendador.step()

  # Mostrar resultados
  print(f'Epoca: {epoca}, Perda: {perda.item()} (Contorno: {perda_contorno.item()}, Equacao: {perda_equacao.item()})')

#%% [markdown]
# ## Exibir resultados
#
# Agora, vamos preparar uma função que calcula a rna em um grid e outra que plota a solução

#%%
def calcular_grid(rna, comprimento_x, tempo_final, nx=101, nt=101):

    # Definir grid
    x = np.linspace(0.,comprimento_x,nx)
    t = np.linspace(0.,tempo_final,nt)
    [t_grid, x_grid] = np.meshgrid(t,x)
    x = torch.tensor(x_grid.flatten()[:,None],requires_grad=True,dtype=torch.float).to(device)
    t = torch.tensor(t_grid.flatten()[:,None],requires_grad=True,dtype=torch.float).to(device)

    # Avaliar modelor
    rna.eval()
    Y_pred = rna(torch.hstack((x,t)))

    # Formatar resultados em array
    u_pred = Y_pred.cpu().detach().numpy()[:,0].reshape(x_grid.shape)

    return x_grid, t_grid, u_pred

#%%
# Calcular valores da função e gerar grids
x_grid, t_grid, u_pred = calcular_grid(rna, comprimento_x, tempo_final)

#%% [markdown]
# Treinamento completo do modelo

#%%
numero_de_epocas = 20000
perda_historico = np.zeros(numero_de_epocas)
perda_contorno_historico = np.zeros(numero_de_epocas)
perda_equacao_historico = np.zeros(numero_de_epocas)
epocas = np.array(range(numero_de_epocas))

# Colocar rede em modo de treinamento
rna.train()

# FAZER ITERAÇÃO
for epoca in epocas:

  # Resortear pontos
  #X_equacao = gerar_pontos_equacao(pontos_no_dominio,comprimento_x,tempo_final)
  #X_equacao = torch.tensor(X_equacao,requires_grad=True,dtype=torch.float).to(device)

  # Inicializar gradientes
  otimizador.zero_grad()

  # Calcular perdas
  perda, perda_contorno, perda_equacao = calc_perda(rna,X_contorno,Y_contorno,X_equacao,alpha=alpha)

  # Backpropagation
  perda.backward()

  # Passo do otimizador
  otimizador.step()
  agendador.step()

  # Guardar logs
  perda_historico[epoca] = perda.item()
  perda_contorno_historico[epoca] = perda_contorno.item()
  perda_equacao_historico[epoca] = perda_equacao.item()

  if epoca%500==0:
    print(f'Epoca: {epoca}, Perda: {perda.item()} (Contorno: {perda_contorno.item()}, Equacao: {perda_equacao.item()})')

#%% [markdown]
# # Visualização dos resultados

#%%
# Calcular valores da função e gerar grids
x_grid, y_grid, z_pred = calcular_grid(rna, comprimento_x, tempo_final)

# Plotar figura
fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=z_pred)])
fig.update_layout(scene=dict(aspectratio=dict(x=1.5, y=1.5, z=0.5)))
fig.show()

#%% [markdown]
# Plotar histórico de perdas

#%%
# Plotar histórico
fig = go.Figure()
fig.add_trace(go.Scatter(x=epocas, y=perda_historico, name='Total', line=dict(color='black', width=4)))
fig.add_trace(go.Scatter(x=epocas, y=perda_contorno_historico, name='Contorno', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=epocas, y=perda_equacao_historico, name='Equacao', line=dict(color='red', width=2)))
fig.update_yaxes(type="log")
fig.show()


# %%

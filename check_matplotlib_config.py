import matplotlib
print("Diretório de configuração do matplotlib:")
print(matplotlib.get_configdir())
print("\nArquivo matplotlibrc atual:")
print(matplotlib.matplotlib_fname())

# Para IPython
import IPython
print("\nDiretório do IPython:")
print(IPython.paths.get_ipython_dir())

# Para ver o backend atual
print("\nBackend atual do matplotlib:")
print(matplotlib.get_backend())
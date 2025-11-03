"""Script para verificar configuração CUDA do PyTorch"""
import torch
import sys

print("=" * 60)
print("VERIFICAÇÃO DE CUDA NO PYTORCH")
print("=" * 60)

# Versão do Python
print(f"\nPython version: {sys.version}")

# Versão do PyTorch
print(f"PyTorch version: {torch.__version__}")

# Verificar se CUDA está disponível
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Informações da GPU
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Testar criação de tensor na GPU
    print("\n" + "=" * 60)
    print("TESTE DE CRIAÇÃO DE TENSOR NA GPU")
    print("=" * 60)
    try:
        device = torch.device('cuda')
        x = torch.randn(3, 3).to(device)
        print(f"✓ Tensor criado na GPU com sucesso!")
        print(f"  Tensor device: {x.device}")
        print(f"  Tensor shape: {x.shape}")
        
        # Teste de operação
        y = torch.matmul(x, x)
        print(f"✓ Operação matemática executada na GPU!")
        
    except Exception as e:
        print(f"✗ Erro ao criar tensor na GPU: {e}")
else:
    print("\n" + "=" * 60)
    print("CUDA NÃO DISPONÍVEL")
    print("=" * 60)
    print("\nPossíveis causas:")
    print("1. PyTorch foi instalado sem suporte CUDA (versão CPU-only)")
    print("2. Drivers NVIDIA não estão instalados ou atualizados")
    print("3. CUDA Toolkit não está instalado ou compatível")
    print("\nSolução:")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)


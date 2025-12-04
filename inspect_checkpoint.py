import torch

c = torch.load('checkpoints/inverse_pinn/inverse_pinn_latest.pth', weights_only=False)

print("Keys no checkpoint:", list(c.keys()))
print("\nConfig:")
for k, v in c['config'].items():
    print(f"  {k}: {v}")

print("\nModel state keys:")
for k in c['model_state'].keys():
    print(f"  {k}: {c['model_state'][k].shape}")

print(f"\nBest loss: {c.get('best_loss', 'N/A')}")
print(f"Total epochs: {c.get('total_epochs', 'N/A')}")


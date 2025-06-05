import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

###############################################################################
# Reproducibility
###############################################################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

###############################################################################
# Dataset – noisy sine wave
###############################################################################

def generate_sine_data(n: int, x_min: float, x_max: float, sigma: float = 0.05):
    """Return tensors x ∈ [x_min,x_max] and noisy y = sin(5x) + ε."""
    x = torch.linspace(x_min, x_max, n).unsqueeze(1)
    y_clean = torch.sin(5 * x)
    y = y_clean + torch.randn_like(y_clean) * sigma
    return x, y

###############################################################################
# Model with Kaiming init and named ReLU layers so we can install hooks easily
###############################################################################

class DebugMLP(nn.Module):
    def __init__(self, hidden_dim: int = 50):
        super().__init__()
        self.lin1 = nn.Linear(1, hidden_dim)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, 1)

        # Proper Kaiming init for ReLU layers
        for layer in (self.lin1, self.lin2):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        return self.lin3(x)

###############################################################################
# Hook utilities – collect activation stats to detect "dead" ReLUs
###############################################################################

def dead_relu_ratio(tensor: torch.Tensor):
    """Return fraction of elements that are exactly 0."""
    return (tensor == 0).float().mean().item()

class ActivationMonitor:
    """Keeps running stats of each hooked activation layer."""

    def __init__(self):
        self.stats = {}

    def make_hook(self, name):
        def hook(module, inputs, outputs):
            self.stats[name] = {
                "mean": outputs.mean().item(),
                "std": outputs.std().item(),
                "dead_frac": dead_relu_ratio(outputs),
            }
        return hook

###############################################################################
# Training with detailed logging
###############################################################################

def train_debug():
    # Hyper‑parameters
    N = 1000
    sigma = 0.05
    x_min, x_max = -math.pi, math.pi
    hidden_dim = 50
    lr = 1e-3
    epochs = 2000
    batch_size = 128

    # Data (scale x to [-1,1] for better conditioning)
    x_raw, y = generate_sine_data(N, x_min, x_max, sigma)
    x = x_raw / math.pi
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = DebugMLP(hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Hook setup
    mon = ActivationMonitor()
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(mon.make_hook(name)))

    # Storage for curves
    loss_hist, grad_hist, dead_hist = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        loss_hist.append(epoch_loss)

        # ---- gradient norm per epoch ----
        total_grad_norm = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = math.sqrt(total_grad_norm)
        grad_hist.append(total_grad_norm)

        # ---- dead ReLU metrics ----
        dead_hist.append({k: v["dead_frac"] for k, v in mon.stats.items()})

        # ---- periodic console report ----
        if epoch % 100 == 0 or epoch == 1:
            dead_report = ", ".join(
                f"{k}: {v*100:.1f}% dead" for k, v in dead_hist[-1].items()
            )
            print(
                f"Epoch {epoch:4d} | loss={epoch_loss:.5f} | grad‖·‖={total_grad_norm:.3e} | {dead_report}"
            )

    # Remove hooks to avoid memory leaks
    for h in hooks:
        h.remove()

    # ======================= plots =======================
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.semilogy(loss_hist)
    plt.title("Training loss (MSE)")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.semilogy(grad_hist)
    plt.title("Total gradient norm")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for name in dead_hist[0].keys():
        plt.plot([d[name] for d in dead_hist], label=name)
    plt.title("Dead ReLU fraction per layer")
    plt.xlabel("Epoch")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final fit plot
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(x_min, x_max, 1000).unsqueeze(1)
        x_scaled = x_plot / math.pi
        y_plot_pred = model(x_scaled)
        y_true = torch.sin(5 * x_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_raw.numpy(), y.numpy(), s=10, alpha=0.3, label="Noisy data")
    plt.plot(x_plot.numpy(), y_true.numpy(), label="True sin(5x)")
    plt.plot(x_plot.numpy(), y_plot_pred.numpy(), "--", label="MLP prediction")
    plt.legend()
    plt.grid(True)
    plt.title("DebugMLP fit of sin(5x)")
    plt.show()

if __name__ == "__main__":
    train_debug()

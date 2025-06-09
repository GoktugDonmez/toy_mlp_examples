#!/usr/bin/env python3
"""
Deep Ensemble Uncertainty Demo
==============================

• Constructs a piece‑wise dataset with a deliberate gap (x ∈ (2, 3)).
• Trains an ensemble of 10 independent MLP regressors (different seeds).
• Visualises individual ensemble functions, the ensemble mean,
  and ±2 σ epistemic uncertainty together with the training data.

Dependencies: math, random, numpy, torch, matplotlib (CPU‑only by default, but
will use CUDA if available). Run this file directly:

    python deep_ensemble_uncertainty_demo.py

"""
import math
import random
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ==================== Configuration ====================
SEEDS: List[int] = list(range(10))        # 10 models → deep ensemble
N_EPOCHS = 300                          # Training iterations per model
LR = 0.001                                 # Adam learning‑rate
BATCH_SIZE = 32                           # SGD mini‑batch size
HIDDEN = [64, 64]                         # Two hidden layers → width 64
NOISE_STD = 0.10                          # σ of i.i.d. Gaussian label noise
N_PER_SEGMENT = 300                      # Number of points in each segment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Data generation ====================

def generate_data(n_per_segment: int = N_PER_SEGMENT,
                   noise_std: float = NOISE_STD,
                   seed: int = 0):
    """Generate training data for the two piece‑wise quadratic/linear curves."""
    rng = np.random.default_rng(seed)

    # Segment A: 0 ≤ x ≤ 2  
    x_a = rng.uniform(0, 2, size=(n_per_segment,))
    y_a = 2 * x_a ** 2 -0.5

    # Segment B: 2.5 ≤ x ≤ 5  
    x_b = rng.uniform(2.5, 5, size=(n_per_segment,))
    y_b = -3 * (x_b - 4) + 4

    # Optional homoscedastic noise for realism
    y_a += rng.normal(0, noise_std, size=y_a.shape)
    y_b += rng.normal(0, noise_std, size=y_b.shape)

    # Concatenate and shuffle
    x = np.concatenate([x_a, x_b]).astype(np.float32)
    y = np.concatenate([y_a, y_b]).astype(np.float32)
    idx = rng.permutation(len(x))
    return x[idx].reshape(-1, 1), y[idx].reshape(-1, 1)

# ==================== Model definition ====================

class MLP(nn.Module):
    """Fully‑connected feed‑forward neural network (scalar input/output)."""

    def __init__(self, hidden_layers: List[int] = HIDDEN):
        super().__init__()
        layers = []
        in_dim = 1  # scalar x
        for width in hidden_layers:
            layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))  # scalar ŷ
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x.shape = (batch, 1)
        return self.net(x)

# ==================== Training routine ====================

def train_single_seed(seed: int, x_train: np.ndarray, y_train: np.ndarray) -> nn.Module:
    """Train one MLP with a fixed PRNG seed and return the fitted model."""
    # ♥ Keep every RNG in sync for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MLP().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(N_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
    return model.eval()  # switch to eval‑mode for inference

# ==================== Main script ====================

def main():
    # 1. Assemble dataset
    x_train, y_train = generate_data()

    # 2. Fit ensemble
    ensemble = []
    for s in SEEDS:
        print(f"Training model with seed {s} …")
        ensemble.append(train_single_seed(s, x_train, y_train))

    # 3. Prepare evaluation grid
    x_grid = np.linspace(0, 5, 500, dtype=np.float32).reshape(-1, 1)
    x_grid_t = torch.from_numpy(x_grid).to(DEVICE)

    with torch.no_grad():
        # Stack predictions: shape = (ensemble_size, n_points, 1)
        preds = torch.stack([m(x_grid_t) for m in ensemble]).cpu().numpy()
    mean = preds.mean(axis=0).squeeze()
    std = preds.std(axis=0).squeeze()

    # 4. Visualisation
    plt.figure(figsize=(9, 5))

    # (a) Individual ensemble members (faint)
    for i, y_hat in enumerate(preds):
        plt.plot(x_grid, y_hat.squeeze(), alpha=0.25, lw=1.2)

    # (b) Ensemble mean + 2σ band
    plt.plot(x_grid, mean, lw=2.0, label="Ensemble mean", zorder=3)
    plt.fill_between(x_grid.squeeze(), mean - 2 * std, mean + 2 * std,
                     alpha=0.20, label="±2 σ epistemic")

    # (c) Training data
    plt.scatter(x_train, y_train, s=14, c="red", label="Training data", zorder=4)

    plt.title("Deep Ensemble Regression with Uncertainty (gap at 2.5 ≤ x ≤ 3)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ensemble_plot.png')
    print("Plot has been saved as 'ensemble_plot.png'")


if __name__ == "__main__":
    main()

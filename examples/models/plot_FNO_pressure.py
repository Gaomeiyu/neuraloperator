"""
Pressure-Time Prediction with FNO
==================================

Use a 1-D Fourier Neural Operator to predict wellbore pressure-time curves.

Problem Description
-------------------
The well-testing PDE in dimensionless form:

    ∂²pD/∂rD² + (1/rD)(∂pD/∂rD) = ∂pD/∂tD

    Initial condition:   pD(rD, 0) = 0
    Outer boundary:      pD(∞, tD) = 0
    Inner boundaries:
        CD·dpwD/dtD − (∂pD/∂rD)|_{rD=1} = 1     (wellbore storage)
        pwD = [pD − S·(∂pD/∂rD)]|_{rD=1}         (skin effect)

All cases share the **same initial condition**; only the wellbore storage
coefficient **CD** and skin factor **S** differ.  Because of this, the
early-time pressure curves look identical and then diverge as the boundary
effects become dominant.

Approach
--------
1. **Data generation**: For many (CD, S) pairs, solve the PDE via the
   Laplace-domain analytical solution (Stehfest numerical inversion) to
   obtain pwD(tD) on a log-spaced time grid.

2. **Input / Output split**: For each curve, take the first ``n_input``
   time steps (where curves begin to diverge) as the *input function* and
   the subsequent ``n_predict`` time steps as the *target function*.

3. **FNO mapping**: Train a 1-D FNO that learns the operator

       F : pwD(tD)_{early} → pwD(tD)_{later}

   The Fourier layers capture the spectral structure of the pressure curves,
   enabling the model to generalize across unseen (CD, S) combinations.

4. **Evaluation**: Compare model predictions against the analytical
   solution on held-out (CD, S) combinations.
"""

# %%
# Import dependencies
# -------------------

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuralop.models import FNO
from neuralop.utils import count_model_params
from neuralop import LpLoss

# Allow importing the data generator from examples/data_gen
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_gen")
)
from welltest_pressure import generate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Configurable parameters
# -----------------------

# Time grid settings
N_TIME_POINTS = 256  # total time grid points
TD_MIN = 0.01  # minimum dimensionless time
TD_MAX = 1e6  # maximum dimensionless time

# Input / output split
N_INPUT = 64  # number of early time steps used as input
N_PREDICT = 192  # number of later time steps to predict (N_TIME_POINTS - N_INPUT)

# Ranges of physical parameters for data generation (log-spaced for CD)
CD_VALUES = np.logspace(-2, 4, 13)  # 0.01 to 10000, log-uniform coverage
S_VALUES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])

# Training parameters
N_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# FNO architecture
HIDDEN_CHANNELS = 64
N_MODES = (32,)  # 1-D Fourier modes
N_LAYERS = 4

# %%
# Generate dataset
# ----------------
# Solve the wellbore storage + skin PDE for every (CD, S) combination.

print("Generating pressure data...")
sys.stdout.flush()

td_array, pressures, params = generate_dataset(
    cd_values=CD_VALUES,
    s_values=S_VALUES,
    n_time_points=N_TIME_POINTS,
    td_min=TD_MIN,
    td_max=TD_MAX,
)
n_samples = pressures.shape[0]
print(f"Generated {n_samples} pressure curves with {N_TIME_POINTS} time points each.")
sys.stdout.flush()

# %%
# Prepare train / test split
# ---------------------------
# Normalize the pressure curves and split into input (early) and target (later).

# Normalize to [0, 1] for numerical stability
p_min = pressures.min()
p_max = pressures.max()
pressures_norm = (pressures - p_min) / (p_max - p_min + 1e-10)

# Split each curve into input and output segments
x_data = pressures_norm[:, :N_INPUT]  # shape: (n_samples, N_INPUT)
y_data = pressures_norm[:, N_INPUT:]  # shape: (n_samples, N_PREDICT)

# Add channel dimension: (n_samples, 1, n_points)
x_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

# Random train / test split (80/20) with fixed seed for reproducibility
torch.manual_seed(42)
n_train = int(0.8 * n_samples)
indices = torch.randperm(n_samples)
train_idx, test_idx = indices[:n_train], indices[n_train:]

x_train, y_train = x_data[train_idx], y_data[train_idx]
x_test, y_test = x_data[test_idx], y_data[test_idx]

print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
print(f"Input shape:  {x_train.shape}  (batch, channels, time_steps)")
print(f"Target shape: {y_train.shape}")
sys.stdout.flush()

# Build DataLoaders using dict-style datasets (compatible with neuralop Trainer)
from torch.utils.data import DataLoader, Dataset


class WelltestDataset(Dataset):
    """Simple dataset wrapping input/output pressure tensors."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


train_dataset = WelltestDataset(x_train, y_train)
test_dataset = WelltestDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# Build the 1-D FNO model
# ------------------------
# The FNO maps the input pressure function (early time) to the output
# pressure function (later time).  Because input and output live on
# grids of *different* lengths, we project the FNO output to the target
# length with a linear layer.


class PressureFNO(nn.Module):
    """FNO-based model for 1-D pressure curve prediction.

    Wraps a 1-D FNO with an output projection so that the input
    and output can have different temporal resolutions.

    Parameters
    ----------
    n_modes : tuple
        Fourier modes to keep.
    hidden_channels : int
        FNO hidden width.
    n_layers : int
        Number of Fourier layers.
    n_input : int
        Number of input time steps.
    n_predict : int
        Number of output time steps to predict.
    """

    def __init__(self, n_modes, hidden_channels, n_layers, n_input, n_predict):
        super().__init__()
        self.n_input = n_input
        self.n_predict = n_predict

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=1,
            out_channels=1,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid",
        )
        # Map from input length to output length
        self.output_proj = nn.Linear(n_input, n_predict)

    def forward(self, x, **kwargs):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 1, n_input)

        Returns
        -------
        Tensor of shape (batch, 1, n_predict)
        """
        # FNO forward: (B, 1, n_input) -> (B, 1, n_input)
        out = self.fno(x)
        # Project temporal dimension: (B, 1, n_input) -> (B, 1, n_predict)
        out = self.output_proj(out)
        return out


model = PressureFNO(
    n_modes=N_MODES,
    hidden_channels=HIDDEN_CHANNELS,
    n_layers=N_LAYERS,
    n_input=N_INPUT,
    n_predict=N_PREDICT,
).to(device)

n_params = count_model_params(model)
print(f"\nPressureFNO model has {n_params} parameters.")
sys.stdout.flush()

# %%
# Training
# --------

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# MSE for training; relative L2 for final evaluation (function-space metric)
l2_loss = LpLoss(d=1, p=2)
mse_loss = nn.MSELoss()

print("\n--- Training ---")
sys.stdout.flush()

train_losses = []
test_losses = []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        loss = mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * x.size(0)

    scheduler.step()
    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)

    # Evaluate every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                test_loss += mse_loss(pred, y).item() * x.size(0)
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d}/{N_EPOCHS} | "
            f"Train MSE: {epoch_loss:.6f} | "
            f"Test MSE: {test_loss:.6f} | "
            f"LR: {lr:.6f}"
        )
        sys.stdout.flush()

# %%
# Plot training curve
# -------------------

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, N_EPOCHS + 1), train_losses, label="Train MSE")
eval_epochs = [1] + list(range(10, N_EPOCHS + 1, 10))
ax.plot(eval_epochs[: len(test_losses)], test_losses, "o-", label="Test MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_yscale("log")
ax.set_title("Training Curve – Pressure-Time FNO")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pressure_fno_training_curve.png", dpi=150)
print("\nSaved pressure_fno_training_curve.png")
sys.stdout.flush()

# %%
# Visualize predictions
# ---------------------
# Show a few test-set predictions alongside the analytical solution.

model.eval()
td_input = td_array[:N_INPUT]
td_output = td_array[N_INPUT:]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Pick up to 6 test samples
n_show = min(6, len(test_idx))
for i in range(n_show):
    idx = test_idx[i].item()
    cd_val, s_val = params[idx]

    x_sample = x_data[idx].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_sample).cpu().squeeze().numpy()

    # Denormalize
    pred_denorm = pred * (p_max - p_min) + p_min
    true_input = pressures[idx, :N_INPUT]
    true_output = pressures[idx, N_INPUT:]

    ax = axes[i]
    ax.semilogx(td_input, true_input, "b-", linewidth=1.5, label="Input (given)")
    ax.semilogx(td_output, true_output, "g-", linewidth=1.5, label="True (target)")
    ax.semilogx(td_output, pred_denorm, "r--", linewidth=1.5, label="FNO prediction")
    ax.axvline(x=td_array[N_INPUT - 1], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"CD={cd_val:.2f}, S={s_val:.1f}", fontsize=10)
    ax.set_xlabel("tD")
    ax.set_ylabel("pwD")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle(
    "FNO Pressure-Time Predictions\n"
    "(blue = input early time steps, green = true curve, red dashed = FNO prediction)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("pressure_fno_predictions.png", dpi=150)
print("Saved pressure_fno_predictions.png")
sys.stdout.flush()

# %%
# Quantitative evaluation
# -----------------------

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

rel_l2 = l2_loss(all_preds, all_targets).item()
abs_mse = mse_loss(all_preds, all_targets).item()

print(f"\n--- Test Set Results ---")
print(f"Relative L2 error: {rel_l2:.6f}")
print(f"Absolute MSE:      {abs_mse:.6f}")
sys.stdout.flush()

# %%
# Summary of approach
# -------------------
#
# .. note::
#    **Approach overview:**
#
#    1. The well-testing PDE has a known analytical solution in the Laplace
#       domain. We use Stehfest numerical inversion to generate pwD(tD)
#       curves for many (CD, S) combinations.
#
#    2. All curves share the same initial condition pD(rD,0)=0, so the
#       early-time behavior is similar.  As time increases, different
#       (CD, S) values cause the curves to diverge.
#
#    3. We split each curve into an *input segment* (first ``N_INPUT`` steps
#       where divergence starts) and a *target segment* (remaining
#       ``N_PREDICT`` steps).
#
#    4. A 1-D FNO learns the operator mapping from the early pressure
#       function to the later pressure function.  The Fourier layers
#       capture global spectral features of the curves, while the
#       output projection adapts to the different temporal resolution.
#
#    5. At inference, given only the first few time steps of a new
#       (CD, S) curve, the trained model predicts the full later evolution.

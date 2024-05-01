import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ode import AdaGradODE
from sgd import AdaGrad

plt.rcParams['font.size'] = 20
# Parameters
e_var = 0
b_0 = 1
# e_var = 1
# b_0 = 0.3
G = 1
T = 10
num_runs = 30  # Number of stochastic runs for AdaGrad per dimension
# kind = "least squares"  # Kind of optimization problem
kind = "logistic regression"  # Kind of optimization problem
dimensions = [2**i for i in [5, 7, 9, 11]]
# dimensions = [2**i for i in [8, 10, 12, 14]]

# Cache directory setup
cache_dir = "simulation_cache"
os.makedirs(cache_dir, exist_ok=True)

# Initialize storage for losses
all_sgd_losses = {}
all_sgd_stepsizes = {}

# Fixed eigenvalues for all simulations
# Use the largest dimension to set all eigenvalues to 1
fixed_eigs = np.ones(max(dimensions))

# Single ODE run with fixed eigenvalues
ode = AdaGradODE(kind, G=G, b_0=b_0, e_var=e_var)
X_ = np.random.normal(size=max(dimensions))
X_ = X_ / np.sqrt((X_ @ X_))
ode_losses, ode_stepsizes = ode.get_losses(fixed_eigs, T=T, X_=X_)

# Simulation loop for AdaGrad across dimensions
for d in dimensions:
    cache_path = os.path.join(
        cache_dir, f"sgd_losses_kind={kind}_d={d}_e_var={e_var}_b_0={b_0}_G={G}_T={T}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            sgd_losses, sgd_stepsizes = pickle.load(f)
    else:
        sgd_losses = []
        sgd_stepsizes = []
        for run in range(num_runs):
            X_ = np.random.normal(size=d)
            X_ = X_ / np.sqrt((X_ @ X_))
            sgd = AdaGrad(kind, G=G, b_0=b_0, e_var=e_var)
            losses, stepsizes = sgd.get_losses(fixed_eigs[:d], T=T)
            sgd_losses.append(losses)
            sgd_stepsizes.append(stepsizes)
        with open(cache_path, "wb") as f:
            both = (sgd_losses, sgd_stepsizes)
            pickle.dump(both, f)

    all_sgd_losses[d] = sgd_losses
    all_sgd_stepsizes[d] = sgd_stepsizes

fig, ax1 = plt.subplots(figsize=(12, 8))
colors = plt.cm.plasma(np.linspace(0, 1, len(dimensions)))

for idx, d in enumerate(dimensions):
    sgd_losses = np.array(all_sgd_losses[d])
    sgd_lower = np.percentile(sgd_losses, 10, axis=0)
    sgd_upper = np.percentile(sgd_losses, 90, axis=0)
    ax1.fill_between(np.linspace(0, T, len(sgd_upper)), sgd_lower, sgd_upper,
                     color=colors[idx], label=f"$d={d}$")

    sgd_stepsizes = np.array(all_sgd_stepsizes[d])
    sgd_lower = np.percentile(sgd_stepsizes, 10, axis=0)
    sgd_upper = np.percentile(sgd_stepsizes, 90, axis=0)

ax2 = ax1.twinx()
for idx, d in enumerate(dimensions):
    sgd_stepsizes = np.array(all_sgd_stepsizes[d])
    sgd_lower = np.percentile(sgd_stepsizes, 10, axis=0)
    sgd_upper = np.percentile(sgd_stepsizes, 90, axis=0)
    ax2.fill_between(np.linspace(0, T, len(sgd_upper)), sgd_lower, sgd_upper,
                     color=colors[idx], label=f"$d={d}$")

ax1.loglog(np.linspace(0, T, len(ode_losses)), ode_losses, color='red',
           linewidth=2, label="Predicted stepsize", linestyle="dotted")
ax1.loglog(np.linspace(0, T, len(ode_losses)), ode_losses, color='red',
           linewidth=2, label="Predicted loss")

ax2.loglog(np.linspace(0, T, len(ode_stepsizes)), ode_stepsizes, color='red',
           linewidth=2, linestyle="dotted")

# Add grid lines
ax1.grid(True, linestyle='--', alpha=0.5)

ax1.set_title("AdaGrad-Norm Logistic Regression", fontsize=25)
# ax1.set_title("AdaGrad-Norm Least Squares", fontsize=25)
ax1.set_xlabel("SGD Iterations/$d$")
ax1.set_ylabel("Loss")
ax1.legend(loc="lower left")
# ax1.legend()
ax1.set_xlim(left=0.1)
ax1.set_ylim(top=0.75)
ax2.set_ylabel("Stepsize")
plt.tight_layout()
plt.savefig(
    f"plots/sgd_concentration_vs_dimension_{kind.replace(' ', '_')}.png", dpi=1000)
plt.show()

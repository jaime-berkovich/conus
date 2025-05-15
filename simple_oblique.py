import numpy as np
import matplotlib.pyplot as plt
from actinh import ActivatorInhibitor  # updated name
import os

# ─────────────────────────────────────────────────────
# Grid setup
# ─────────────────────────────────────────────────────
N = 200
dx = 1.0
dt = 0.01
steps = 20000

# ─────────────────────────────────────────────────────
# Initial conditions: mostly zero, with spikes in A
# ─────────────────────────────────────────────────────
A = np.zeros(N)
B = np.ones(N)*0.5

# introduce a few random high‐concentration spikes in A
num_spikes = 10
spike_indices = np.random.choice(N, size=num_spikes, replace=False)
A[spike_indices] = 1.0  # spike height

initial_conditions = {'A': A, 'B': B}

# ─────────────────────────────────────────────────────
# Initialize parameters: high b_b
# ─────────────────────────────────────────────────────
parameters = {
    's':    1.0,     # source strength
    'b_a':  0.00,    # no basic activator production
    'r_a':  0.1,    # slow activator decay
    'r_b':  0.1,     # slow inhibitor decay
    'b_b':  0.1      # <<-- slow inhibitor production
}

# ─────────────────────────────────────────────────────
# Diffusion rates: no inhibitor diffusion
# ─────────────────────────────────────────────────────
diffusion_rates = {
    'A': 0.1,    # slow activator diffusion
    'B': 0.0       # <<-- no diffusion for inhibitor
}

# ─────────────────────────────────────────────────────
# Initialize model
# ─────────────────────────────────────────────────────
model = ActivatorInhibitor(
    species_names=['A', 'B'],
    initial_conditions=initial_conditions,
    diffusion_rates=diffusion_rates,
    parameters=parameters
)

# ─────────────────────────────────────────────────────
# Run simulation and record
# ─────────────────────────────────────────────────────
record_A = np.zeros((steps, N))
record_B = np.zeros((steps, N))

for t in range(steps):
    model.step(dx, dt)
    state = model.get_state()
    record_A[t] = state['A']
    record_B[t] = state['B']

# ─────────────────────────────────────────────────────
# Prepare output folder
# ─────────────────────────────────────────────────────
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)

# ─────────────────────────────────────────────────────
# Plot spatiotemporal concentrations
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

im0 = axes[0].imshow(
    record_A, aspect='auto', cmap='viridis',
    extent=[0, N, steps*dt, 0]
)
axes[0].set_ylabel("Time")
axes[0].set_title("Activator A")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    record_B, aspect='auto', cmap='magma',
    extent=[0, N, steps*dt, 0]
)
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Time")
axes[1].set_title("Inhibitor B")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()

# ─────────────────────────────────────────────────────
# Add parameters text at the bottom
# ─────────────────────────────────────────────────────
param_text = (
    "Parameters: " +
    ", ".join(f"{k}={v}" for k, v in parameters.items()) +
    "\nDiffusion Rates: " +
    ", ".join(f"D_{k}={v}" for k, v in diffusion_rates.items())
)
fig.subplots_adjust(bottom=0.25)
fig.text(0.3, 0.05, param_text, ha='center', va='bottom', fontsize=9, wrap=True)

# ─────────────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────────────
output_path = os.path.join(output_folder, "spatiotemporal_concentrations_oblique.png")
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
# plt.show()  # not needed when saving
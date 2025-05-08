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
steps = 5000

# ─────────────────────────────────────────────────────
# Initial conditions
# ─────────────────────────────────────────────────────
A = np.random.rand(N) * 0.5
B = np.zeros(N) # no inhibitor to start
B = np.random.rand(N) * 0.5


# # Perturb center
# A[N//2 - 5:N//2 + 5] += 0.2 * np.random.randn(1)
# B[N//2 - 5:N//2 + 5] += 0.2 * np.random.randn(1)

initial_conditions = {'A': A, 'B': B}

# ─────────────────────────────────────────────────────
# Initialize parameters
# ─────────────────────────────────────────────────────
parameters = {
    's':    1,     # source strength
    'b_a':  0.05,    # baseline activator production
    'r_a':  0.01,    # activator decay
    'r_b':  0.1,     # inhibitor decay (>> s * b_a = 0.025)
    'b_b':  0.005    # baseline inhibitor production
}

diffusion_rates = {
    'A': 0.001,      # slow activator
    'B': 1        # fast inhibitor (D_B >> D_A)
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
# Run simulation and store time evolution
# ─────────────────────────────────────────────────────
record_A = np.zeros((steps, N))
record_B = np.zeros((steps, N))

for t in range(steps):
    model.step(dx, dt)
    state = model.get_state()
    record_A[t] = state['A']
    record_B[t] = state['B']

# ─────────────────────────────────────────────────────
# Create output folder if it doesn't exist
# ─────────────────────────────────────────────────────
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)  # creates folder if not present

# ─────────────────────────────────────────────────────
# Plot: x = position, y = time, color = concentration
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

im0 = axes[0].imshow(record_A, aspect='auto', cmap='viridis',
                     extent=[0, N, steps*dt, 0])
axes[0].set_ylabel("Time")
axes[0].set_title("Activator A")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(record_B, aspect='auto', cmap='magma',
                     extent=[0, N, steps*dt, 0])
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Time")
axes[1].set_title("Inhibitor B")
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
# ─────────────────────────────────────────────────────
# Add parameter text at the bottom of the figure
# ─────────────────────────────────────────────────────
param_text = "Parameters: " + ', '.join([f"{k}={v}" for k, v in parameters.items()]) + "\n" + \
             "Diffusion Rates: " + ', '.join([f"D_{k}={v}" for k, v in diffusion_rates.items()])

fig.subplots_adjust(bottom=0.25)  # leave space at bottom
fig.text(0.5, 0.02, param_text, ha='center', va='bottom', fontsize=9, wrap=True)

# ─────────────────────────────────────────────────────
# Save the figure to file
# ─────────────────────────────────────────────────────
output_path = os.path.join(output_folder, "spatiotemporal_concentrations_turing.png")
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")

# plt.show() <--- don't show in this case
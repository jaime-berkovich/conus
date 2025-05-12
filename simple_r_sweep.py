import numpy as np
import matplotlib.pyplot as plt
from actinh import ActivatorInhibitor
import os

# ─────────────────────────────────────────────────────
# Global settings
# ─────────────────────────────────────────────────────
N      = 200
steps  = 5000
dx     = 1.0
dt     = 0.01
rng    = np.random.default_rng(42)

# ─────────────────────────────────────────────────────
# Baseline parameters (all fixed except r_b)
# ─────────────────────────────────────────────────────
baseline_params = dict(
    s   = 1.0,
    b_a = 0.05,
    r_a = 0.01,
    b_b = 0.005               # constant inhibitor input
)
D = dict(A=0.001, B=1.0)      # Turing-friendly diffusion contrast

# critical threshold from your analysis: r_b > s * b_a
r_crit    = baseline_params['s'] * baseline_params['b_a']
r_factors = [0.5, 1.0, 2.0]   # <1: sub-critical, 1: critical, >1: super-critical
r_values  = [f * r_crit for f in r_factors]

# ─────────────────────────────────────────────────────
# Build one scenario per r_b value
# ─────────────────────────────────────────────────────
scenarios = []
for r in r_values:
    params = baseline_params | {'r_b': r}
    init   = dict(
        A = rng.random(N) * 0.5,
        B = rng.random(N) * 0.5
    )
    scenarios.append(dict(
        dx=dx, dt=dt,
        initial_conditions=init,
        parameters=params,
        diffusion_rates=D
    ))

# ─────────────────────────────────────────────────────
# Simulate each scenario
# ─────────────────────────────────────────────────────
records = []
for scen in scenarios:
    model = ActivatorInhibitor(
        species_names=['A', 'B'],
        initial_conditions=scen['initial_conditions'],
        diffusion_rates=scen['diffusion_rates'],
        parameters=scen['parameters']
    )
    recA = np.zeros((steps, N))
    recB = np.zeros((steps, N))
    for t in range(steps):
        model.step(scen['dx'], scen['dt'])
        st = model.get_state()
        recA[t], recB[t] = st['A'], st['B']
    records.append((recA, recB))

# ─────────────────────────────────────────────────────
# Plot 2 rows × len(r_values) columns
# ─────────────────────────────────────────────────────
ncols = len(r_values)
fig, axes = plt.subplots(
    nrows=2, ncols=ncols,
    figsize=(6 * ncols, 8),
    sharex=False, sharey=False
)
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.05, right=0.98,
                    wspace=0.25, hspace=0.4)

for i, ((recA, recB), scen) in enumerate(zip(records, scenarios)):
    axA, axB = axes[0, i], axes[1, i]

    # Activator heatmap
    imA = axA.imshow(recA, aspect='auto', cmap='viridis',
                     extent=[0, N, steps*scen['dt'], 0])
    axA.set_title("Activator A")
    if i == 0: axA.set_ylabel("Time")
    if i == ncols - 1: axA.set_xlabel("Position")
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

    # Inhibitor heatmap
    imB = axB.imshow(recB, aspect='auto', cmap='magma',
                     extent=[0, N, steps*scen['dt'], 0])
    axB.set_title("Inhibitor B")
    if i == 0: axB.set_ylabel("Time")
    if i == ncols - 1: axB.set_xlabel("Position")
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    # Sub-figure tag
    axA.text(0, 1.02, f"{chr(97+i)})", transform=axA.transAxes,
             fontweight='bold', fontsize=16, va='bottom')

    # Parameter block
    param_str = (
        f"r_b = {scen['parameters']['r_b']:.4f}\n"
        f"(r_b / r_crit = {r_factors[i]:.1f})\n"
        f"D_A = {D['A']}, D_B = {D['B']}"
    )
    axB.text(0.5, -0.35, param_str, transform=axB.transAxes,
             ha='center', va='top', fontsize=8)

# ─────────────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────────────
out_dir  = "output_plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "r_b_sweep.png")
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")

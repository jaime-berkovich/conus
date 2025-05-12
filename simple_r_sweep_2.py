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
    b_b = 0.005
)
D = dict(A=0.001, B=1.0)       # diffusion contrast

# critical threshold r_crit = s * b_a
r_crit    = baseline_params['s'] * baseline_params['b_a']
r_factors = [0.5, 1.0, 2.0]    # sub-, crit-, super-critical
r_values  = [f * r_crit for f in r_factors]

# ─────────────────────────────────────────────────────
# Assemble scenarios
# ─────────────────────────────────────────────────────
scenarios = []
for r in r_values:
    params = baseline_params | {'r_b': r}
    init   = dict(A=rng.random(N) * 0.5,
                  B=rng.random(N) * 0.5)
    scenarios.append(dict(
        dx=dx, dt=dt,
        initial_conditions=init,
        parameters=params,
        diffusion_rates=D
    ))

# ─────────────────────────────────────────────────────
# Simulate
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
    avgA = np.zeros(steps)
    avgB = np.zeros(steps)
    varA = np.zeros(steps)
    varB = np.zeros(steps)

    for t in range(steps):
        model.step(scen['dx'], scen['dt'])
        st = model.get_state()
        recA[t], recB[t] = st['A'], st['B']

        avgA[t] = recA[t].mean()
        avgB[t] = recB[t].mean()
        varA[t] = recA[t].var()
        varB[t] = recB[t].var()

    records.append((recA, recB, avgA, avgB, varA, varB))

# ─────────────────────────────────────────────────────
# Plot grid: 4 rows × len(r_values) columns
# ─────────────────────────────────────────────────────
ncols = len(r_values)
fig, axes = plt.subplots(
    nrows=4, ncols=ncols,
    figsize=(6 * ncols, 12),
    sharex=False, sharey=False
)
plt.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.98,
                    wspace=0.35, hspace=0.55)

time = np.arange(steps) * dt

for i, ((recA, recB, avgA, avgB, varA, varB), scen) in enumerate(zip(records, scenarios)):
    axA, axB   = axes[0, i], axes[1, i]
    axMeanA    = axes[2, i]              # primary for ⟨A⟩
    axVarA     = axes[3, i]              # primary for Var[A]

    # ─── heat-maps ────────────────────────────────────────────────
    imA = axA.imshow(recA, aspect='auto', cmap='viridis',
                     extent=[0, N, steps*dt, 0])
    axA.set_title("Activator A")
    if i == 0: axA.set_ylabel("Time")
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

    imB = axB.imshow(recB, aspect='auto', cmap='magma',
                     extent=[0, N, steps*dt, 0])
    axB.set_title("Inhibitor B")
    if i == 0: axB.set_ylabel("Time")
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    axA.text(0, 1.02, f"{chr(97+i)})", transform=axA.transAxes,
             fontweight='bold', fontsize=16, va='bottom')

    # ─── spatial mean: twin y-axes ───────────────────────────────
    axMeanA.plot(time, avgA, label='⟨A⟩', color='tab:blue')
    axMeanA.set_title("Spatial mean vs time")
    if i == 0: axMeanA.set_ylabel("⟨A⟩")

    axMeanB = axMeanA.twinx()
    axMeanB.plot(time, avgB, label='⟨B⟩', color='tab:orange')
    if i == ncols-1: axMeanB.set_ylabel("⟨B⟩")

    # legends
    if i == 0:
        axMeanA.legend(loc='upper left', fontsize=8, frameon=False)
        axMeanB.legend(loc='upper right', fontsize=8, frameon=False)

    # ─── spatial variance: twin y-axes (log) ─────────────────────
    axVarA.plot(time, varA, label='Var[A]', color='tab:blue')
    axVarA.set_yscale('log')
    axVarA.set_title("Spatial variance vs time (log)")
    if i == 0: axVarA.set_ylabel("Var[A]")

    axVarB = axVarA.twinx()
    axVarB.plot(time, varB, label='Var[B]', color='tab:orange')
    axVarB.set_yscale('log')
    if i == ncols-1: axVarB.set_ylabel("Var[B]")

    if i == ncols-1:  # x-labels only on last column
        axMeanA.set_xlabel("time")
        axVarA.set_xlabel("time")

    # parameter block
    param_str = (
        f"r_b = {scen['parameters']['r_b']:.4f}\n"
        f"(r_b / r_crit = {r_factors[i]:.1f})\n"
        f"D_A = {D['A']}, D_B = {D['B']}"
    )
    axVarA.text(0.5, -0.55, param_str, transform=axVarA.transAxes,
                ha='center', va='top', fontsize=8)

# ─────────────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────────────
out_dir = "output_plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "r_b_sweep_stats_dualaxis.png")
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")

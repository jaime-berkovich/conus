import numpy as np
import matplotlib.pyplot as plt
from actinh import ActivatorInhibitor
import os

# ─────────────────────────────────────────────────────
# Grid setup
# ─────────────────────────────────────────────────────
N = 200
steps = 5000

# ─────────────────────────────────────────────────────
# Define the three scenarios
# ─────────────────────────────────────────────────────
scenarios = [
    # a) Turing pattern regime
    {
        'dx': 1.0,
        'dt': 0.01,
        'initial_conditions': {
            'A': np.random.rand(N) * 0.5,
            'B': np.random.rand(N) * 0.5
        },
        'parameters': {
            's':   1.0,
            'b_a': 0.05,
            'r_a': 0.01,
            'r_b': 0.10,
            'b_b': 0.005
        },
        'diffusion_rates': {
            'A': 0.001,
            'B': 1.0
        }
    },
    # b) Local Hopf (no long‐range inhibition)
    {
        'dx': 0.01,
        'dt': 0.01,
        'initial_conditions': {
            'A': np.random.rand(N) * 0.1 + 0.05,
            'B': np.random.rand(N) * 0.1 + 0.05
        },
        'parameters': {
            's':   1.0,
            'b_a': 0.01,
            'r_a': 1.0,
            'r_b': 1.0,
            'b_b': 0.01
        },
        'diffusion_rates': {
            'A': 0.001,
            'B': 0.001
        }
    },
    # c) Mixed Hopf–Turing regime
    {
        'dx': 0.01,
        'dt': 0.01,
        'initial_conditions': {
            'A': np.random.rand(N) * 0.1 + 0.05,
            'B': np.random.rand(N) * 0.1 + 0.05
        },
        'parameters': {
            's':   1.0,
            'b_a': 0.01,
            'r_a': 1.0,
            'r_b': 0.80,
            'b_b': 0.01
        },
        'diffusion_rates': {
            'A': 0.0001,
            'B': 0.001
        }
    }
]

# ─────────────────────────────────────────────────────
# Run each scenario
# ─────────────────────────────────────────────────────
records = []
for scen in scenarios:
    model = ActivatorInhibitor(
        species_names=['A','B'],
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
# Plot horizontally: 2 rows × 3 columns
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(18, 8),
    sharex=False, sharey=False
)

plt.subplots_adjust(
    top=0.92, bottom=0.15,
    left=0.05, right=0.98,
    wspace=0.25, hspace=0.4
)

for i, ((recA, recB), scen) in enumerate(zip(records, scenarios)):
    axA = axes[0, i]
    axB = axes[1, i]

    # Activator panel
    imA = axA.imshow(
        recA, aspect='auto', cmap='viridis',
        extent=[0, N, steps*scen['dt'], 0]
    )
    axA.set_title("Activator A")
    if i == 0:
        axA.set_ylabel("Time")
    else:
        axA.set_yticks([])
    if i == 2:
        axA.set_xlabel("Position")
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

    # Inhibitor panel
    imB = axB.imshow(
        recB, aspect='auto', cmap='magma',
        extent=[0, N, steps*scen['dt'], 0]
    )
    axB.set_title("Inhibitor B")
    if i == 0:
        axB.set_ylabel("Time")
    else:
        axB.set_yticks([])
    if i == 2:
        axB.set_xlabel("Position")
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    # Subfigure label a), b), c)
    axA.text(
        0, 1.02,
        f"{chr(97 + i)})",
        transform=axA.transAxes,
        fontweight='bold',
        fontsize=16,
        va='bottom'
    )

    # Parameter block under the inhibitor panel
    param_str = (
        "Parameters:\n" +
        ", ".join(f"{k}={v}" for k,v in scen['parameters'].items()) +
        "\nDiffusion:\n" +
        ", ".join(f"D_{k}={v}" for k,v in scen['diffusion_rates'].items())
    )
    axB.text(
        0.5, -0.35,
        param_str,
        transform=axB.transAxes,
        ha='center',
        va='top',
        fontsize=8
    )

# ─────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)
out_path = os.path.join(output_folder, "combined_horizontal.png")
plt.savefig(out_path, dpi=300)
print(f"Saved to {out_path}")

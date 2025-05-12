"""
Auto-search for a Hopf bifurcation in the 1-D
Meinhardt–Gierer activator–inhibitor model.

  da/dt = s (a²/b + b_a) − r_a a
  db/dt = s  a²            − r_b b + b_b

Hopf ⇔  tr(J)=0 & det(J)>0   (J = Jacobian at the fixed point).

Author: ChatGPT
"""

import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import itertools, os, sys

# ─────────────────────────────────────────────────────
# 0) GLOBAL SETTINGS
# ─────────────────────────────────────────────────────
s      = 1.0
b_a    = 0.01
b_b    = 0.01
search_ranges = dict(
    r_a = (0.001, 1, 1000),   # (min , max , grid points)
    r_b = (0.001, 1, 1000)
)
tol_trace   = 1e-3             # how close is “on Hopf”
simulate_pde = True            # run PDE once a point is found
# PDE/plot settings
N, steps, dx, dt = 200, 5000, 0.01, 0.01
equal_diffusion  = 0.001
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────
# 1) FUNCTIONS
# ─────────────────────────────────────────────────────
def fixed_point(r_a, r_b):
    """Return (a*, b*) or None if no positive root."""
    A3 = s / r_b
    A2 = (s * b_a - r_b) / r_b
    A1 = (r_a * b_b) / (s * r_b)
    A0 = -b_a * b_b
    roots = np.roots([A3, A2, A1, A0])
    pos = roots[(roots.real > 0) & np.isclose(roots.imag, 0)].real
    if len(pos) == 0:
        return None
    a_star = pos.min()
    b_star = (a_star**2 + b_b) / r_b
    return a_star, b_star

def jacobian(a,b,r_a,r_b):
    Fa =  2*a/b - r_a
    Fb = -a**2 / b**2
    Ga =  2*s*a
    Gb = -r_b
    return np.array([[Fa, Fb],[Ga, Gb]])

# ─────────────────────────────────────────────────────
# 2) GRID SEARCH
# ─────────────────────────────────────────────────────
grid_r_a = np.linspace(*search_ranges['r_a'])
grid_r_b = np.linspace(*search_ranges['r_b'])
best = None    # (|trace|, r_a, r_b, a*, b*, trace, det)

for r_a, r_b in itertools.product(grid_r_a, grid_r_b):
    fp = fixed_point(r_a, r_b)
    if fp is None:
        continue
    a_, b_ = fp
    J      = jacobian(a_, b_, r_a, r_b)
    tr     = np.trace(J)
    det    = np.linalg.det(J)
    if det <= 0:               # need det>0 for Hopf
        continue
    score = abs(tr)
    if best is None or score < best[0]:
        best = (score, r_a, r_b, a_, b_, tr, det)

if best is None:
    sys.exit("No fixed point with det(J)>0 found in the scan rectangle.")

score, r_a_opt, r_b_opt, a_opt, b_opt, tr_opt, det_opt = best
print(f"\nBest candidate in scan:")
print(f"  r_a = {r_a_opt:.6f} , r_b = {r_b_opt:.6f}")
print(f"  trace(J) = {tr_opt:.4e}   det(J) = {det_opt:.4e}")
print(f"  |trace| = {score:.3e}   (tolerance = {tol_trace})")

if score > tol_trace:
    sys.exit("Closest point is still outside tolerance — "
             "widen grid or refine step size.")

print("\n*** Hopf condition met to tolerance; running PDE … ***")

# ─────────────────────────────────────────────────────
# 3) INTEGRATE THE PDE ONCE (optional)
# ─────────────────────────────────────────────────────
if not simulate_pde:
    sys.exit()

try:
    from actinh import ActivatorInhibitor
except ImportError:
    sys.exit("Cannot import actinh; adjust PYTHONPATH or install package.")

params = dict(s=s, b_a=b_a, r_a=r_a_opt, b_b=b_b, r_b=r_b_opt)
D      = dict(A=equal_diffusion, B=equal_diffusion)
init   = dict(A=rng.random(N)*0.1 + 0.05,
              B=rng.random(N)*0.1 + 0.05)

model = ActivatorInhibitor(
    species_names=['A','B'],
    initial_conditions=init,
    diffusion_rates=D,
    parameters=params
)

recA = np.zeros((steps, N))
recB = np.zeros((steps, N))
for t in range(steps):
    model.step(dx, dt)
    st = model.get_state()
    recA[t], recB[t] = st['A'], st['B']

# ─── Quick visualisation ─────────────────────────────
fig, (axA, axB) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
imA = axA.imshow(recA, cmap='viridis', aspect='auto',
                 extent=[0, N, steps*dt, 0])
axA.set_title(f"Activator A (Hopf: r_a={r_a_opt:.4f}, r_b={r_b_opt:.4f})")
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
imB = axB.imshow(recB, cmap='magma', aspect='auto',
                 extent=[0, N, steps*dt, 0])
axB.set_title("Inhibitor B")
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)
plt.tight_layout()

out_dir = "output_plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "auto_hopf.png")
plt.savefig(out_path, dpi=300)
print(f"Saved heat-maps to {out_path}")

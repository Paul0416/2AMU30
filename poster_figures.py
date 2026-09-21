"""Poster-only figures: split the two fixed-center panels of fig-comparison
into standalone images so each can sit in its own poster band.

Reuses the functions from cluster_comparison.py.
Outputs: fig-coincide.png (panel a), fig-limitation.png (panel b).
"""

import numpy as np
import matplotlib.pyplot as plt

from cluster_comparison import (
    pdc_membership, fcm_membership, pdc_joint_distance, fcm_contribution,
)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})

C = np.array([[-1.0, 0.0], [1.0, 0.0]])  # the fixed centers used in the report

# fig-coincide.png: membership along y=0, spread vs m, m=3 = PDC
fig, ax = plt.subplots(figsize=(6.6, 3.7))
xs = np.linspace(-2.0, 2.0, 400)
line = np.column_stack([xs, np.zeros_like(xs)])
ax.plot(xs, pdc_membership(line, C)[:, 0], color="black", lw=2.6, label="PDC")
for m, ls in zip([1.5, 2.0, 3.0, 5.0], ["-", "--", ":", "-."]):
    ax.plot(xs, fcm_membership(line, C, m)[:, 0], ls=ls, lw=1.6,
            label=f"FCM $m={m:g}$")
ax.set_xlabel("position $x$ on the axis $y=0$")
ax.set_ylabel("membership in cluster 1")
ax.set_ylim(0, 1)
ax.set_xlim(-2, 2)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig("fig-coincide.png", dpi=300, bbox_inches="tight")

# fig-limitation.png: assignment stays 0.5 while absolute fit grows
fig, ax = plt.subplots(figsize=(6.6, 3.7))
ts = np.linspace(0.0, 12.0, 400)
ray = np.column_stack([np.zeros_like(ts), ts])
ax.plot(ts, pdc_joint_distance(ray, C), color="black", lw=2.4, label="PDC $D_i$")
ax.plot(ts, fcm_contribution(ray, C, 2.0), color="C3", ls="--", lw=2.0,
        label=r"FCM $\sum_k u_{ik}^2 d_{ik}^2$")
ax.plot(ts, pdc_membership(ray, C)[:, 0], color="C0", ls=":", lw=2.0,
        label="largest membership")
ax.set_xlabel("distance $t$ from the midpoint")
ax.set_ylabel("value")
ax.legend(fontsize=10, loc="upper left")
fig.tight_layout()
fig.savefig("fig-limitation.png", dpi=300, bbox_inches="tight")

print("wrote fig-coincide.png and fig-limitation.png")

"""A0 poster figure: the fig-coincide.pdf plot from poster_figures.py
(same size, fonts, lines and colours), with the two fixed centers marked
on the x-axis.

Reuses the functions from cluster_comparison.py.
Output: fig-coincide-a0.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt

from cluster_comparison import pdc_membership, fcm_membership

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})

C = np.array([[-1.0, 0.0], [1.0, 0.0]])  # the fixed centers used in the report

fig, ax = plt.subplots(figsize=(6.6, 3.7))
xs = np.linspace(-2.0, 2.0, 400)
line = np.column_stack([xs, np.zeros_like(xs)])
ax.plot(xs, pdc_membership(line, C)[:, 0], color="black", lw=2.6, label="PDC")
for m, ls in zip([1.5, 2.0, 3.0, 5.0], ["-", "--", ":", "-."]):
    ax.plot(xs, fcm_membership(line, C, m)[:, 0], ls=ls, lw=1.6,
            label=f"FCM $m={m:g}$")
for c in (-1.0, 1.0):  # the two fixed centers
    ax.axvline(c, color="0.6", ls=(0, (2, 3)), lw=1, zorder=0)
ticks = np.arange(-2.0, 2.01, 0.5)  # same ticks as fig-coincide.pdf
names = {-1.0: "\n$c_1$", 1.0: "\n$c_2$"}
ax.set_xticks(ticks, [f"{t:.1f}".replace("-", "−") + names.get(t, "")
                      for t in ticks])
ax.set_xlabel("position $x$ on the axis $y=0$")
ax.set_ylabel("membership in cluster 1")
ax.set_ylim(0, 1)
ax.set_xlim(-2, 2)
ax.legend(fontsize=10, loc="upper right")
fig.tight_layout()
fig.savefig("fig-coincide-a0.pdf", bbox_inches="tight")

print("wrote fig-coincide-a0.pdf")

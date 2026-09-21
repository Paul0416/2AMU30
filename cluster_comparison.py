"""Probabilistic D-clustering and fuzzy c-means.
Used for Table 1 and fig-datasets.pdf and fig-comparison.pdf in the report.
Dependencies: numpy, matplotlib."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
K = 2
EPS = 1e-12  # Prevent division by zero at center

def dists(P, C):
    # Euclidean distance from each point to each center, shape (N, K)
    return np.sqrt(np.sum((P[:, None, :] - C[None, :, :]) ** 2, axis=2)) + EPS

def pdc_membership(P, C):
    # PDC: membership proportional to inverse distance
    inv = 1 / dists(P, C)
    return inv / inv.sum(1, keepdims=True)

def fcm_membership(P, C, m):
    # FCM: inverse distance raised to 2/(m-1), then normalized
    inv = dists(P, C) ** (-2 / (m - 1))
    return inv / inv.sum(1, keepdims=True)

def pdc_joint_distance(P, C):
    # PDC absolute-fit quantity D_i
    return 1 / (1 / dists(P, C)).sum(1)

def fcm_contribution(P, C, m):
    # FCM absolute-fit quantity: per-point objective contribution
    u = fcm_membership(P, C, m)
    return (u ** m * dists(P, C) ** 2).sum(1)

def fit_pdc(X, iters=200):
    # alternate membership and inverse-distance weighted center update
    C = X[rng.choice(len(X), K, replace=False)].astype(float)
    for _ in range(iters):
        d = dists(X, C)
        p = (1 / d) / (1 / d).sum(1, keepdims=True)
        w = p ** 2 / d
        C = (w.T @ X) / w.sum(0)[:, None]
    return C

def fit_fcm(X, m, iters=200):
    # alternate membership and u^m weighted center update
    C = X[rng.choice(len(X), K, replace=False)].astype(float)
    for _ in range(iters):
        u = fcm_membership(X, C, m) ** m
        C = (u.T @ X) / u.sum(0)[:, None]
    return C

def table1():
    # fixed-center example: two centers, two chosen points
    C = np.array([[-1.0, 0.0], [1.0, 0.0]])
    P = np.array([[0.0, 0.0], [0.0, 10.0]])
    p = pdc_membership(P, C)
    u = fcm_membership(P, C, 2.0)
    D = pdc_joint_distance(P, C)
    J = fcm_contribution(P, C, 2.0)
    for i, name in enumerate(["x_B", "x_C"]):
        print(name, np.round(p[i], 3), np.round(u[i], 3),
              round(D[i], 3), round(J[i], 3))

def fig_datasets():
    # cluster one shared two-blob dataset with each method
    c1, c2 = np.array([0.0, 0.0]), np.array([6.0, 0.0])
    X = np.vstack([rng.normal(c1, 1.0, (120, 2)), rng.normal(c2, 1.0, (120, 2))])
    order = lambda C: C[np.argsort(C[:, 0])]  # sort centers for consistent coloring
    runs = [("PDC", order(fit_pdc(X)), lambda C: pdc_membership(X, C)),
            ("FCM $m=2$", order(fit_fcm(X, 2.0)), lambda C: fcm_membership(X, C, 2.0)),
            ("FCM $m=3$", order(fit_fcm(X, 3.0)), lambda C: fcm_membership(X, C, 3.0))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, C, memb) in zip(axes, runs):
        # color points by membership in cluster 1, mark fitted centers
        sc = ax.scatter(X[:, 0], X[:, 1], c=memb(C)[:, 0], cmap="coolwarm",
                        s=8, vmin=0, vmax=1)
        ax.scatter(C[:, 0], C[:, 1], marker="x", s=90, c="black", linewidths=2)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01,
                 label="membership in cluster 1")
    fig.savefig("fig-datasets.pdf", bbox_inches="tight")

def fig_comparison():
    # two panels over the fixed centers c1=(-1,0), c2=(1,0)
    C = np.array([[-1.0, 0.0], [1.0, 0.0]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    # panel (a): membership along the axis y=0, FCM m=3 overlays PDC
    xs = np.linspace(-2.0, 2.0, 400)
    line = np.column_stack([xs, np.zeros_like(xs)])
    ax1.plot(xs, pdc_membership(line, C)[:, 0], color="black", lw=2.2, label="PDC")
    for m, ls in zip([1.5, 2.0, 3.0, 5.0], ["-", "--", ":", "-."]):
        ax1.plot(xs, fcm_membership(line, C, m)[:, 0], ls=ls, lw=1.3,
                 label=f"FCM $m={m:g}$")
    ax1.set_xlabel("position $x$ on the axis $y=0$")
    ax1.set_ylabel("membership in cluster 1")
    ax1.set_title("(a) fuzzifier sets sharpness; $m=3$ matches PDC", fontsize=8.5)
    ax1.legend(fontsize=7)
    ax1.set_ylim(0, 1)
    # panel (b): points receding from the midpoint; assignment vs absolute fit
    ts = np.linspace(0.0, 12.0, 400)
    ray = np.column_stack([np.zeros_like(ts), ts])
    ax2.plot(ts, pdc_joint_distance(ray, C), color="black", lw=2.0, label="PDC $D_i$")
    ax2.plot(ts, fcm_contribution(ray, C, 2.0), color="C3", ls="--", lw=1.6,
             label=r"FCM $\sum_k u_{ik}^2 d_{ik}^2$")
    ax2.plot(ts, pdc_membership(ray, C)[:, 0], color="C0", ls=":", lw=1.4,
             label="largest membership")
    ax2.set_xlabel("distance $t$ from the midpoint")
    ax2.set_ylabel("value")
    ax2.set_title("(b) assignment stays 0.5; absolute fit grows", fontsize=8.5)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig-comparison.pdf", bbox_inches="tight")

if __name__ == "__main__":
    table1()
    fig_datasets()
    fig_comparison()

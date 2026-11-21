# EVOLVE-BLOCK-START
"""Multi-start SLSQP search for 26 circles in the unit square, maximising the sum of radii.

Variables: 26 centre x, 26 centre y, 26 radii. Constraints: pairwise non-overlap and
containment in [0,1]^2. Search: random restarts plus perturbation of the incumbent, each
polished with SLSQP, then repaired and shrunk by a tiny margin so the zero-tolerance
checker accepts it. Not executed in the authoring environment; logic checked by reading.
"""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEARCH_SECONDS = 150.0  # harness stops the job at 300 s; leave a wide margin
MARGIN = 1e-9           # radius shrink so float rounding cannot cause an overlap
SEED = 20260

II, JJ = np.triu_indices(N, 1)
NP = len(II)
NV = 3 * N
BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _cons(z):
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    return np.concatenate([
        d - r[II] - r[JJ],
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
    ])


def _cons_jac(z):
    x = z[:N]
    y = z[N:2 * N]
    dx = x[II] - x[JJ]
    dy = y[II] - y[JJ]
    d = np.sqrt(dx * dx + dy * dy + 1e-18)
    ux = dx / d
    uy = dy / d
    J = np.zeros((NP + 4 * N, NV))
    k = np.arange(NP)
    J[k, II] = ux
    J[k, JJ] = -ux
    J[k, N + II] = uy
    J[k, N + JJ] = -uy
    J[k, 2 * N + II] = -1.0
    J[k, 2 * N + JJ] = -1.0
    i = np.arange(N)
    b = NP
    J[b + i, i] = 1.0
    J[b + i, 2 * N + i] = -1.0
    b = NP + N
    J[b + i, i] = -1.0
    J[b + i, 2 * N + i] = -1.0
    b = NP + 2 * N
    J[b + i, N + i] = 1.0
    J[b + i, 2 * N + i] = -1.0
    b = NP + 3 * N
    J[b + i, N + i] = -1.0
    J[b + i, 2 * N + i] = -1.0
    return J


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_jac(z):
    g = np.zeros(NV)
    g[2 * N:] = -1.0
    return g


def _slsqp(z0):
    res = minimize(
        _obj, z0, jac=_obj_jac, method="SLSQP", bounds=BOUNDS,
        constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    return res.x


def _random_start(rng):
    xy = rng.uniform(0.05, 0.95, size=(N, 2))
    r = np.full(N, 0.02)
    return np.concatenate([xy[:, 0], xy[:, 1], r])


def _perturb(z, rng):
    z = z.copy()
    x = z[:N]
    y = z[N:2 * N]
    r = z[2 * N:]
    if rng.random() < 0.5:
        sigma = rng.uniform(0.01, 0.06)
        x += rng.normal(0.0, sigma, N)
        y += rng.normal(0.0, sigma, N)
    else:
        k = int(rng.integers(1, 4))
        idx = rng.choice(N, size=k, replace=False)
        x[idx] = rng.uniform(0.05, 0.95, k)
        y[idx] = rng.uniform(0.05, 0.95, k)
    r *= 0.9
    np.clip(z, 0.0, 1.0, out=z[:2 * N])
    z[2 * N:] = np.clip(r, 0.0, 0.5)
    return z


def _repair(x, y, r):
    """Return (x, y, r) that satisfy all constraints in exact double arithmetic."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    r = np.maximum(np.asarray(r, dtype=float), 0.0)
    r = np.minimum(r, np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)))
    converged = False
    for _ in range(200):
        changed = False
        for a in range(N):
            for b in range(a + 1, N):
                d = np.hypot(x[a] - x[b], y[a] - y[b])
                s = r[a] + r[b]
                if s > d:
                    f = d / s if s > 0.0 else 0.0
                    r[a] *= f
                    r[b] *= f
                    changed = True
        if not changed:
            converged = True
            break
    if not converged:
        # Guaranteed-valid fallback: r_a <= d_ab / 2 for every b.
        for a in range(N):
            for b in range(N):
                if a != b:
                    d = np.hypot(x[a] - x[b], y[a] - y[b])
                    r[a] = min(r[a], d / 2.0)
    r = np.maximum(r - MARGIN, 0.0)
    return x, y, r


def construct_packing():
    """
    Search for a packing of 26 circles in the unit square maximising the sum of radii.

    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26,) with radius of each circle
    """
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    best_val = -1.0
    best = None
    incumbent = None

    while time.perf_counter() - t0 < SEARCH_SECONDS:
        if incumbent is None or rng.random() < 0.3:
            z0 = _random_start(rng)
        else:
            z0 = _perturb(incumbent, rng)
        try:
            z = _slsqp(z0)
        except Exception:
            continue
        x, y, r = _repair(z[:N], z[N:2 * N], z[2 * N:])
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (x, y, r)
            incumbent = z

    if best is None:
        # Every solver run failed: return a valid (if weak) repaired packing.
        z0 = _random_start(rng)
        best = _repair(z0[:N], z0[N:2 * N], z0[2 * N:])

    x, y, r = best
    centers = np.column_stack([x, y])
    return centers, r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing search for n=26"""
    centers, radii = construct_packing()
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

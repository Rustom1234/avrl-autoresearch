# EVOLVE-BLOCK-START
"""Multi-start SLSQP with basin hopping: 26 circles in the unit square, maximise sum of radii."""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = 20260101
TIME_LIMIT = 200.0   # seconds of wall time for the whole search (harness cap is 300 s)
N_STARTS = 60        # random multi-starts
N_HOPS = 150         # basin-hopping rounds from the incumbent
SHRINK = 1e-9        # relative safety margin applied to every radius

IU, JU = np.triu_indices(N, k=1)
M_PAIR = IU.size
IDX = np.arange(N)
ROWS_PAIR = 4 * N + np.arange(M_PAIR)
_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _split(z):
    return z[:N], z[N:2 * N], z[2 * N:]


def _constraints(z):
    """All SLSQP inequality constraints, each must be >= 0."""
    x, y, r = _split(z)
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    rs = r[IU] + r[JU]
    border = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return np.concatenate([border, dx * dx + dy * dy - rs * rs])


def _constraint_jac(z):
    x, y, r = _split(z)
    J = np.zeros((4 * N + M_PAIR, 3 * N))
    # border rows
    J[IDX, IDX] = 1.0
    J[IDX, 2 * N + IDX] = -1.0
    J[N + IDX, IDX] = -1.0
    J[N + IDX, 2 * N + IDX] = -1.0
    J[2 * N + IDX, N + IDX] = 1.0
    J[2 * N + IDX, 2 * N + IDX] = -1.0
    J[3 * N + IDX, N + IDX] = -1.0
    J[3 * N + IDX, 2 * N + IDX] = -1.0
    # pair rows
    dx = x[IU] - x[JU]
    dy = y[IU] - y[JU]
    rs = r[IU] + r[JU]
    J[ROWS_PAIR, IU] = 2.0 * dx
    J[ROWS_PAIR, JU] = -2.0 * dx
    J[ROWS_PAIR, N + IU] = 2.0 * dy
    J[ROWS_PAIR, N + JU] = -2.0 * dy
    J[ROWS_PAIR, 2 * N + IU] = -2.0 * rs
    J[ROWS_PAIR, 2 * N + JU] = -2.0 * rs
    return J


def _objective(z):
    return -np.sum(z[2 * N:])


def _objective_grad(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _optimise(z0):
    try:
        res = minimize(
            _objective,
            z0,
            jac=_objective_grad,
            method="SLSQP",
            bounds=_BOUNDS,
            constraints=[{"type": "ineq", "fun": _constraints, "jac": _constraint_jac}],
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return z0


def _is_valid(c, r):
    if np.any(c < 0.0) or np.any(c > 1.0) or np.any(r < 0.0):
        return False
    if np.any(c - r[:, None] < 0.0) or np.any(c + r[:, None] > 1.0):
        return False
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    s = r[:, None] + r[None, :]
    return bool(np.all(d[IU, JU] >= s[IU, JU]))


def _repair(z):
    """Turn any candidate into a strictly valid packing by shrinking radii only."""
    x, y, r = _split(z)
    c = np.stack([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)], axis=1)
    r = np.clip(r, 0.0, None)
    wall = np.minimum(np.minimum(c[:, 0], c[:, 1]), np.minimum(1.0 - c[:, 0], 1.0 - c[:, 1]))
    r = np.minimum(r, wall)
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                s = r[i] + r[j]
                if s > d:
                    f = d / s
                    r[i] *= f
                    r[j] *= f
                    changed = True
        if not changed:
            break
    r = r * (1.0 - SHRINK)
    for _ in range(100):
        if _is_valid(c, r):
            break
        r = r * (1.0 - 1e-7)
    return c, r


def _random_start(rng):
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    return np.concatenate([x, y, np.full(N, 0.05)])


def _perturb(c_best, r_best, rng):
    c = c_best.copy()
    r = 0.9 * r_best
    k = int(rng.integers(1, 4))
    pick = rng.choice(N, size=k, replace=False)
    c[pick] = rng.uniform(0.05, 0.95, (k, 2))
    r[pick] = 0.02
    c = np.clip(c + rng.normal(0.0, 0.01, c.shape), 0.0, 1.0)
    return np.concatenate([c[:, 0], c[:, 1], r])


def run_packing():
    """Search for a good packing and return (centers, radii, sum_radii)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    best = None  # (sum, centers, radii)

    def consider(z):
        nonlocal best
        c, r = _repair(z)
        s = float(np.sum(r))
        if best is None or s > best[0]:
            best = (s, c, r)

    consider(_random_start(rng))

    for _ in range(N_STARTS):
        if time.perf_counter() - t0 > 0.5 * TIME_LIMIT:
            break
        consider(_optimise(_random_start(rng)))

    for _ in range(N_HOPS):
        if time.perf_counter() - t0 > TIME_LIMIT:
            break
        _, c_best, r_best = best
        consider(_optimise(_perturb(c_best, r_best, rng)))

    _, centers, radii = best
    centers = np.ascontiguousarray(centers, dtype=float)
    radii = np.ascontiguousarray(radii, dtype=float)
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
# run_packing() is defined above and returns (centers, radii, sum_radii).

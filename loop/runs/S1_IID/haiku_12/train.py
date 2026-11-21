# EVOLVE-BLOCK-START
"""Multi-start SLSQP optimisation of centres and radii for 26 circles in the unit square."""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
TIME_BUDGET = 170.0  # seconds of wall time for the search (hard limit is 300 s)
SEED = 20260417
RESTART_PROB = 0.2   # chance of a fresh random start during basin hopping

_I, _J = np.triu_indices(N, k=1)  # all 325 pairs
_P = _I.size
_M = 4 * N + _P                   # boundary constraints + pair constraints
_ROWS = np.arange(N)
_RC = 2 * N                       # offset of radii inside the flat vector z
_BOUNDS = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _unpack(z):
    return z[:_RC].reshape(N, 2), z[_RC:]


def _cons(z):
    X, r = _unpack(z)
    dx = X[_I, 0] - X[_J, 0]
    dy = X[_I, 1] - X[_J, 1]
    s = r[_I] + r[_J]
    bnd = np.concatenate([X[:, 0] - r, 1.0 - X[:, 0] - r,
                          X[:, 1] - r, 1.0 - X[:, 1] - r])
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([bnd, pair])


def _cons_jac(z):
    X, _ = _unpack(z)
    _, r = _unpack(z)
    dx = X[_I, 0] - X[_J, 0]
    dy = X[_I, 1] - X[_J, 1]
    s = r[_I] + r[_J]
    Jm = np.zeros((_M, 3 * N))
    Jm[_ROWS, 2 * _ROWS] = 1.0            # x_i - r_i >= 0
    Jm[_ROWS, _RC + _ROWS] = -1.0
    Jm[N + _ROWS, 2 * _ROWS] = -1.0       # 1 - x_i - r_i >= 0
    Jm[N + _ROWS, _RC + _ROWS] = -1.0
    Jm[2 * N + _ROWS, 2 * _ROWS + 1] = 1.0  # y_i - r_i >= 0
    Jm[2 * N + _ROWS, _RC + _ROWS] = -1.0
    Jm[3 * N + _ROWS, 2 * _ROWS + 1] = -1.0  # 1 - y_i - r_i >= 0
    Jm[3 * N + _ROWS, _RC + _ROWS] = -1.0
    pr = 4 * N + np.arange(_P)
    Jm[pr, 2 * _I] = 2.0 * dx
    Jm[pr, 2 * _J] = -2.0 * dx
    Jm[pr, 2 * _I + 1] = 2.0 * dy
    Jm[pr, 2 * _J + 1] = -2.0 * dy
    Jm[pr, _RC + _I] = -2.0 * s
    Jm[pr, _RC + _J] = -2.0 * s
    return Jm


def _obj(z):
    return -np.sum(z[_RC:])


def _obj_grad(z):
    g = np.zeros_like(z)
    g[_RC:] = -1.0
    return g


def _make_valid(X, r):
    """Return a packing that is feasible in double precision (with a tiny safety margin)."""
    X = np.clip(np.asarray(X, dtype=float), 0.0, 1.0)
    r = np.clip(np.asarray(r, dtype=float), 0.0, 0.5)
    wall = np.minimum(np.minimum(X[:, 0], 1.0 - X[:, 0]),
                      np.minimum(X[:, 1], 1.0 - X[:, 1]))
    r = np.minimum(r, wall)
    dx = X[_I, 0] - X[_J, 0]
    dy = X[_I, 1] - X[_J, 1]
    d = np.sqrt(dx * dx + dy * dy)
    for p in np.nonzero(r[_I] + r[_J] > d)[0]:
        i, j = _I[p], _J[p]
        target = d[p] * (1.0 - 1e-12)
        si = r[i] + r[j]
        if si > target and si > 0.0:
            f = target / si
            r[i] *= f
            r[j] *= f
    r = np.maximum(r - 1e-10, 0.0)
    return X, r


def _local(X0, r0):
    X0 = np.clip(np.asarray(X0, dtype=float), 0.0, 1.0)
    r0 = np.clip(np.asarray(r0, dtype=float), 0.0, 0.5)
    z0 = np.concatenate([X0.ravel(), r0])
    try:
        res = minimize(_obj, z0, jac=_obj_grad, method="SLSQP", bounds=_BOUNDS,
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _cons_jac}],
                       options={"maxiter": 400, "ftol": 1e-12})
        z = res.x
    except Exception:
        z = z0
    X, r = _unpack(np.asarray(z, dtype=float))
    return _make_valid(X, r)


def _random_start(rng):
    X = rng.uniform(0.05, 0.95, size=(N, 2))
    r = rng.uniform(0.02, 0.12, size=N)
    return X, r


def _perturb(X, r, rng):
    sigma = rng.uniform(0.02, 0.08)
    X2 = X + rng.normal(0.0, sigma, size=X.shape)
    k = int(rng.integers(1, 3))
    idx = rng.choice(N, size=k, replace=False)
    X2[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
    r2 = r * 0.9
    return np.clip(X2, 0.0, 1.0), r2


def construct_packing():
    """
    Find a packing of 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii): centers shape (26, 2), radii shape (26,).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    best_X, best_r, best_s = None, None, -1.0

    # Phase 1: independent random starts
    while time.perf_counter() - t0 < 0.5 * TIME_BUDGET:
        X0, r0 = _random_start(rng)
        X, r = _local(X0, r0)
        s = float(np.sum(r))
        if s > best_s:
            best_X, best_r, best_s = X, r, s

    # Phase 2: basin hopping around the incumbent, with occasional fresh restarts
    while time.perf_counter() - t0 < TIME_BUDGET:
        if rng.random() < RESTART_PROB:
            X0, r0 = _random_start(rng)
        else:
            X0, r0 = _perturb(best_X, best_r, rng)
        X, r = _local(X0, r0)
        s = float(np.sum(r))
        if s > best_s:
            best_X, best_r, best_s = X, r, s

    return best_X, best_r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

# EVOLVE-BLOCK-START
"""Multi-start SLSQP with basin hopping: 26 circles in the unit square, maximize the sum of radii."""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
SEED = 20260
N_STARTS = 80        # fixed budget so results are repeatable
TIME_GUARD = 190.0   # seconds; only fires on a very slow machine
MAX_ITER = 300

_I, _J = np.triu_indices(N, 1)
_P = _I.size
_ROWS = np.arange(_P)

# Boundary constraints, stacked as: x - r, 1 - x - r, y - r, 1 - y - r
_EYE = np.eye(N)
_ZERO = np.zeros((N, N))
_BJ = np.block([
    [_EYE, _ZERO, -_EYE],
    [-_EYE, _ZERO, -_EYE],
    [_ZERO, _EYE, -_EYE],
    [_ZERO, -_EYE, -_EYE],
])

_BOUNDS = [(None, None)] * (2 * N) + [(0.0, None)] * N


def _unpack(z):
    return z[:N], z[N:2 * N], z[2 * N:]


def _objective(z):
    return -np.sum(z[2 * N:])


def _objective_grad(z):
    g = np.zeros_like(z)
    g[2 * N:] = -1.0
    return g


def _boundary_cons(z):
    x, y, r = _unpack(z)
    return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])


def _pair_cons(z):
    x, y, r = _unpack(z)
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    rs = r[_I] + r[_J]
    return dx * dx + dy * dy - rs * rs


def _pair_jac(z):
    x, y, r = _unpack(z)
    dx = x[_I] - x[_J]
    dy = y[_I] - y[_J]
    rs = r[_I] + r[_J]
    G = np.zeros((_P, 3 * N))
    G[_ROWS, _I] = 2 * dx
    G[_ROWS, _J] = -2 * dx
    G[_ROWS, N + _I] = 2 * dy
    G[_ROWS, N + _J] = -2 * dy
    G[_ROWS, 2 * N + _I] = -2 * rs
    G[_ROWS, 2 * N + _J] = -2 * rs
    return G


_CONS = [{
    "type": "ineq",
    "fun": lambda z: np.concatenate([_boundary_cons(z), _pair_cons(z)]),
    "jac": lambda z: np.vstack([_BJ, _pair_jac(z)]),
}]


def _local_solve(z0):
    res = minimize(_objective, z0, jac=_objective_grad, method="SLSQP",
                   bounds=_BOUNDS, constraints=_CONS,
                   options={"maxiter": MAX_ITER, "ftol": 1e-12})
    return res.x if np.all(np.isfinite(res.x)) else z0


def _repair(z):
    """Return a provably valid (centers, radii) pair from an approximate solution."""
    x = np.clip(z[:N], 0.0, 1.0)
    y = np.clip(z[N:2 * N], 0.0, 1.0)
    r = np.clip(z[2 * N:], 0.0, None)
    r = np.minimum(r, np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)))
    for k in range(_P):
        i, j = _I[k], _J[k]
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        s = r[i] + r[j]
        if s > d:
            f = d / s
            r[i] *= f
            r[j] *= f
    r = r * (1.0 - 1e-9)
    return np.column_stack([x, y]), r


def _pack_z(c, r):
    return np.concatenate([c[:, 0], c[:, 1], r])


def _random_start(rng):
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    return _pack_z(c, np.full(N, 0.02))


def _perturb(best_c, best_r, rng):
    c = best_c + rng.normal(0.0, rng.uniform(0.01, 0.08), size=(N, 2))
    r = best_r.copy()
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        idx = rng.choice(N, size=k, replace=False)
        c[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
        r[idx] = 0.0
    c = np.clip(c, 0.0, 1.0)
    return _pack_z(c, r)


def construct_packing():
    """
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
    """
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    best_c, best_r, best_s = None, None, -np.inf

    for k in range(N_STARTS):
        if best_c is None or k < 24 or rng.random() < 0.35:
            z0 = _random_start(rng)
        else:
            z0 = _perturb(best_c, best_r, rng)
        c, r = _repair(_local_solve(z0))
        s = float(np.sum(r))
        if s > best_s:
            best_c, best_r, best_s = c, r, s
        if time.perf_counter() - t0 > TIME_GUARD:
            break

    # Final polish of the best packing found.
    c, r = _repair(_local_solve(_pack_z(best_c, best_r)))
    if float(np.sum(r)) > best_s:
        best_c, best_r = c, r

    return best_c, best_r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

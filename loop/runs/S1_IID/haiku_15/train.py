# EVOLVE-BLOCK-START
"""Multi-start SLSQP optimisation of centres and radii for 26 circles in a unit square."""

import time

import numpy as np
from scipy.optimize import minimize

N = 26
_I, _J = np.triu_indices(N, 1)
_P = len(_I)
_SEED = 12345
_N_RANDOM_STARTS = 12
_N_TOTAL_STARTS = 40
_SOFT_LIMIT = 170.0   # stop launching new starts after this many seconds
_HARD_LIMIT = 215.0   # abort a running optimisation after this many seconds


class _Timeout(Exception):
    pass


def _unpack(z):
    # layout: z[2i], z[2i+1] are centre x, y of circle i; z[2N+i] is its radius
    return z[:2 * N].reshape(N, 2), z[2 * N:]


def _obj(z):
    return -np.sum(z[2 * N:])


def _obj_jac(z):
    g = np.zeros(3 * N)
    g[2 * N:] = -1.0
    return g


def _cons(z):
    c, r = _unpack(z)
    x, y = c[:, 0], c[:, 1]
    bnd = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    dx = c[_I, 0] - c[_J, 0]
    dy = c[_I, 1] - c[_J, 1]
    s = r[_I] + r[_J]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([bnd, pair])


def _cons_jac(z):
    c, r = _unpack(z)
    k = np.arange(N)
    dx = c[_I, 0] - c[_J, 0]
    dy = c[_I, 1] - c[_J, 1]
    s = r[_I] + r[_J]
    jm = np.zeros((4 * N + _P, 3 * N))
    jm[k, 2 * k] = 1.0
    jm[k, 2 * N + k] = -1.0
    jm[N + k, 2 * k] = -1.0
    jm[N + k, 2 * N + k] = -1.0
    jm[2 * N + k, 2 * k + 1] = 1.0
    jm[2 * N + k, 2 * N + k] = -1.0
    jm[3 * N + k, 2 * k + 1] = -1.0
    jm[3 * N + k, 2 * N + k] = -1.0
    p = 4 * N + np.arange(_P)
    jm[p, 2 * _I] = 2 * dx
    jm[p, 2 * _J] = -2 * dx
    jm[p, 2 * _I + 1] = 2 * dy
    jm[p, 2 * _J + 1] = -2 * dy
    jm[p, 2 * N + _I] = -2 * s
    jm[p, 2 * N + _J] = -2 * s
    return jm


def _optimise(z0, deadline):
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

    def cb(_zk):
        if time.perf_counter() > deadline:
            raise _Timeout()

    res = minimize(_obj, z0, jac=_obj_jac, method="SLSQP", bounds=bounds,
                   constraints=[dict(type="ineq", fun=_cons, jac=_cons_jac)],
                   callback=cb, options=dict(maxiter=500, ftol=1e-12))
    return res.x


def _repair(c, r):
    """Make the packing exactly feasible in double precision (only shrinks radii)."""
    c = np.clip(c, 0.0, 1.0)
    r = np.maximum(r, 0.0)
    r = np.minimum(r, np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                                 np.minimum(c[:, 1], 1 - c[:, 1])))
    for _ in range(100000):
        d = np.sqrt(np.sum((c[_I] - c[_J]) ** 2, axis=1))
        s = r[_I] + r[_J]
        bad = s > d
        if not bad.any():
            break
        ratio = np.where(bad, d / np.maximum(s, 1e-300) * (1 - 1e-12), 1.0)
        f = np.ones(N)
        np.minimum.at(f, _I, ratio)
        np.minimum.at(f, _J, ratio)
        r = r * f
    r = r * (1 - 1e-9)
    return c, r


def _random_start(rng):
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    return np.concatenate([c.ravel(), np.full(N, 0.02)])


def _perturbed_start(best_z, rng):
    z = best_z.copy()
    sigma = rng.uniform(0.03, 0.12)
    z[:2 * N] = np.clip(z[:2 * N] + rng.normal(0.0, sigma, 2 * N), 0.0, 1.0)
    z[2 * N:] = np.clip(z[2 * N:] * 0.9, 0.0, 0.5)
    return z


# EVOLVE-BLOCK-END


def run_packing():
    """Run the multi-start optimiser for n=26; returns (centers, radii, sum_radii)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(_SEED)
    best_z = None
    best_c = None
    best_r = None
    best_s = -1.0
    for k in range(_N_TOTAL_STARTS):
        if time.perf_counter() - t0 > _SOFT_LIMIT:
            break
        if k < _N_RANDOM_STARTS or best_z is None:
            z0 = _random_start(rng)
        else:
            z0 = _perturbed_start(best_z, rng)
        try:
            zf = _optimise(z0, t0 + _HARD_LIMIT)
        except _Timeout:
            break
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(zf)):
            continue
        c, r = _unpack(zf)
        c, r = _repair(c.copy(), r.copy())
        s = float(np.sum(r))
        if s > best_s:
            best_s, best_c, best_r, best_z = s, c, r, zf

    if best_c is None:
        # Fallback: all radii zero is always a valid packing.
        best_c = np.random.default_rng(_SEED).uniform(0.05, 0.95, size=(N, 2))
        best_r = np.zeros(N)

    centers = np.asarray(best_c, dtype=float).reshape(N, 2)
    radii = np.asarray(best_r, dtype=float).reshape(N)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

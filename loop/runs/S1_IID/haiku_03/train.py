# EVOLVE-BLOCK-START
"""Optimised circle packing for n=26 circles in the unit square (maximise sum of radii).

Method: seeded multi-start. Each start is relaxed with a quadratic penalty
(L-BFGS-B, increasing weight), polished with SLSQP on the exact constraints
(joint optimisation of centres and radii), then radii are re-optimised for the
fixed centres with an LP (HiGHS). Basin hopping from the incumbent is mixed in.
The returned packing is repaired and shrunk by a tiny margin so that it is
valid in double precision.
"""

import time

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
SEED = 2026
TIME_BUDGET = 165.0  # seconds of search; the harness stops jobs at 5 min
MAX_ITERS = 400  # cap on number of local optimisations
SHRINK = 1e-9  # relative safety margin applied to every radius

_I, _J = np.triu_indices(N, k=1)
_NP = len(_I)
_ROWS = np.arange(_NP)
_AR = np.arange(N)
_ZB = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N


def _pen(z, w):
    """Quadratic penalty energy and its analytic gradient."""
    c = z[: 2 * N].reshape(N, 2)
    r = z[2 * N:]
    x = c[:, 0]
    y = c[:, 1]
    diff = c[_I] - c[_J]
    d = np.sqrt(np.sum(diff * diff, axis=1)) + 1e-12
    v = np.maximum(0.0, r[_I] + r[_J] - d)
    a1 = np.maximum(0.0, r - x)
    a2 = np.maximum(0.0, r + x - 1.0)
    b1 = np.maximum(0.0, r - y)
    b2 = np.maximum(0.0, r + y - 1.0)
    f = -np.sum(r) + w * (np.sum(v * v) + np.sum(a1 * a1 + a2 * a2 + b1 * b1 + b2 * b2))

    gr = np.full(N, -1.0)
    np.add.at(gr, _I, 2.0 * w * v)
    np.add.at(gr, _J, 2.0 * w * v)
    gr += 2.0 * w * (a1 + a2 + b1 + b2)

    gc = np.zeros((N, 2))
    coef = (2.0 * w * v / d)[:, None] * diff
    np.add.at(gc, _I, -coef)
    np.add.at(gc, _J, coef)
    gc[:, 0] += 2.0 * w * (-a1 + a2)
    gc[:, 1] += 2.0 * w * (-b1 + b2)
    return f, np.concatenate([gc.ravel(), gr])


def _relax(z):
    for w in (10.0, 100.0, 1e3, 1e4, 1e5):
        z = minimize(_pen, z, args=(w,), jac=True, method="L-BFGS-B",
                     bounds=_ZB, options={"maxiter": 300}).x
    return z


def _cfun(z):
    c = z[: 2 * N].reshape(N, 2)
    r = z[2 * N:]
    diff = c[_I] - c[_J]
    d = np.sqrt(np.sum(diff * diff, axis=1)) + 1e-12
    x = c[:, 0]
    y = c[:, 1]
    return np.concatenate([d - r[_I] - r[_J], x - r, 1.0 - x - r, y - r, 1.0 - y - r])


def _cjac(z):
    c = z[: 2 * N].reshape(N, 2)
    diff = c[_I] - c[_J]
    d = np.sqrt(np.sum(diff * diff, axis=1)) + 1e-12
    u = diff / d[:, None]
    m = _NP + 4 * N
    J = np.zeros((m, 3 * N))
    J[_ROWS, 2 * _I] = u[:, 0]
    J[_ROWS, 2 * _I + 1] = u[:, 1]
    J[_ROWS, 2 * _J] = -u[:, 0]
    J[_ROWS, 2 * _J + 1] = -u[:, 1]
    J[_ROWS, 2 * N + _I] = -1.0
    J[_ROWS, 2 * N + _J] = -1.0
    ar = _AR
    J[_NP + ar, 2 * ar] = 1.0
    J[_NP + ar, 2 * N + ar] = -1.0
    J[_NP + N + ar, 2 * ar] = -1.0
    J[_NP + N + ar, 2 * N + ar] = -1.0
    J[_NP + 2 * N + ar, 2 * ar + 1] = 1.0
    J[_NP + 2 * N + ar, 2 * N + ar] = -1.0
    J[_NP + 3 * N + ar, 2 * ar + 1] = -1.0
    J[_NP + 3 * N + ar, 2 * N + ar] = -1.0
    return J


def _slsqp(z):
    def fobj(zz):
        g = np.zeros_like(zz)
        g[2 * N:] = -1.0
        return -np.sum(zz[2 * N:]), g

    res = minimize(fobj, z, jac=True, method="SLSQP", bounds=_ZB,
                   constraints=[{"type": "ineq", "fun": _cfun, "jac": _cjac}],
                   options={"maxiter": 400, "ftol": 1e-13})
    return res.x


def _lp_radii(c):
    """Optimal radii for fixed centres (LP)."""
    x = c[:, 0]
    y = c[:, 1]
    cap = np.maximum(0.0, np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y)))
    diff = c[_I] - c[_J]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    A = np.zeros((_NP, N))
    A[_ROWS, _I] = 1.0
    A[_ROWS, _J] = 1.0
    res = linprog(-np.ones(N), A_ub=A, b_ub=d * (1.0 - 1e-9),
                  bounds=[(0.0, float(v)) for v in cap], method="highs")
    if res.status != 0 or res.x is None:
        return None
    return res.x


def _finalize(c, r):
    """Clip into the square, enforce all constraints by proportional shrinking, add margin."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    x = c[:, 0]
    y = c[:, 1]
    cap = np.minimum(np.minimum(x, y), np.minimum(1.0 - x, 1.0 - y))
    r = np.clip(np.minimum(np.asarray(r, dtype=float), cap), 0.0, None)
    for _ in range(1000):
        diff = c[_I] - c[_J]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        s = r[_I] + r[_J]
        bad = s > d * (1.0 - 1e-12)
        if not bad.any():
            break
        f_pair = np.where(bad, d * (1.0 - 1e-12) / np.maximum(s, 1e-300), 1.0)
        fac = np.ones(N)
        np.minimum.at(fac, _I, f_pair)
        np.minimum.at(fac, _J, f_pair)
        r = r * fac
    r = np.clip(r * (1.0 - SHRINK), 0.0, None)
    return c, r


def _valid(c, r):
    if np.any(r < 0) or np.any(c < 0) or np.any(c > 1):
        return False
    if np.any(c - r[:, None] < 0) or np.any(c + r[:, None] > 1):
        return False
    diff = c[_I] - c[_J]
    d = np.sqrt(np.sum(diff * diff, axis=1))
    return bool(np.all(r[_I] + r[_J] <= d))


def _refine(c, r):
    """Pick the better of the SLSQP radii and the LP radii for the same centres."""
    cands = [_finalize(c, r)]
    rl = _lp_radii(np.clip(c, 0.0, 1.0))
    if rl is not None:
        cands.append(_finalize(c, rl))
    return max(cands, key=lambda t: float(np.sum(t[1])))


def _perturb(c, r, rng):
    c = c.copy()
    if rng.integers(2) == 0:
        k = int(rng.integers(1, 4))
        idx = rng.choice(N, size=k, replace=False)
        c[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
    else:
        c = c + rng.normal(0.0, rng.uniform(0.01, 0.05), size=c.shape)
    c = np.clip(c, 0.0, 1.0)
    return c, np.clip(r * 0.9, 0.0, 0.5)


def _search():
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    best_c, best_r, best_s = None, None, -1.0
    for it in range(MAX_ITERS):
        if time.perf_counter() - t0 > TIME_BUDGET:
            break
        try:
            if best_c is None or it < 10 or rng.random() < 0.25:
                c0 = rng.uniform(0.05, 0.95, size=(N, 2))
                r0 = np.full(N, 0.05)
            else:
                c0, r0 = _perturb(best_c, best_r, rng)
            z = _relax(np.concatenate([c0.ravel(), r0]))
            z = _slsqp(z)
            c, r = _refine(z[: 2 * N].reshape(N, 2), z[2 * N:])
            s = float(np.sum(r))
            if s > best_s and _valid(c, r):
                best_s, best_c, best_r = s, c, r
        except Exception:
            continue
    return best_c, best_r


def construct_packing():
    """
    Construct an arrangement of 26 circles in the unit square maximising the sum of radii.

    Returns:
        Tuple of (centers, radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
    """
    c, r = _search()
    if c is None:
        return np.full((N, 2), 0.5), np.zeros(N)
    for extra in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
        rr = r * (1.0 - extra)
        if _valid(c, rr):
            return c, rr
    return c, np.zeros(N)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

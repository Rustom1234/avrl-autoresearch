# EVOLVE-BLOCK-START
"""Circle packing for n=26: seeded multistart SLSQP + basin hopping, LP polish."""

import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, 1)
NP_ = len(I_IDX)


def _borders(c):
    return np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])


def _build_lin():
    A = np.zeros((4 * N, 3 * N))
    b = np.zeros(4 * N)
    for i in range(N):
        A[i, i] = 1; A[i, 2 * N + i] = -1                      # cx - r >= 0
        A[N + i, N + i] = 1; A[N + i, 2 * N + i] = -1          # cy - r >= 0
        A[2 * N + i, i] = -1; A[2 * N + i, 2 * N + i] = -1     # 1 - cx - r >= 0
        b[2 * N + i] = 1
        A[3 * N + i, N + i] = -1; A[3 * N + i, 2 * N + i] = -1
        b[3 * N + i] = 1
    return A, b


A_LIN, B_LIN = _build_lin()
ROWS = np.arange(NP_)


def _cons(v):
    cx, cy, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    pair = dx * dx + dy * dy - s * s
    return np.concatenate([pair, A_LIN @ v + B_LIN])


def _jac(v):
    cx, cy, r = v[:N], v[N:2 * N], v[2 * N:]
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    s = r[I_IDX] + r[J_IDX]
    Jp = np.zeros((NP_, 3 * N))
    Jp[ROWS, I_IDX] = 2 * dx
    Jp[ROWS, J_IDX] = -2 * dx
    Jp[ROWS, N + I_IDX] = 2 * dy
    Jp[ROWS, N + J_IDX] = -2 * dy
    Jp[ROWS, 2 * N + I_IDX] = -2 * s
    Jp[ROWS, 2 * N + J_IDX] = -2 * s
    return np.vstack([Jp, A_LIN])


OBJ_G = np.concatenate([np.zeros(2 * N), -np.ones(N)])
BOUNDS = [(0, 1)] * (2 * N) + [(0, 0.5)] * N


def _solve(c0, r0, maxiter=250):
    v0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
    try:
        res = minimize(lambda v: -np.sum(v[2 * N:]), v0, jac=lambda v: OBJ_G,
                       bounds=BOUNDS, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": _cons, "jac": _jac}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
    except Exception:
        return None
    v = res.x
    c = np.stack([v[:N], v[N:2 * N]], axis=1)
    c = np.clip(c, 0.0, 1.0)
    return c


def _lp_radii(c):
    """Optimal radii for fixed centers."""
    d = np.sqrt(((c[I_IDX] - c[J_IDX]) ** 2).sum(1))
    A = np.zeros((NP_, N))
    A[ROWS, I_IDX] = 1
    A[ROWS, J_IDX] = 1
    try:
        res = linprog(-np.ones(N), A_ub=A, b_ub=d,
                      bounds=[(0, max(b, 0)) for b in _borders(c)],
                      method="highs")
        if res.status == 0:
            return res.x
    except Exception:
        pass
    return None


def _valid(c, r):
    if np.any(r < 0):
        return False
    if np.any(c[:, 0] - r < 0) or np.any(c[:, 1] - r < 0):
        return False
    if np.any(c[:, 0] + r > 1) or np.any(c[:, 1] + r > 1):
        return False
    d = np.sqrt(((c[I_IDX] - c[J_IDX]) ** 2).sum(1))
    return not np.any(r[I_IDX] + r[J_IDX] > d)


def _score(c):
    r = _lp_radii(c)
    if r is None:
        return -1.0, None
    return float(r.sum()), r


def _init(rng, kind):
    if kind == 0:
        c = rng.uniform(0.05, 0.95, (N, 2))
    elif kind == 1:
        # perturbed 5x5 grid plus one extra
        g = (np.arange(5) + 0.5) / 5
        pts = np.array([(x, y) for x in g for y in g])
        pts = np.vstack([pts, rng.uniform(0.1, 0.9, (1, 2))])
        c = pts + rng.normal(0, 0.03, pts.shape)
    else:
        # perturbed hex-like rows 5,5,6,5,5
        rows = [5, 5, 6, 5, 5]
        pts = []
        for k, m in enumerate(rows):
            y = (k + 0.5) / 5
            for t in range(m):
                pts.append(((t + 0.5) / m, y))
        c = np.array(pts) + rng.normal(0, 0.03, (N, 2))
    c = np.clip(c, 0.03, 0.97)
    r0 = np.full(N, rng.uniform(0.06, 0.1))
    return c, r0


def construct_packing():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    budget = 170.0
    best_s, best_c, best_r = -1.0, None, None

    it = 0
    while time.time() - t0 < budget:
        elapsed = time.time() - t0
        if best_c is not None and elapsed > 0.45 * budget and rng.random() < 0.8:
            # basin hopping around the best
            c = best_c.copy()
            mode = rng.integers(3)
            if mode == 0:
                c += rng.normal(0, rng.choice([0.01, 0.03, 0.06]), c.shape)
            elif mode == 1:
                k = rng.integers(1, 4)
                idx = rng.choice(N, k, replace=False)
                c[idx] = rng.uniform(0.05, 0.95, (k, 2))
            else:
                # swap a small circle location with a random spot
                i = int(np.argmin(best_r))
                c[i] = rng.uniform(0.05, 0.95, 2)
                c += rng.normal(0, 0.01, c.shape)
            c = np.clip(c, 0.02, 0.98)
            r0 = best_r * 0.8
        else:
            c, r0 = _init(rng, it % 3)
        it += 1
        c2 = _solve(c, r0)
        if c2 is None:
            continue
        s, r = _score(c2)
        if r is not None and s > best_s:
            best_s, best_c, best_r = s, c2, r

    if best_c is None:
        # fallback: simple grid
        g = (np.arange(5) + 0.5) / 5
        pts = np.array([(x, y) for x in g for y in g] + [(0.5, 0.5)])
        best_c = pts
        best_r = np.full(N, 0.05)
        best_r[-1] = 0.0

    c = best_c
    r = np.maximum(best_r, 0.0)
    # shrink by a tiny margin until strictly valid in double precision
    r = r - 1e-12
    r = np.maximum(r, 0.0)
    for k in range(200):
        if _valid(c, r):
            break
        r = np.maximum(r * (1 - 1e-10 * (k + 1)) - 1e-12, 0.0)
    if not _valid(c, r):
        r = np.zeros(N)
    return c, r


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

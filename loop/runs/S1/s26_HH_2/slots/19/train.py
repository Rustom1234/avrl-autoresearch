# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # Then, limit by distance to other circles
            # Each pair of circles with centers at distance d can have
            # sum of radii at most d to avoid overlap
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

                    # If current radii would cause overlap
                    if radii[i] + radii[j] > dist:
                        # Scale both radii proportionally
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale

    # Ensure no negative radii due to numerical issues
    radii = np.maximum(radii, 0.0)
    return radii


from scipy.optimize import linprog
import time as _time

_N = 26
_MARGIN = 1e-9

from scipy.optimize import minimize as _minimize


def _slsqp_refine(c, r, maxiter=150):
    """Joint SLSQP over centers and radii: maximise sum r subject to pair and wall constraints."""
    n = c.shape[0]
    iu, ju = np.triu_indices(n, 1)

    def obj(z):
        return -np.sum(z[2 * n:])

    def obj_grad(z):
        g = np.zeros_like(z)
        g[2 * n:] = -1.0
        return g

    def cons(z):
        cc = z[:2 * n].reshape(n, 2)
        rr = z[2 * n:]
        d = np.sqrt(np.sum((cc[iu] - cc[ju]) ** 2, axis=1) + 1e-18)
        pair = d - rr[iu] - rr[ju]
        x, y = cc[:, 0], cc[:, 1]
        wall = np.concatenate([x - rr, y - rr, 1.0 - x - rr, 1.0 - y - rr])
        return np.concatenate([pair, wall])

    z0 = np.concatenate([c.ravel(), np.maximum(r, 0.0)])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    try:
        res = _minimize(obj, z0, jac=obj_grad, method="SLSQP", bounds=bounds,
                        constraints=[{"type": "ineq", "fun": cons}],
                        options={"maxiter": maxiter, "ftol": 1e-12})
        z = res.x
    except Exception:
        z = z0
    cc = np.clip(z[:2 * n].reshape(n, 2), 0.0, 1.0)
    rr = np.maximum(z[2 * n:], 0.0)
    return cc, rr


def _two_phase():
    """Phase 1: coarse global search keeping top five. Phase 2: stiff relax + SLSQP + LP per candidate."""
    rng = np.random.RandomState(2024)
    t0 = _time.time()
    pool = []
    while len(pool) < 12 and _time.time() - t0 < 60:
        c = rng.uniform(0.1, 0.9, size=(_N, 2))
        c, _r = _relax(c, np.full(_N, 0.01), 400, 0.3, 0.05)
        rv, s = _score(c)
        if rv is not None:
            pool.append((s, c.copy()))
    while len(pool) >= 2 and _time.time() - t0 < 100:
        i, j = rng.choice(len(pool), 2, replace=False)
        A = pool[i][1]
        B = pool[j][1]
        child = np.vstack([A[A[:, 0] < 0.5], B[B[:, 0] >= 0.5]])
        if child.shape[0] > _N:
            child = child[rng.choice(child.shape[0], _N, replace=False)]
        elif child.shape[0] < _N:
            child = np.vstack([child, rng.uniform(0.1, 0.9, size=(_N - child.shape[0], 2))])
        child = np.clip(child + rng.normal(0.0, 0.005, size=child.shape), 0.0, 1.0)
        c, _r = _relax(child, np.full(_N, 0.005), 300, 0.2, 0.02)
        rv, s = _score(c)
        if rv is None:
            continue
        worst = min(range(len(pool)), key=lambda k: pool[k][0])
        if s > pool[worst][0]:
            pool[worst] = (s, c.copy())
    pool.sort(key=lambda p: -p[0])
    del pool[5:]
    best_s, best_c, best_r = -1.0, None, None
    for s0, c0 in pool:
        if _time.time() - t0 > 140:
            break
        c, r = _relax(c0.copy(), np.full(_N, 0.005), 600, 0.05, 0.01, 0.5, c0)
        c, r = _slsqp_refine(c, r)
        rv, s = _score(c)
        if rv is not None and s > best_s:
            best_s, best_c, best_r = s, c.copy(), rv
        rm = _make_valid(c, r)
        sm = float(rm.sum())
        if sm > best_s:
            best_s, best_c, best_r = sm, c.copy(), rm
    if best_c is None:
        best_c = rng.uniform(0.1, 0.9, size=(_N, 2))
        best_r = np.zeros(_N)
    radii = _make_valid(best_c, best_r)
    return best_c, radii


def _template():
    rows = [5, 4, 5, 4, 4, 4]
    pts = []
    for k, m in enumerate(rows):
        y = (k + 0.5) / len(rows)
        for j in range(m):
            pts.append([(j + 0.5) / m, y])
    return np.clip(np.array(pts), 0.05, 0.95)


_TPL = _template()


def _lp_radii(c):
    """Exact LP: maximise sum of radii for fixed centers."""
    n = c.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = iu.shape[0]
    d = np.sqrt(np.sum((c[iu] - c[ju]) ** 2, axis=1))
    A = np.zeros((m, n))
    A[np.arange(m), iu] = 1.0
    A[np.arange(m), ju] = 1.0
    x, y = c[:, 0], c[:, 1]
    ub = np.maximum(np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]), 0.0)
    bounds = np.column_stack([np.zeros(n), ub])
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    return res.x


def _make_valid(c, r):
    """Clip radii to walls, enforce pair constraints exactly, add tiny margin."""
    n = c.shape[0]
    x, y = c[:, 0], c[:, 1]
    r = np.maximum(np.minimum.reduce([r, x, y, 1.0 - x, 1.0 - y]), 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
            s = r[i] + r[j]
            if s > d:
                f = d / s if s > 0 else 0.0
                r[i] *= f
                r[j] *= f
    r = r * (1.0 - _MARGIN)
    return np.maximum(r, 0.0)


def _relax(c, r, steps, lr_c, lr_r, lam=0.0, tpl=None):
    """Repulsion from overlaps and walls moves centers; radii grow into free space."""
    n = r.shape[0]
    eye = np.eye(n)
    for k in range(steps):
        decay = 1.0 - 0.9 * k / max(steps - 1, 1)
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1) + eye)
        gap = d - (r[:, None] + r[None, :])
        np.fill_diagonal(gap, np.inf)
        ov = np.maximum(-gap, 0.0)
        unit = diff / d[:, :, None]
        f = np.sum(ov[:, :, None] * unit, axis=1)
        gl = c[:, 0] - r
        gr = 1.0 - c[:, 0] - r
        gb = c[:, 1] - r
        gt = 1.0 - c[:, 1] - r
        f[:, 0] += np.maximum(-gl, 0.0) - np.maximum(-gr, 0.0)
        f[:, 1] += np.maximum(-gb, 0.0) - np.maximum(-gt, 0.0)
        if lam > 0.0 and tpl is not None:
            lk = lam * (1.0 - k / max(steps - 1, 1))
            f = f - lk * (c - tpl)
        c = np.clip(c + lr_c * decay * f, 0.0, 1.0)
        g = np.minimum(np.minimum.reduce([gl, gr, gb, gt]), gap.min(axis=1))
        r = np.maximum(r + lr_r * decay * g, 0.0)
    return c, r


def _score(c):
    r = _lp_radii(c)
    if r is None:
        return None, -1.0
    rv = _make_valid(c, r)
    return rv, float(rv.sum())


def construct_packing():
    T0 = _time.time()
    centers, radii = _two_phase()
    cur_c = centers.copy()
    cur_r = np.asarray(radii, dtype=float).copy()
    cur_s = float(cur_r.sum())
    best_c, best_r, best_s = cur_c.copy(), cur_r.copy(), cur_s
    rng = np.random.RandomState(7)
    T = 0.002
    while _time.time() - T0 < 215:
        c = cur_c.copy()
        u = rng.rand()
        if u < 0.3:
            c = _dual_step(c)
        elif u < 0.65:
            i0 = rng.randint(_N)
            g = np.linspace(0.02, 0.98, 49)
            G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
            others = np.delete(c, i0, axis=0)
            dd = np.sqrt(((G[:, None, :] - others[None, :, :]) ** 2).sum(-1)).min(axis=1)
            wall = np.minimum(np.minimum(G[:, 0], G[:, 1]), np.minimum(1 - G[:, 0], 1 - G[:, 1]))
            hole = np.minimum(dd, 2.0 * wall)
            c[i0] = G[np.argmax(hole)] + rng.normal(0.0, 0.003, size=2)
            c = np.clip(c, 0.0, 1.0)
        else:
            k = rng.randint(2, 5)
            idx = rng.choice(_N, k, replace=False)
            scale = rng.uniform(0.05, 0.15)
            c[idx] += rng.normal(0.0, scale, size=(k, 2))
            c = np.clip(c, 0.0, 1.0)
        c, rr = _relax(c, np.full(_N, 0.005), 200, 0.05, 0.01, 0.5, cur_c)
        c, rr = _slsqp_refine(c, rr, 80)
        rv, s = _score(c)
        if rv is None:
            continue
        if s > cur_s or rng.rand() < np.exp((s - cur_s) / T):
            cur_c, cur_r, cur_s = c.copy(), rv.copy(), s
            if s > best_s:
                best_c, best_r, best_s = c.copy(), rv.copy(), s
    return best_c, _make_valid(best_c, best_r)


def _dual_step(c, step=0.01):
    """LP-dual sensitivity move: step centers along dual-weighted pair gradients; keep only if the LP sum improves."""
    n = c.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = iu.shape[0]
    d = np.sqrt(np.sum((c[iu] - c[ju]) ** 2, axis=1))
    A = np.zeros((m, n))
    A[np.arange(m), iu] = 1.0
    A[np.arange(m), ju] = 1.0
    x, y = c[:, 0], c[:, 1]
    ub = np.maximum(np.minimum.reduce([x, y, 1.0 - x, 1.0 - y]), 0.0)
    bounds = np.column_stack([np.zeros(n), ub])
    res = linprog(-np.ones(n), A_ub=A, b_ub=d, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return c
    lam = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    unit = (c[iu] - c[ju]) / np.maximum(d, 1e-12)[:, None]
    g = np.zeros((n, 2))
    np.add.at(g, iu, lam[:, None] * unit)
    np.add.at(g, ju, -lam[:, None] * unit)
    gm = float(np.max(np.abs(g)))
    if gm < 1e-12:
        return c
    c_new = np.clip(c + step * g / gm, 0.0, 1.0)
    _, s0 = _score(c)
    _, s1 = _score(c_new)
    return c_new if s1 > s0 else c


def _old_construct_packing():
    """Inflation relaxation multistart, then basin hopping on the LP-scored packing."""
    rng = np.random.RandomState(2024)
    t0 = _time.time()
    best_c = rng.uniform(0.1, 0.9, size=(_N, 2))
    best_r = np.zeros(_N)
    best_s = 0.0
    for _ in range(30):
        c = rng.uniform(0.1, 0.9, size=(_N, 2))
        r = np.full(_N, 0.01)
        c, r = _relax(c, r, 1200, 0.3, 0.05, 0.1, _TPL)
        rv, s = _score(c)
        if rv is not None and s > best_s:
            best_s, best_c, best_r = s, c.copy(), rv
        if _time.time() - t0 > 100:
            break
    cur_c, cur_r, cur_s = best_c.copy(), best_r.copy(), best_s
    while _time.time() - t0 < 180:
        c = cur_c.copy()
        k = rng.randint(1, 4)
        idx = rng.choice(_N, k, replace=False)
        c[idx] = rng.uniform(0.05, 0.95, size=(k, 2))
        c = np.clip(c + rng.normal(0.0, 0.01, size=c.shape), 0.0, 1.0)
        r0 = np.maximum(cur_r * 0.8, 0.0)
        c, _ = _relax(c, r0, 300, 0.1, 0.02)
        rv, s = _score(c)
        if rv is not None and s > cur_s + 1e-9:
            cur_c, cur_r, cur_s = c.copy(), rv, s
    centers = cur_c
    radii = _make_valid(centers, cur_r)
    return centers, radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

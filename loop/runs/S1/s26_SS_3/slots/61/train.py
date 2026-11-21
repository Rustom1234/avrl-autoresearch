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


# EVOLVE-BLOCK-END


def solve_packing(time_budget=200.0):
    import time
    from scipy.optimize import minimize

    t0 = time.time()
    n = 26
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)
    ar = np.arange(m)
    N3 = 3 * n

    # constant wall Jacobian
    Jw = np.zeros((4 * n, N3))
    idx = np.arange(n)
    Jw[idx, idx] = 1.0
    Jw[idx, 2 * n + idx] = -1.0
    Jw[n + idx, idx] = -1.0
    Jw[n + idx, 2 * n + idx] = -1.0
    Jw[2 * n + idx, n + idx] = 1.0
    Jw[2 * n + idx, 2 * n + idx] = -1.0
    Jw[3 * n + idx, n + idx] = -1.0
    Jw[3 * n + idx, 2 * n + idx] = -1.0

    def cons(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        s = r[iu] + r[ju]
        return np.concatenate([
            dx * dx + dy * dy - s * s,
            x - r, 1 - x - r, y - r, 1 - y - r])

    def jac(z):
        x = z[:n]
        y = z[n:2 * n]
        r = z[2 * n:]
        dx = x[iu] - x[ju]
        dy = y[iu] - y[ju]
        s = r[iu] + r[ju]
        J = np.zeros((m, N3))
        J[ar, iu] = 2 * dx
        J[ar, ju] = -2 * dx
        J[ar, n + iu] = 2 * dy
        J[ar, n + ju] = -2 * dy
        J[ar, 2 * n + iu] = -2 * s
        J[ar, 2 * n + ju] = -2 * s
        return np.vstack([J, Jw])

    g = np.zeros(N3)
    g[2 * n:] = -1.0

    def obj(z):
        return -np.sum(z[2 * n:])

    def grad(z):
        return g

    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    rng = np.random.default_rng(0)

    def finalize(z):
        x = z[:n].copy()
        y = z[n:2 * n].copy()
        r = np.maximum(z[2 * n:] - 2e-9, 0.0)
        r = np.minimum(r, np.minimum.reduce([x, y, 1 - x, 1 - y]))
        r = np.maximum(r, 0.0)
        c = np.stack([x, y], axis=1)
        d = np.sqrt((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2)
        over = r[iu] + r[ju] - d
        for _ in range(5):
            bad = np.where(over > 0)[0]
            if len(bad) == 0:
                break
            for k in bad:
                i, j = iu[k], ju[k]
                s = r[i] + r[j]
                if s > 0:
                    f = d[k] / s * (1 - 1e-12)
                    r[i] *= f
                    r[j] *= f
            over = r[iu] + r[ju] - d
        if np.any(over > 0):
            return None
        return c, r

    best = None
    best_val = -1.0
    it = 0
    while time.time() - t0 < 0.7 * time_budget:
        it += 1
        c0 = rng.random((n, 2)) * 0.9 + 0.05
        pats = [[5, 5, 6, 5, 5], [6, 5, 4, 5, 6], [4, 5, 5, 5, 4, 3],
                [6, 5, 5, 5, 5], [5, 6, 5, 6, 4], [4, 5, 4, 5, 4, 4],
                [3, 5, 5, 5, 5, 3], [5, 4, 5, 4, 5, 3], [6, 5, 6, 5, 4]]
        if it % 2 == 1:
            pat = pats[(it // 2) % len(pats)]
            if rng.random() < 0.5:
                pat = pat[::-1]
            k = len(pat)
            pts = []
            for ri, mm in enumerate(pat):
                sh = (0.15 / mm) * (1 if ri % 2 else -1)
                for jj in range(mm):
                    pts.append([(jj + 0.5) / mm + sh, (ri + 0.5) / k])
            pts = np.array(pts)
            if len(pts) == n:
                c0 = np.clip(pts + rng.normal(0, 0.02, pts.shape), 0.03, 0.97)
        r0 = np.full(n, 0.05) + rng.random(n) * 0.03
        z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
        try:
            d2 = (c0[iu, 0] - c0[ju, 0]) ** 2 + (c0[iu, 1] - c0[ju, 1]) ** 2
            rows = np.concatenate([np.where(d2 < 0.16)[0], m + np.arange(4 * n)])
            res0 = minimize(obj, z0, jac=grad, method="SLSQP", bounds=bounds,
                            constraints=[{"type": "ineq", "fun": lambda z: cons(z)[rows],
                                          "jac": lambda z: jac(z)[rows]}],
                            options={"maxiter": 150, "ftol": 1e-9})
            res = minimize(obj, res0.x, jac=grad, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
                           options={"maxiter": 300, "ftol": 1e-11})
        except Exception:
            continue
        z = res.x
        if -res.fun < best_val - 0.02:
            continue
        out = finalize(z)
        if out is None:
            continue
        c, r = out
        v = r.sum()
        if v > best_val:
            best_val = v
            best = (c, r)
    while best is not None and time.time() - t0 < time_budget:
        c, r = best
        k = rng.choice(np.argsort(r)[:4])
        keep = np.arange(n) != k
        cc = c[keep]
        rr = r[keep]
        P = rng.random((3000, 2))
        dd = np.sqrt(((P[:, None, :] - cc[None]) ** 2).sum(2)) - rr[None]
        clear = np.minimum(dd.min(1), np.minimum(P.min(1), (1 - P).min(1)))
        top = np.argsort(-clear)[:10]
        p = P[rng.choice(top)]
        c0 = np.vstack([cc, p[None]])
        c0 = np.clip(c0 + rng.normal(0, 0.003, c0.shape), 0.01, 0.99)
        r0 = np.concatenate([rr * 0.97, [0.03]])
        z0 = np.concatenate([c0[:, 0], c0[:, 1], r0])
        try:
            res = minimize(obj, z0, jac=grad, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "ineq", "fun": cons, "jac": jac}],
                           options={"maxiter": 300, "ftol": 1e-11})
        except Exception:
            continue
        out = finalize(res.x)
        if out is None:
            continue
        c2, r2 = out
        if r2.sum() > best_val:
            best_val = r2.sum()
            best = (c2, r2)
    if best is None:
        c = np.stack([np.linspace(0.05, 0.95, n), np.full(n, 0.5)], axis=1)
        return c, np.full(n, 1e-3)
    return best


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = solve_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

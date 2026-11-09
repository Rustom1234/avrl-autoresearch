"""AR3 statistics helpers (see exp/AR3_DESIGN.md section 10).

Pure functions, no I/O. Import from ar3_analyze.py; run this file directly to execute the unit tests.
  wilson, clopper_pearson   binomial intervals (CP has one-sided variants, exact for zero counts)
  holm                      step-down adjusted p-values
  mcnemar_exact, mcnemar_midp   paired binary test from the discordant counts
  cluster_bootstrap         resample clusters (entries), recompute a user statistic on the resampled frame
  signflip_perm             paired sign-flip permutation test on per-cluster differences
  weighted_auroc            AUROC with per-item weights (ties count one half)
  weighted_quantile         threshold t and tie probability p that hit an exact target fraction
  mde, n_star, power_row    minimum detectable effect and the N needed for a target delta
"""
import numpy as np
import pandas as pd
from scipy import stats

# one-sided alpha 0.05 and power 0.80 (z 1.645 + z 0.842, rounded as in the design)
MDE_K = 2.485


def wilson(k, n, alpha=0.05):
    """Two-sided Wilson score interval for k successes in n trials. Returns (lo, hi)."""
    if n <= 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def clopper_pearson(k, n, alpha=0.05, alternative="two-sided"):
    """Exact (Clopper-Pearson) interval. Returns (lo, hi) always.
    alternative 'two-sided': alpha/2 in each tail. 'upper': one-sided upper bound at level 1-alpha, lo = 0
    (for k = 0 this is 1 - alpha**(1/n)). 'lower': one-sided lower bound, hi = 1."""
    if alternative not in ("two-sided", "upper", "lower"):
        raise ValueError(alternative)
    if n <= 0:
        return (0.0, 1.0)
    a_lo = alpha / 2 if alternative == "two-sided" else alpha
    a_hi = a_lo
    lo = 0.0 if (k == 0 or alternative == "upper") else float(stats.beta.ppf(a_lo, k, n - k + 1))
    hi = 1.0 if (k == n or alternative == "lower") else float(stats.beta.ppf(1 - a_hi, k + 1, n - k))
    return (lo, hi)


def holm(pvals):
    """Holm step-down adjusted p-values, returned in input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mcnemar_exact(n10, n01):
    """Exact two-sided McNemar p from discordant counts (n10 = A right and B wrong, n01 = the reverse)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5)))


def mcnemar_midp(n10, n01):
    """Mid-p two-sided McNemar p: exact p minus the point probability of the observed minimum."""
    n = n10 + n01
    if n == 0:
        return 1.0
    m = min(n10, n01)
    return float(min(1.0, 2 * stats.binom.cdf(m, n, 0.5) - stats.binom.pmf(m, n, 0.5)))


def cluster_bootstrap(df, cluster_col, stat_fn, B=10000, seed=0, alternative="two-sided", conf=0.95):
    """Cluster bootstrap. Each resample draws the unique values of df[cluster_col] with replacement,
    stacks their rows, and calls stat_fn(frame) -> float. The frame carries an extra column '_boot_cluster'
    (also added for the point estimate on the original frame) that is unique per draw, so a statistic that groups by cluster treats repeated draws as separate clusters.
    Per-item weights in df stay fixed; renormalizing inside stat_fn is the caller's job.
    alternative: 'two-sided' gives the central conf interval (percentiles (1-conf)/2 and 1-(1-conf)/2),
    'greater' gives a one-sided lower bound (hi = inf) at the (1-conf) percentile, 'less' a one-sided upper
    bound (lo = -inf) at the conf percentile. Returns dict(est, lo, hi, n_clusters, B_ok, boot)."""
    groups = df.groupby(cluster_col, sort=False).indices
    keys = list(groups.keys())
    idx = [np.asarray(groups[k]) for k in keys]
    nc = len(keys)
    rng = np.random.default_rng(seed)
    base = df.reset_index(drop=True)
    est = float(stat_fn(base.assign(_boot_cluster=pd.factorize(base[cluster_col])[0])))
    out = np.empty(B)
    for b in range(B):
        draw = rng.integers(0, nc, nc)
        rows = np.concatenate([idx[j] for j in draw])
        lab = np.concatenate([np.full(len(idx[j]), r) for r, j in enumerate(draw)])
        fr = base.iloc[rows].reset_index(drop=True)
        fr["_boot_cluster"] = lab
        out[b] = stat_fn(fr)
    ok = out[~np.isnan(out)]
    if len(ok) == 0:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "n_clusters": nc, "B_ok": 0, "boot": out}
    a = 1 - conf
    if alternative == "two-sided":
        lo, hi = np.percentile(ok, [100 * a / 2, 100 * (1 - a / 2)])
    elif alternative == "greater":
        lo, hi = np.percentile(ok, 100 * a), np.inf
    elif alternative == "less":
        lo, hi = -np.inf, np.percentile(ok, 100 * (1 - a))
    else:
        raise ValueError(alternative)
    return {"est": est, "lo": float(lo), "hi": float(hi), "n_clusters": nc, "B_ok": int(len(ok)), "boot": out}


def signflip_perm(diffs, n_perm=9999, seed=0, alternative="two-sided"):
    """Sign-flip permutation test on paired per-cluster differences, statistic = mean.
    alternative 'two-sided' (|mean|), 'greater' (mean > 0) or 'less' (mean < 0).
    p = (1 + #{perm as extreme as observed}) / (n_perm + 1). Returns dict(p, obs, n)."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    obs = float(d.mean()) if n else float("nan")
    if n == 0:
        return {"p": float("nan"), "obs": obs, "n": 0}
    rng = np.random.default_rng(seed)
    tol = 1e-12
    hits = 0
    done = 0
    chunk = max(1, min(n_perm, 2_000_000 // n))
    while done < n_perm:
        c = min(chunk, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(c, n))
        m = (signs * d).mean(axis=1)
        if alternative == "two-sided":
            hits += int(np.sum(np.abs(m) >= abs(obs) - tol))
        elif alternative == "greater":
            hits += int(np.sum(m >= obs - tol))
        elif alternative == "less":
            hits += int(np.sum(m <= obs + tol))
        else:
            raise ValueError(alternative)
        done += c
    return {"p": (1 + hits) / (n_perm + 1), "obs": obs, "n": n}


def weighted_auroc(y, score, w=None):
    """AUROC of score for positives y == 1 against y == 0 with item weights w (default 1).
    Pair weight is w_i * w_j; tied scores count one half. Equals sklearn roc_auc_score(sample_weight=w).
    Returns nan if either class has zero weight."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    wp = np.where(y == 1, w, 0.0)
    wn = np.where(y == 0, w, 0.0)
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0:
        return float("nan")
    u, inv = np.unique(s, return_inverse=True)
    gp = np.bincount(inv, weights=wp, minlength=len(u))
    gn = np.bincount(inv, weights=wn, minlength=len(u))
    neg_below = np.cumsum(gn) - gn
    return float(np.sum(gp * (neg_below + 0.5 * gn)) / (Wp * Wn))


def weighted_quantile(x, w, frac):
    """Threshold (t, p) so that escalating x < t always, and x == t with probability p, escalates exactly
    weight fraction frac of the total weight (analytic tie handling, no randomness needed to report it).
    frac <= 0 gives p = 0 at the minimum; frac >= 1 gives p = 1 at the maximum."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    frac = float(min(max(frac, 0.0), 1.0))
    u, inv = np.unique(x, return_inverse=True)
    g = np.bincount(inv, weights=w, minlength=len(u))
    cum = np.cumsum(g)
    j = int(np.searchsorted(cum, frac - 1e-12))
    j = min(j, len(u) - 1)
    below = cum[j] - g[j]
    p = 0.0 if g[j] <= 0 else float(min(1.0, max(0.0, (frac - below) / g[j])))
    return float(u[j]), p


def escalation_prob(x, t, p):
    """Per-item probability of escalation under (t, p): 1 if x < t, p if x == t, else 0."""
    x = np.asarray(x, dtype=float)
    return np.where(x < t, 1.0, np.where(x == t, p, 0.0))


def mde(sd, n, k=MDE_K):
    """Minimum detectable paired difference at n clusters: k * sd / sqrt(n)."""
    return k * sd / np.sqrt(n)


def n_star(sd, delta, k=MDE_K):
    """Clusters needed to detect a paired difference delta: (k * sd / delta)^2."""
    return (k * sd / delta) ** 2


def power_row(sd, n, delta):
    """MDE at n, N* for delta, and q = n / N* (q >= 1 means the test is powered for delta)."""
    ns = n_star(sd, delta)
    return {"sd": float(sd), "n": int(n), "mde": float(mde(sd, n)), "delta": float(delta), "n_star": float(ns), "q": float(n / ns)}


def _tests():
    def close(a, b, tol, what):
        assert abs(a - b) <= tol, f"{what}: {a} vs {b}"

    lo, hi = wilson(5, 12)
    close(lo, 0.193, 0.001, "wilson lo"), close(hi, 0.680, 0.001, "wilson hi")
    lo, hi = clopper_pearson(5, 12)
    close(lo, 0.152, 0.001, "cp lo"), close(hi, 0.723, 0.001, "cp hi")
    lo, hi = clopper_pearson(0, 20, 0.05, "upper")
    close(lo, 0.0, 0, "cp0 lo"), close(hi, 1 - 0.05 ** (1 / 20), 1e-12, "cp0 upper")
    close(clopper_pearson(20, 20, 0.05, "lower")[0], 0.05 ** (1 / 20), 1e-12, "cp all-success lower")
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    close(mcnemar_exact(5, 1), 0.219, 0.001, "mcnemar exact")
    close(mcnemar_midp(5, 1), 0.125, 0.001, "mcnemar mid-p")
    assert mcnemar_exact(0, 0) == 1.0 and abs(mcnemar_midp(3, 3) - 1.0) < 1e-12  # tie: 2*cdf - pmf = 1.3125 - 0.3125

    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    for s in (rng.normal(size=200), rng.integers(0, 5, 200).astype(float)):  # continuous and heavily tied
        w = rng.uniform(0.1, 3, 200)
        close(weighted_auroc(y, s, w), roc_auc_score(y, s, sample_weight=w), 1e-12, "weighted auroc")
        close(weighted_auroc(y, s), roc_auc_score(y, s), 1e-12, "unweighted auroc")
    assert np.isnan(weighted_auroc([1, 1], [0.1, 0.2]))

    # weighted quantile: tied example, equal weights. 3 of 8 below 1.0 is not reachable without ties at 0.5
    x = np.array([0.2, 0.5, 0.5, 0.5, 1, 1, 1, 1])
    w = np.ones(8)
    t, p = weighted_quantile(x, w, 0.25)
    assert t == 0.5 and abs(p - 1 / 3) < 1e-12, (t, p)
    close(float((escalation_prob(x, t, p) * w / w.sum()).sum()), 0.25, 1e-12, "tied target")
    for frac in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):  # random tied weighted data hits every target
        xr = rng.integers(0, 4, 60) / 3
        wr = rng.uniform(0.05, 2, 60)
        t, p = weighted_quantile(xr, wr, frac)
        close(float((escalation_prob(xr, t, p) * wr).sum() / wr.sum()), frac, 1e-9, f"quantile frac {frac}")
        assert 0 <= p <= 1

    # sign-flip: strong shift gives small p, centered noise does not; one-sided direction respected
    d = rng.normal(1.0, 1, 40)
    assert signflip_perm(d, 9999, 0)["p"] < 0.01
    assert signflip_perm(d, 9999, 0, "less")["p"] > 0.9
    dn = rng.normal(0, 1, 40)  # permutation p tracks the t-test p
    close(signflip_perm(dn, 9999, 0)["p"], stats.ttest_1samp(dn, 0).pvalue, 0.03, "signflip vs t")
    assert signflip_perm(np.zeros(5), 999, 0)["p"] == 1.0

    # cluster bootstrap: mean of cluster-structured data, interval brackets estimate, one-sided bounds order
    cl = np.repeat(np.arange(30), 4)
    df = pd.DataFrame({"c": cl, "v": rng.normal(0, 1, 120) + np.repeat(rng.normal(1, 1, 30), 4)})
    f = lambda fr: fr["v"].mean()
    r2 = cluster_bootstrap(df, "c", f, B=1000, seed=0)
    assert r2["lo"] < r2["est"] < r2["hi"] and r2["n_clusters"] == 30 and r2["B_ok"] == 1000
    rg = cluster_bootstrap(df, "c", f, B=1000, seed=0, alternative="greater")
    rl = cluster_bootstrap(df, "c", f, B=1000, seed=0, alternative="less")
    assert rg["lo"] > r2["lo"] and rg["hi"] == np.inf and rl["hi"] < r2["hi"] and rl["lo"] == -np.inf
    assert cluster_bootstrap(df, "c", f, B=50, seed=3)["boot"].tolist() == cluster_bootstrap(df, "c", f, B=50, seed=3)["boot"].tolist()
    # per-draw cluster ids are unique, so a groupby-based statistic sees 30 clusters every time
    nclu = lambda fr: fr["_boot_cluster"].nunique()
    assert set(cluster_bootstrap(df, "c", nclu, B=20, seed=0)["boot"]) == {30.0}
    # bootstrap SE of a cluster mean is close to the analytic cluster SE
    se = r2["boot"].std()
    se_an = df.groupby("c")["v"].mean().std() / np.sqrt(30)
    assert abs(se - se_an) / se_an < 0.15, (se, se_an)

    close(mde(0.2, 100), 2.485 * 0.2 / 10, 1e-12, "mde")
    close(n_star(0.2, 0.05), (2.485 * 0.2 / 0.05) ** 2, 1e-9, "n_star")
    close(mde(0.2, n_star(0.2, 0.05)), 0.05, 1e-12, "mde at n_star equals delta")
    close(power_row(0.2, 50, 0.05)["q"], 50 / n_star(0.2, 0.05), 1e-12, "q")
    print("ar3_stats: all tests passed")


if __name__ == "__main__":
    _tests()

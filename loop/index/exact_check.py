#!/usr/bin/env python3
"""Exact checks of the cost-of-pass index rule on small problems where the options are NOT independent.

    python loop/index/exact_check.py --n 300

No model calls. Every number is an exact expectation computed by dynamic programming over belief states.

Check 1 (one stage, two models, shared unknown): the task has a hidden type; the type sets the success chance of
both models. An attempt costs call price + experiment price. Goal: first success at least expected cost.
Check 2 (two stages): the same, but two successes in a row are needed and the option to give up (lose W) exists.
Check 3 (redraw): one model; each new child program is hidden good or bad; choose between trying a child again
and drawing a new child.

Compared against the Bayes-optimal policy (OPT):
  index       attempt the option with the lowest cost / success chance, beliefs updated with everything seen
  index_indep the same rule with one separate belief per option (ignores what other options reveal)
  escalate3   three cheap attempts, then the strong model
  single      the better single model with its own optimal stopping (chosen with hindsight per problem)
"""
import argparse, random
from statistics import fmean, median

A = (0.01, 0.08)  # call price in USD of the cheap and the strong model (S1 single-call control)
INF = float("inf")
DEBUG, WORST = False, [""]


def summary(name, ratios):
    r = sorted(ratios)
    q = lambda p: r[min(len(r) - 1, int(p * len(r)))]
    print("  %-12s mean %.3f  median %.3f  95th %.3f  worst %.3f  within 1%%: %3.0f%%  within 5%%: %3.0f%%" % (
        name, fmean(r), median(r), q(0.95), r[-1], 100 * sum(x <= 1.01 for x in r) / len(r), 100 * sum(x <= 1.05 for x in r) / len(r)))


def chain(rng, K, n_inst, cap, structured, stop, pmin=0.05):
    res = {k: [] for k in ("index", "index_indep", "escalate3", "single")}
    trivial = 0
    for _ in range(n_inst):
        Z = rng.choice([2, 3, 4])
        pi = [rng.random() + 0.2 for _ in range(Z)]
        pi = [x / sum(pi) for x in pi]
        if structured:  # shared difficulty per type, the strong model better by a fixed margin, later stages harder
            import math
            gap = rng.uniform(0.0, 1.5)
            P = []
            for z in range(Z):
                d = rng.uniform(-1.0, 3.0)
                P.append([[min(0.9, max(pmin, 1 / (1 + math.exp(d + 0.8 * k - gap * m + rng.gauss(0, 0.3))))) for m in (0, 1)] for k in range(K)])
        else:  # no structure at all: every chance drawn on its own (can be negatively related)
            import math
            P = [[[math.exp(rng.uniform(math.log(pmin), math.log(0.7))) for m in (0, 1)] for k in range(K)] for z in range(Z)]
        e = rng.choice([0.004, 0.06, 0.3])
        c = (A[0] + e, A[1] + e)
        W = rng.choice([0.5, 2.0, 8.0]) if stop else None

        pm = {}

        def post(k, f, win):
            key = (k, f, win)
            if key not in pm:
                w = []
                for z in range(Z):
                    x = pi[z]
                    for j in range(K):
                        x *= (1 - P[z][j][0]) ** f[2 * j] * (1 - P[z][j][1]) ** f[2 * j + 1]
                    for j in range(k):
                        x *= P[z][j][win[j]]
                    w.append(x)
                s = sum(w)
                pm[key] = [x / s for x in w]
            return pm[key]

        def pred(w, j, m):
            return sum(w[z] * P[z][j][m] for z in range(Z))

        def tail(k, w):
            return W if stop else min(sum(w[z] * c[m] / P[z][k][m] for z in range(Z)) for m in (0, 1))

        def bump(f, i):
            return f[:i] + (f[i] + 1,) + f[i + 1:]

        def solve(allowed):
            memo = {}

            def V(k, f, win):
                if k == K:
                    return 0.0
                key = (k, f, win)
                if key in memo:
                    return memo[key]
                w = post(k, f, win)
                if f[2 * k] + f[2 * k + 1] >= cap:
                    v = tail(k, w)
                else:
                    v = W if stop else INF
                    for m in allowed:
                        ph = pred(w, k, m)
                        v = min(v, c[m] + (1 - ph) * V(k, bump(f, 2 * k + m), win) + ph * V(k + 1, f, win + (m,)))
                memo[key] = v
                return v

            return V(0, (0,) * (2 * K), ())

        def evaluate(policy):
            memo = {}

            def U(k, f, win):
                if k == K:
                    return 0.0
                key = (k, f, win)
                if key in memo:
                    return memo[key]
                w = post(k, f, win)
                if f[2 * k] + f[2 * k + 1] >= cap:
                    v = tail(k, w)
                else:
                    m = policy(k, f, win, w)
                    if m is None:
                        v = W
                    else:
                        ph = pred(w, k, m)
                        v = c[m] + (1 - ph) * U(k, bump(f, 2 * k + m), win) + ph * U(k + 1, f, win + (m,))
                memo[key] = v
                return v

            return U(0, (0,) * (2 * K), ())

        def rule(k, belief, force=None):
            """lowest cost of pass; stop when it is no smaller than the worth of passing this stage"""
            cp = [c[m] / belief(k, m) for m in (0, 1)]
            m = min((0, 1), key=lambda i: cp[i])
            if stop:
                later = sum(min(c[i] / belief(j, i) for i in (0, 1)) for j in range(k + 1, K))
                if cp[m] >= W - later:
                    return None
            return m if force is None else force

        def index(k, f, win, w):
            return rule(k, lambda j, m: pred(w, j, m))

        def indep_belief(f):
            def b(j, m):
                n = f[2 * j + m]
                num = sum(pi[z] * (1 - P[z][j][m]) ** n * P[z][j][m] for z in range(Z))
                return num / sum(pi[z] * (1 - P[z][j][m]) ** n for z in range(Z))
            return b

        def index_indep(k, f, win, w):
            return rule(k, indep_belief(f))

        def escalate3(k, f, win, w):
            return rule(k, lambda j, m: pred(w, j, m), force=0 if f[2 * k] < 3 else 1)

        opt = solve((0, 1))
        if stop and opt >= W - 1e-12:
            trivial += 1
            continue
        res["index"].append(evaluate(index) / opt)
        if DEBUG and K == 1 and res["index"][-1] >= max(res["index"]):
            def path(policy):  # actions along the all-fail path
                f, s = (0, 0), ""
                while f[0] + f[1] < cap:
                    m = policy(0, f, (), post(0, f, ()))
                    if m is None:
                        break
                    s += "HS"[m]
                    f = bump(f, m)
                return s or "(give up)"

            def opt_policy(k, f, win, w):  # greedy with respect to the optimal values
                best, arg = (W if stop else INF), None
                for m in (0, 1):
                    ph = pred(w, 0, m)
                    v = c[m] + (1 - ph) * solve_from(bump(f, m))
                    if v < best - 1e-12:
                        best, arg = v, m
                return arg

            def solve_from(f0):
                memo = {}

                def V(f):
                    if f in memo:
                        return memo[f]
                    w = post(0, f, ())
                    if f[0] + f[1] >= cap:
                        v = tail(0, w)
                    else:
                        v = W if stop else INF
                        for m in (0, 1):
                            v = min(v, c[m] + (1 - pred(w, 0, m)) * V(bump(f, m)))
                    memo[f] = v
                    return v

                return V(f0)

            WORST[0] = "ratio %.3f  W %s  cost H %.3f S %.3f  prior %s  chances (H,S) per type %s\n    index: %s\n    OPT:   %s" % (
                res["index"][-1], W, c[0], c[1], [round(x, 2) for x in pi], [[round(x, 2) for x in P[z][0]] for z in range(Z)], path(index), path(opt_policy))
        res["index_indep"].append(evaluate(index_indep) / opt)
        res["escalate3"].append(evaluate(escalate3) / opt)
        res["single"].append(min(solve((0,)), solve((1,))) / opt)
    print("%d stage(s), %s chances, %s: %d problems (%d more where giving up at once is optimal)" % (
        K, "structured" if structured else "unstructured", "may give up" if stop else "must finish", len(res["index"]), trivial))
    for k, v in res.items():
        summary(k, v)
    if DEBUG and K == 1:
        print("  worst case for index: " + WORST[0])


def redraw(rng, n_inst, Cmax=12, F=6):
    res = {k: [] for k in ("index_simple", "index_restart", "never_redraw", "always_fresh")}
    trivial = 0
    for _ in range(n_inst):
        r, q, g = rng.uniform(0.1, 0.6), rng.uniform(0.2, 0.7), rng.uniform(0.2, 0.8)
        b = rng.choice([0.0, rng.uniform(0.0, 0.05)])
        c, W = 1.0, rng.choice([20.0, 60.0, 200.0])
        S = lambda i: q * (1 - g) ** i + (1 - q) * (1 - b) ** i  # chance a child has not succeeded after i tries
        ph = lambda f: (S(f) - S(f + 1)) / S(f)
        rho = min((c / r + c * sum(S(i) for i in range(n))) / (1 - S(n)) for n in range(1, F + 1))  # best restart cutoff

        def step(policy):
            memo = {}

            def V(st):
                if st in memo:
                    return memo[st]
                alive = sorted({f for f in st if f < F})
                acts = ([("root", None)] if len(st) < Cmax else []) + [("child", f) for f in alive]
                if policy is not None:
                    a = policy(st, alive, len(st) < Cmax)
                    acts = [] if a is None else [a]
                v = W if (policy is None or not acts) else INF
                for kind, f in acts:
                    if kind == "root":
                        v = min(v, c / r + V(tuple(sorted(st + (0,)))))
                    else:
                        i = st.index(f)
                        v = min(v, c + (1 - ph(f)) * V(tuple(sorted(st[:i] + (f + 1,) + st[i + 1:]))))
                memo[st] = v
                return v

            return V(())

        def by_index(root_cost):
            def pol(st, alive, root_ok):
                opts = []
                if alive:
                    opts.append((W - c / ph(alive[0]), ("child", alive[0])))
                if root_ok:
                    opts.append((W - root_cost, ("root", None)))
                if not opts:
                    return None
                best = max(opts, key=lambda o: o[0])
                return best[1] if best[0] > 0 else None
            return pol

        def never_redraw(st, alive, root_ok):
            if not st:
                return ("root", None) if W - c / r - c / ph(0) > 0 else None
            return ("child", alive[0]) if alive and W - c / ph(alive[0]) > 0 else None

        def always_fresh(st, alive, root_ok):
            if (c / r + c) / ph(0) >= W:
                return None
            if alive and alive[0] == 0:
                return ("child", 0)
            if root_ok:
                return ("root", None)
            return ("child", alive[0]) if alive else None

        opt = step(None)
        if opt >= W - 1e-12:
            trivial += 1
            continue
        res["index_simple"].append(step(by_index(c / r + c / ph(0))) / opt)
        res["index_restart"].append(step(by_index(rho)) / opt)
        res["never_redraw"].append(step(never_redraw) / opt)
        res["always_fresh"].append(step(always_fresh) / opt)
    print("redraw problem (hidden good or bad children): %d problems (%d more where giving up at once is optimal)" % (len(res["index_simple"]), trivial))
    for k, v in res.items():
        summary(k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--debug", action="store_true", help="print the worst one-stage problem for the index rule")
    a = ap.parse_args()
    DEBUG = a.debug
    rng = random.Random(0)
    print("ratio = expected cost of the policy / expected cost of the Bayes-optimal policy (1.000 is optimal)\n")
    # caps on attempts are set high enough that they almost never bind (a binding cap is a slot budget, a different problem)
    for structured in (True, False):
        chain(rng, 1, a.n, 120, structured, stop=False)
    for structured in (True, False):
        chain(rng, 1, a.n, 150, structured, stop=True)
    redraw(rng, a.n)
    for structured in (True, False):
        chain(rng, 2, max(10, a.n // 5), 20, structured, stop=True, pmin=0.25)

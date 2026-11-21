#!/usr/bin/env python3
"""SR2: keep refining, start over, pay for a stronger model, or stop, decided by a Markov-process index.
Rules and claims: loop/design/SR2_PREREG.md.

    chain_policy.py --chains loop/runs/SR2 --fresh loop/runs/SR1 [--worlds 100] [--boot 200] [--scale fable=5]
    chain_policy.py --selftest

Stored per problem: Haiku chains (each step is shown the best candidates of the earlier steps of its chain; a chain
ends at the first success or after 12 steps) and fresh Sonnet and Fable attempts. On a problem a policy can
continue a started chain, start a new chain, or pay for a fresh Sonnet or Fable attempt. All problems of a
portfolio share one budget.

The rule: a chain is a Markov process with state (steps so far, best training error so far). Its grade is
    min over horizons T of (expected dollars over the next T steps) / (chance of success within T steps),
the index of Dumitriu, Tetali and Winkler (2003), a variant of the Gittins index. The option with the lowest grade
over all problems is taken next (greedy knapsack). Expected dollars and chances come from a memory of other
problems: the problem is taken to behave like one of them, weighted by how well each explains what was seen.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, heapq, json, math, re, sys, zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import FRACS, auc, fold_of, logerr  # noqa: E402

STEPS = 12          # longest chain
BW = 1.0            # kernel width, in decades of training error (as in SR1)
KAPPA = 1.0         # weight of the pooled prior inside each remembered problem (as in SR1)
FRESH = ("sonnet", "fable")
RULE = "THE RULE (index with look-ahead)"
REF = "reference: knows the problem (not attainable)"
HIND = "plan: best (K, R, s) in hindsight (bound)"


# ------------------------------------------------------------------ data
class Data:
    """chains[p]: list of chains; a chain = list of steps (dollars, solved, log10 of the chain's best training error
    so far). fresh[p][model]: list of attempts (dollars, solved, log10 training error)."""

    def __init__(self, chains, fresh, scale=None):
        s = scale or {}
        self.problems = sorted(chains)
        self.chains = {p: [[(c * s.get("haiku", 1.0), ok, e) for c, ok, e in ch] for ch in chains[p]]
                       for p in self.problems}
        self.fresh = {p: {m: [(c * s.get(m, 1.0), ok, e) for c, ok, e in fresh[p][m]] for m in FRESH}
                      for p in self.problems}

    def price(self, problems):
        """Mean dollars of one Haiku step and of one fresh attempt on these problems (what a policy may know)."""
        out = {m: float(np.mean([c for p in problems for c, _, _ in self.fresh[p][m]])) for m in FRESH}
        out["haiku"] = float(np.mean([c for p in problems for ch in self.chains[p] for c, _, _ in ch]))
        return out

    def bmax(self, problems):
        """Largest budget: one Fable attempt on every problem."""
        return float(np.mean([c for p in problems for c, _, _ in self.fresh[p]["fable"]])) * len(problems)


def load(chain_root, fresh_root, tau=1e-10):
    import bank as B
    steps = defaultdict(dict)
    for f in Path(chain_root).glob("trie/*/r*/**/record.json"):
        r = json.loads(f.read_text())
        if set(r["path"]) == {"haiku"}:
            steps[(r["problem"], r["rep"])][len(r["path"])] = r
    chains = defaultdict(list)
    for (p, rep), d in sorted(steps.items()):
        assert sorted(d) == list(range(1, len(d) + 1)), (p, rep)
        ch, best = [], 2.0
        for j in range(1, min(len(d), STEPS) + 1):
            r = d[j]
            best = min(best, logerr((r.get("best") or {}).get("train_nmse")))
            ok = r.get("val") is not None and r["val"] <= tau
            ch.append((r["cost_usd"], ok, best))
            if ok:
                break
        chains[p].append(ch)
    fresh = {p: {m: [] for m in FRESH} for p in chains}
    for r in sorted(B.load(fresh_root), key=lambda r: r["k"]):
        if r["model"] in FRESH and r["problem"] in fresh:
            fresh[r["problem"]][r["model"]].append((r["cost_usd"], r.get("val") is not None and r["val"] <= tau,
                                                    logerr((r.get("best") or {}).get("train_nmse"))))
    return dict(chains), fresh


def level(p):
    m = re.search(r"_L(\d+)_", p)
    return int(m.group(1)) if m else 0


# ------------------------------------------------------------------ beliefs from the memory of other problems
class Memory:
    """The remembered problems: every chain of theirs step by step, and their fresh attempts."""

    def __init__(self, D, problems, price):
        self.price, n = price, len(problems)
        ch = [(i, c) for i, p in enumerate(problems) for c in D.chains[p]]
        R, k = len(ch), np.arange(STEPS + 1)
        self.M = np.zeros((n, R))                               # remembered problem x remembered chain
        self.E, usd = np.zeros((R, STEPS + 1)), np.zeros((R, STEPS + 1))
        for r, (i, c) in enumerate(ch):
            self.M[i, r] = 1.0
            for j, (cst, _, e) in enumerate(c, 1):
                self.E[r, j], usd[r, j] = e, cst
        L = np.array([len(c) for _, c in ch])[:, None]
        sol = np.array([bool(c[-1][1]) for _, c in ch])[:, None]
        self.cum = usd.cumsum(1)                                # dollars of the first k steps
        self.alive = (L > k) | ((L == k) & ~sol)                # unsolved after step k (k = 0: every chain)
        self.fut = L > k                                        # has a stored step after step k
        self.won = sol & (L <= k)                               # solved within the first k steps
        self.nch = self.M.sum(1)
        self.q = self.alive.mean(0)
        self.mu = [float(self.E[self.alive[:, j], j].mean()) if j and self.alive[:, j].any() else -3.0 for j in k]
        self.sd = [float(max(self.E[self.alive[:, j], j].std(), 1.5)) if j and self.alive[:, j].any() else 3.0 for j in k]
        src = lambda p, m: D.fresh[p][m]
        S = np.array([[sum(ok for _, ok, _ in src(p, m)) for m in FRESH] for p in problems], float).reshape(n, len(FRESH))
        F = np.array([[sum(not ok for _, ok, _ in src(p, m)) for m in FRESH] for p in problems], float).reshape(n, len(FRESH))
        self.Ef = [[np.array([le for _, ok, le in src(p, m) if not ok]) for m in FRESH] for p in problems]
        pooled = (S.sum(0) + 0.5) / (S.sum(0) + F.sum(0) + 1.0)
        self.A, self.B = S + KAPPA * pooled, F + KAPPA * (1 - pooled)
        allE = [np.concatenate([e[a] for e in self.Ef] or [np.zeros(0)]) for a in range(len(FRESH))]
        self.f_mu = [float(x.mean()) if len(x) else -3.0 for x in allE]
        self.f_sd = [float(max(x.std(), 1.5)) if len(x) else 3.0 for x in allE]

    def _dens(self, j, e, use_error):
        """How likely a chain that is unsolved after step j with best error e is, per remembered problem."""
        a = self.alive[:, j].astype(float)
        if use_error:
            a = a * np.exp(-0.5 * ((e - self.E[:, j]) / BW) ** 2) / (BW * 2.5066)
            prior = self.q[j] * math.exp(-0.5 * ((e - self.mu[j]) / self.sd[j]) ** 2) / (self.sd[j] * 2.5066)
        else:
            prior = self.q[j]
        return np.maximum((self.M @ a + KAPPA * prior) / (self.nch + KAPPA), 1e-300)

    def _fdens(self, a, x):
        """How likely the log error x of a failed fresh attempt of model a is, per remembered problem."""
        prior = math.exp(-0.5 * ((x - self.f_mu[a]) / self.f_sd[a]) ** 2) / (self.f_sd[a] * 2.5066)
        return np.array([(float(np.exp(-0.5 * ((x - e[a]) / BW) ** 2).sum()) / (BW * 2.5066) + KAPPA * prior)
                         / (len(e[a]) + KAPPA) for e in self.Ef])

    def weights(self, chains, ffail, use_error=True):
        """Weight of each remembered problem given this problem's started chains [(step, best error)] and its failed
        fresh attempts [(model index, log error)]; also the failure counts per fresh model."""
        f = np.zeros(len(FRESH))
        for a, _ in ffail:
            f[a] += 1
        lw = (gammaln(self.B + f) - gammaln(self.B) - gammaln(self.A + self.B + f) + gammaln(self.A + self.B)).sum(1)
        for j, e in chains:
            lw = lw + np.log(self._dens(j, e, use_error))
        if use_error:
            for a, x in ffail:
                lw = lw + np.log(self._fdens(a, x))
        w = np.exp(lw - lw.max())
        return w / w.sum(), f

    def grade(self, j, e, w, horizon=STEPS, use_error=True, skip=None):
        """Grade of a chain after step j with best error e (j = 0: a new chain): the lowest expected dollars per
        success over the horizons 1..horizon, read off the futures of the remembered chains that looked like it."""
        T = min(horizon, STEPS - j)
        if T <= 0:
            return float("inf")
        f = self.fut[:, j].astype(float)
        if skip is not None:
            f[skip] = 0.0
        K = f * (np.exp(-0.5 * ((e - self.E[:, j]) / BW) ** 2) if (use_error and j) else 1.0)
        win = self.won[:, j + 1:j + 1 + T] * f[:, None]
        usd = (self.cum[:, j + 1:j + 1 + T] - self.cum[:, [j]]) * f[:, None]
        nf, den = max(f.sum(), 1.0), (self.M @ K + KAPPA)[:, None]
        P = w @ ((self.M @ (K[:, None] * win) + KAPPA * win.sum(0) / nf) / den)
        C = w @ ((self.M @ (K[:, None] * usd) + KAPPA * usd.sum(0) / nf) / den)
        return float(np.where(P > 1e-12, C / np.maximum(P, 1e-12), np.inf).min())

    def grade_fresh(self, a, w, f):
        p = float(w @ (self.A[:, a] / (self.A[:, a] + self.B[:, a] + f[a])))
        return self.price[FRESH[a]] / max(p, 1e-9)


# ------------------------------------------------------------------ one problem in one world
class Prob:
    """A problem's stored chains and attempts in this world's order, and what has been used so far."""

    def __init__(self, D, p, rng):
        self.p = p
        self.ids = [int(i) for i in rng.permutation(len(D.chains[p]))]
        self.ch = [D.chains[p][i] for i in self.ids]
        self.fid = {m: [int(i) for i in rng.permutation(len(D.fresh[p][m]))] for m in FRESH}
        self.fr = {m: [D.fresh[p][m][i] for i in self.fid[m]] for m in FRESH}
        self.pos, self.used = [0] * len(self.ch), {m: 0 for m in FRESH}
        self.ffail, self.done = [], False

    def started(self):                       # chains are started in order
        return sum(1 for x in self.pos if x)

    def state(self):
        return [(self.pos[i], self.ch[i][self.pos[i] - 1][2]) for i in range(self.started())]

    def can(self, opt):
        if opt[0] == "h":
            return opt[1] < len(self.ch) and self.pos[opt[1]] < len(self.ch[opt[1]])
        return self.used[opt[0]] < len(self.fr[opt[0]])

    def act(self, opt):
        if opt[0] == "h":
            c, ok, _ = self.ch[opt[1]][self.pos[opt[1]]]
            self.pos[opt[1]] += 1
        else:
            c, ok, le = self.fr[opt[0]][self.used[opt[0]]]
            self.used[opt[0]] += 1
            if not ok:
                self.ffail.append((FRESH.index(opt[0]), le))
        self.done = ok
        return c, ok


# ------------------------------------------------------------------ strategies: (problem state) -> (priority, option)
def strat_plan(K, R, s, fable=True):
    """R Haiku chains of up to K steps one after the other, then s Sonnet tries, then Fable. Priority 0: the
    portfolio takes problems breadth first (fewest tries first)."""
    def f(prob, rng):
        for c in range(R):
            if prob.pos[c] < K and prob.can(("h", c)):
                return 0.0, ("h", c)
        if prob.used["sonnet"] < s and prob.can(("sonnet",)):
            return 0.0, ("sonnet",)
        if fable and prob.can(("fable",)):
            return 0.0, ("fable",)
        return None
    return f


def strat_thompson(price):
    """Three arms per problem (a Haiku step on the current chain, Sonnet, Fable), Beta(1, 1) beliefs, the sampled
    chance divided by the price. Problems breadth first."""
    def f(prob, rng):
        cur = next((("h", i) for i in range(len(prob.ch)) if prob.can(("h", i))), None)
        arms = [(cur, sum(prob.pos), price["haiku"])] if cur else []
        arms += [((m,), sum(1 for a, _ in prob.ffail if a == j), price[m]) for j, m in enumerate(FRESH) if prob.can((m,))]
        if not arms:
            return None
        return 0.0, max(arms, key=lambda t: rng.beta(1, 1 + t[1]) / t[2])[0]
    return f


def strat_index(mem_of, horizon=STEPS, use_error=True, ref=False):
    """Take the option with the lowest grade; the grade is also the problem's priority in the portfolio."""
    cache = {}

    def f(prob, rng):
        ns = prob.started()
        nxt = prob.ids[ns] if ns < len(prob.ids) else None
        key = (prob.p, tuple(zip(prob.ids[:ns], prob.pos[:ns])), tuple(tuple(prob.fid[m][:prob.used[m]]) for m in FRESH),
               nxt if ref else None)
        g = cache.get(key)
        if g is None:
            mem, st = mem_of(prob.p), prob.state()
            w, fails = mem.weights(st, prob.ffail, use_error)
            g = [(mem.grade(j, e, w, horizon, use_error, prob.ids[i] if ref else None), ("h", i))
                 for i, (j, e) in enumerate(st) if prob.can(("h", i))]
            if prob.can(("h", ns)):
                g.append((mem.grade(0, 0.0, w, horizon, use_error, nxt if ref else None), ("h", ns)))
            g += [(mem.grade_fresh(a, w, fails), (m,)) for a, m in enumerate(FRESH) if prob.can((m,))]
            cache[key] = g
        return min(g, key=lambda t: t[0]) if g else None
    return f


# ------------------------------------------------------------------ replay
def sequences(D, problems, strat, W, seed):
    """seq[(problem, world)] = the strategy's steps on that problem to the end: (priority, dollars, solved, kind).
    A problem's steps do not depend on the other problems; the portfolio only interleaves them."""
    out = {}
    for p in problems:
        h = zlib.crc32(p.encode())
        for w in range(W):
            prob, rng, seq = Prob(D, p, np.random.default_rng([seed, w, h])), np.random.default_rng([seed, w, h, 7]), []
            while not prob.done and (nxt := strat(prob, rng)) is not None:
                c, ok = prob.act(nxt[1])
                seq.append((nxt[0], c, ok, nxt[1][0]))
            out[(p, w)] = seq
    return out


def merge(seqs, order, bmax=float("inf")):
    """Interleave the problems' steps: lowest priority first, then fewest tries, then this world's problem order.
    Returns the dollars spent when each problem was solved (within bmax) and the steps taken by kind."""
    heap = [(s[0][0], 0, int(order[i]), i) for i, s in enumerate(seqs) if s]
    heapq.heapify(heap)
    spend, solved, kinds = 0.0, [], Counter()
    while heap:
        _, t, o, i = heapq.heappop(heap)
        _, c, ok, kind = seqs[i][t]
        if spend + c > bmax + 1e-12:
            break
        spend += c
        kinds[kind] += 1
        if ok:
            solved.append(spend)
        elif t + 1 < len(seqs[i]):
            heapq.heappush(heap, (seqs[i][t + 1][0], t + 1, o, i))
    return solved, kinds


def area_of(solved, n, bmax):
    """Share of problems solved, averaged over the budgets bmax/200, 2 bmax/200, ..., bmax (as in SR1)."""
    return float(np.searchsorted(np.array(solved), np.linspace(0, bmax, 201)[1:] + 1e-12, side="right").mean()) / n


def portfolio_area(D, problems, seqs, W, seed):
    bmax, n = D.bmax(problems), len(problems)
    return float(np.mean([area_of(merge([seqs[(p, w)] for p in problems],
                                        np.random.default_rng([seed, w, 99]).permutation(n), bmax)[0], n, bmax)
                          for w in range(W)]))


GRID = [(0, 0, s) for s in (0, 1, 2, 4)] + [(K, R, s) for K in (1, 2, 3, 4, 6, 8, 12) for R in (1, 2, 3, 4)
                                             for s in (0, 1, 2, 4)]


def tune(D, problems, W=25, seed=5):
    """The plan (R chains of up to K steps, s Sonnet tries, then Fable) with the largest area on these problems."""
    best = None
    for plan in GRID:
        a = portfolio_area(D, problems, sequences(D, problems, strat_plan(*plan), W, seed), W, seed)
        if best is None or a > best[0] + 1e-12:
            best = (a, plan)
    return best[1]


def run_fold(args):
    D, test, memp, W, seed = args
    price = D.price(memp)
    mem, own = Memory(D, memp, price), {p: Memory(D, [p], price) for p in test}
    tuned, hind = tune(D, memp), tune(D, test)
    S = {"always Fable": strat_plan(0, 0, 0),
         "Haiku chains only (4 chains of up to 12 steps)": strat_plan(STEPS, 4, 0, fable=False),
         "plan: 3 fresh Haiku tries, then Fable": strat_plan(1, 3, 0),
         "plan: Haiku chain up to 4 steps, then Fable": strat_plan(4, 1, 0),
         "plan: Haiku chain up to 8 steps, then Fable": strat_plan(8, 1, 0),
         "plan: Haiku chain up to 12 steps, then Fable": strat_plan(12, 1, 0),
         "plan: (K, R, s) tuned on the memory problems": strat_plan(*tuned),
         HIND: strat_plan(*hind),
         "bandit: Thompson, cost aware": strat_thompson(price),
         "rule without look-ahead (horizon 1)": strat_index(lambda p: mem, horizon=1),
         "rule without the error signal": strat_index(lambda p: mem, use_error=False),
         RULE: strat_index(lambda p: mem),
         REF: strat_index(lambda p: own[p], ref=True)}
    return test, {k: sequences(D, test, s, W, seed) for k, s in S.items()}, tuned, hind


def evaluate(D, W, seed=0, procs=3):
    fo = fold_of(D.problems)
    jobs = [([p for p in D.problems if fo[p] == f], [p for p in D.problems if fo[p] != f]) for f in sorted(set(fo.values()))]
    with ProcessPoolExecutor(min(procs, len(jobs))) as ex:
        out = list(ex.map(run_fold, [(D, t, m, W, seed) for t, m in jobs]))
    for i, (test, _, tuned, hind) in enumerate(out):
        print("portfolio %d: %d problems, memory %d; plan tuned on memory (K steps, R chains, s Sonnet tries) = %s, "
              "best in hindsight = %s" % (i, len(test), len(D.problems) - len(test), tuned, hind))
    return [(test, seqs) for test, seqs, _, _ in out]


def report(D, res, W, seed=0):
    names, n = list(res[0][1]), len(D.problems)
    btot = sum(D.bmax(t) for t, _ in res)
    A, at, half, kinds = {}, {}, {}, Counter()
    for k in names:
        A[k] = float(np.mean([portfolio_area(D, t, s[k], W, seed) for t, s in res]))
        at[k], half[k] = np.zeros(len(FRACS)), 0.0
        for t, s in res:
            need, spends = math.ceil(0.5 * len(t)), []
            for w in range(W):
                sol, kd = merge([s[k][(p, w)] for p in t], np.random.default_rng([seed, w, 99]).permutation(len(t)))
                at[k] += np.searchsorted(np.array(sol), np.array(FRACS) * D.bmax(t) + 1e-12, side="right") / W
                spends.append(sol[need - 1] if len(sol) >= need else float("inf"))
                if k == RULE:
                    kd = merge([s[k][(p, w)] for p in t], np.random.default_rng([seed, w, 99]).permutation(len(t)), D.bmax(t))[1]
                    for x, v in kd.items():
                        kinds[x] += v / W
            half[k] += float(np.median(spends))
    print("\n%d problems, %d worlds; largest budget %.2f USD = one Fable attempt on every problem" % (n, W, btot))
    pr = D.price(D.problems)
    print("mean dollars: Haiku step %.4f, Sonnet attempt %.4f, Fable attempt %.4f\n" % (pr["haiku"], pr["sonnet"], pr["fable"]))
    print("Problems solved of %d (mean over worlds, summed over portfolios) at each budget" % n)
    print("| policy | " + " | ".join("%.2f USD" % (fr * btot) for fr in FRACS) + " | area | USD to solve half |\n|---|" + "---|" * (len(FRACS) + 2))
    for k in names:
        print("| %s | %s | %.3f | %s |" % (k, " | ".join("%.1f" % v for v in at[k]), A[k],
                                           "%.2f" % half[k] if math.isfinite(half[k]) else "never"))
    print("\nThe rule's steps inside the largest budget, per world: Haiku steps %.1f, Sonnet %.1f, Fable %.1f" % (
        kinds["h"], kinds["sonnet"], kinds["fable"]))
    print("rule minus best plan in hindsight: %+.3f" % (A[RULE] - A[HIND]))
    return A


_G = None


def _boot_one(b):
    D, res, W, Wb, seed = _G
    rng, d = np.random.default_rng([seed, b]), defaultdict(list)
    for test, seqs in res:
        draw = [str(x) for x in rng.choice(test, len(test), replace=True)]
        bmax, n = D.bmax(draw), len(draw)
        orders = [rng.permutation(n) for _ in range(Wb)]
        a = {k: float(np.mean([area_of(merge([s[(p, (w + 17 * i) % W)] for i, p in enumerate(draw)], orders[w], bmax)[0],
                                       n, bmax) for w in range(Wb)])) for k, s in seqs.items()}
        for k in a:
            d[k].append(a[RULE] - a[k])
    return {k: float(np.mean(v)) for k, v in d.items()}


def bootstrap(D, res, W, n_boot, Wb=25, seed=1, procs=4):
    """Uncertainty over problems: inside each fold draw its problems with replacement (memory and plans unchanged;
    a repeated problem gets another world's draws), area of the rule minus area of each policy."""
    global _G
    _G = (D, res, W, min(Wb, W), seed)
    with ProcessPoolExecutor(procs) as ex:
        rows = list(ex.map(_boot_one, range(n_boot), chunksize=5))
    print("\nArea of the rule minus area of each policy (%d resamples of the problems, 95%% interval)" % n_boot)
    for k in rows[0]:
        if k != RULE:
            d = np.sort([r[k] for r in rows])
            print("  %-50s %+.3f (%+.3f to %+.3f)" % (k, d.mean(), d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]))


# ------------------------------------------------------------------ what the chains look like
def describe(D, n_boot=2000, seed=3, at=4):
    lv = sorted({level(p) for p in D.problems}, key=lambda x: [1, 5, 2, 6, 3, 4, 0].index(x) if x in (1, 5, 2, 6, 3, 4, 0) else 9)
    ks = (1, 2, 4, 8, 12)
    print("\nShare of chains solved within k steps, and fresh attempts, by level (%d chains per problem)" % len(D.chains[D.problems[0]]))
    print("| level | chains | " + " | ".join("k = %d" % k for k in ks) + " | mean USD of a chain | Sonnet per attempt | Fable per attempt |\n|---|" + "---|" * (len(ks) + 4))
    for l in lv:
        P = [p for p in D.problems if level(p) == l]
        ch = [c for p in P for c in D.chains[p]]
        fr = lambda m: "%d of %d" % (sum(ok for p in P for _, ok, _ in D.fresh[p][m]), sum(len(D.fresh[p][m]) for p in P))
        print("| %d | %d | %s | %.3f | %s | %s |" % (l, len(ch), " | ".join(
            "%.2f" % np.mean([c[-1][1] and len(c) <= k for c in ch]) for k in ks), np.mean([sum(x[0] for x in c) for c in ch]),
            fr("sonnet"), fr("fable")))
    print("\nChance that step k solves, given the chain was unsolved before it (solved / chains that took the step)")
    print("| level | " + " | ".join("k = %d" % k for k in range(1, STEPS + 1)) + " |\n|---|" + "---|" * STEPS)
    for l in lv:
        ch = [c for p in D.problems if level(p) == l for c in D.chains[p]]
        print("| %d | %s |" % (l, " | ".join("%d/%d" % (sum(1 for c in ch if len(c) == k and c[-1][1]), sum(1 for c in ch if len(c) >= k))
                                               for k in range(1, STEPS + 1))))
    rows = [(p, c[at - 1][2], bool(c[-1][1])) for p in D.problems for c in D.chains[p] if len(c) > at]
    a = auc([x for _, x, _ in rows], [y for _, _, y in rows])
    rng, byp, bs = np.random.default_rng(seed), defaultdict(list), []
    for p, x, y in rows:
        byp[p].append((x, y))
    P = sorted(byp)
    for _ in range(n_boot):
        r = [xy for p in rng.choice(P, len(P), replace=True) for xy in byp[str(p)]]
        v = auc([x for x, _ in r], [y for _, y in r])
        if math.isfinite(v):
            bs.append(v)
    bs = np.sort(bs)
    print("\nD5: chains unsolved after %d steps: %d, of which solved by step %d: %d. Does the best error after step %d "
          "tell which? AUC %.2f (95%% interval %.2f to %.2f, resampling problems)" % (
              at, len(rows), STEPS, sum(y for _, _, y in rows), at, a, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]))
    for l in lv:
        r = [(x, y) for p, x, y in rows if level(p) == l]
        print("  level %d: %d chains, %d solved later, AUC %s" % (l, len(r), sum(y for _, y in r),
                                                                 "%.2f" % auc([x for x, _ in r], [y for _, y in r]) if r else "-"))


# ------------------------------------------------------------------ self-test on made-up data
def selftest():
    """Four hidden kinds of problem. On kind 1 chains get closer step by step and usually solve; on kind 2 chains
    stay stuck and only Fable helps; kind 3 nobody solves. The rule must beat Fable alone and chains alone and
    stay below the reference."""
    rng = np.random.default_rng(1)
    kinds = [dict(h=lambda j, e: 0.85, drift=0.0, e0=-8, sonnet=0.9, fable=0.95),
             dict(h=lambda j, e: 0.02 if j == 0 else min(0.6, 0.08 * (-e - 3)), drift=-1.0, e0=-3, sonnet=0.3, fable=0.9),
             dict(h=lambda j, e: 0.01, drift=0.0, e0=-3, sonnet=0.05, fable=0.7),
             dict(h=lambda j, e: 0.0, drift=0.0, e0=-1, sonnet=0.0, fable=0.03)]
    chains, fresh = {}, {}
    for i in range(48):
        t, p = kinds[i % 4], "k%02d_L%d_s%d" % (i, i % 4 + 1, i // 16 + 1)
        chains[p] = []
        for _ in range(4):
            ch, e = [], 2.0
            for j in range(STEPS):
                ok = rng.random() < t["h"](j, e if j else t["e0"])
                e = -22.0 if ok else min(e, t["e0"] + t["drift"] * j + rng.normal(0, 0.5))
                ch.append((0.01 * (1 + 0.2 * j) * rng.uniform(0.8, 1.2), ok, e))
                if ok:
                    break
            chains[p].append(ch)
        fresh[p] = {m: [((0.1 if m == "sonnet" else 0.5) * rng.uniform(0.8, 1.2), (ok := rng.random() < t[m]),
                         -22.0 if ok else t["e0"] + rng.normal(0, 1)) for _ in range(4 if m == "sonnet" else 1)] for m in FRESH}
    D = Data(chains, fresh)
    res = evaluate(D, 12)
    A = report(D, res, 12)
    assert A[RULE] >= max(A["always Fable"], A["Haiku chains only (4 chains of up to 12 steps)"]) - 1e-9, A
    assert A[RULE] <= A[REF] + 0.03, A
    bootstrap(D, res, 12, 8, Wb=6)
    describe(D, n_boot=200)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains")
    ap.add_argument("--fresh")
    ap.add_argument("--tau", type=float, default=1e-10)
    ap.add_argument("--worlds", type=int, default=100)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--scale", nargs="*", default=[], help="what-if prices, e.g. fable=5 haiku=10")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.chains is None:
        selftest()
    else:
        scale = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.scale}
        if scale:
            print("WHAT-IF PRICES (same stored sessions, prices multiplied): %s" % scale)
        D = Data(*load(a.chains, a.fresh, a.tau), scale)
        res = evaluate(D, a.worlds)
        report(D, res, a.worlds)
        if a.boot:
            bootstrap(D, res, a.worlds, a.boot)
        if not scale:
            describe(D)

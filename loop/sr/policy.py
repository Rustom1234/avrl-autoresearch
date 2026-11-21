#!/usr/bin/env python3
"""Replay routing policies on a bank of stored attempts (no model calls).

    policy.py --root loop/runs/SR1 --prior loop/runs/SR0 [--tau 1e-10] [--worlds 100] [--boot 200]
    policy.py --selftest

Setting: n problems, a few models. One attempt = one tool-using session of one model on one problem. It costs
dollars and either reaches the target or not; when it fails we still see how close it got (the training error of its
best candidate). A policy decides, step by step, which problem gets its next attempt from which model. It is judged
by how many problems are solved for the money spent.

The bank holds stored attempts per (problem, model). A world is one random order of each such stream; every policy
reads the same streams, so policies are compared on identical draws. A stream that is used up is closed.

THE RULE (index policy; known parts: Weitzman 1979, Streeter and Golovin 2008, Gergatsouli and Tzamos 2023):
  always make the attempt with the lowest expected dollars per success,  rho = cost(model) / P(success),
  where P(success) is the current belief for that problem and model.
Beliefs are joint and come from memory: the problem at hand is assumed to behave like one of the problems in
memory. Each remembered problem is weighted by how well it explains what has happened here so far: which models
failed, and how close the failed attempts got. A failure of the cheap model therefore moves the belief about the
strong model too, up or down.
Spending in order of rho is the greedy solution of the knapsack "most problems solved for B dollars"; the rho at
which the money runs out is the knapsack price (what one more solved problem costs at that budget).

Evaluation: problems are split into folds by generator seed. Each fold is one portfolio; its memory is every
problem outside the fold (other folds and the memory-only gate problems), the same memory for all its problems.
No problem is judged with knowledge of itself, and all problems of a portfolio are judged alike.
"""
import argparse, math, re, sys, zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BW = 1.0            # kernel width for "how close a failed attempt got", in decades of training error
KAPPA = 1.0         # weight of the pooled prior inside each remembered problem
FRACS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
RULE = "INDEX RULE (memory: failures and how close)"


def base(p):
    return p.split("#")[0]


def fold_of(problems):
    """problem -> fold. By generator seed when the ids carry at least 3 seeds, else 4 folds by a hash of the id."""
    seeds = {b: re.search(r"_s(\d+)$", b) for b in {base(p) for p in problems}}
    if all(seeds.values()) and len({m.group(1) for m in seeds.values()}) >= 3:
        return {p: int(seeds[base(p)].group(1)) for p in problems}
    return {p: zlib.crc32(base(p).encode()) % 4 for p in problems}


def logerr(x):
    return float(np.clip(math.log10(x), -30, 2)) if x and x > 0 and math.isfinite(x) else 0.0


# ------------------------------------------------------------------ bank and worlds
class Bank:
    """out[(problem, model)] = list of (cost, success, log10 training error of the attempt's best candidate).
    problems are scored; mem_only problems serve as memory and are never scored."""

    def __init__(self, recs, models, tau, prior_recs=(), scale=None):
        """scale: model -> factor on its price (a what-if on the same stored attempts)."""
        scale = scale or {}
        self.models = [m for m in models if any(r["model"] == m for r in recs)]
        self.problems = sorted({r["problem"] for r in recs})
        self.mem_only = sorted({r["problem"] for r in prior_recs} - set(self.problems))
        self.out = {(p, m): [] for p in self.problems + self.mem_only for m in self.models}
        for r in sorted(list(recs) + [r for r in prior_recs if r["problem"] in self.mem_only], key=lambda r: r["k"]):
            if r["model"] in self.models:
                ok = r.get("val") is not None and r["val"] <= tau
                self.out[(r["problem"], r["model"])].append(
                    (r["cost_usd"] * scale.get(r["model"], 1.0), ok, logerr((r.get("best") or {}).get("train_nmse"))))
        self._costs()

    def _costs(self):
        mean = lambda P, m: float(np.mean([c for p in P for c, _, _ in self.out[(base(p), m)]] or [np.nan]))
        self.cost_scored = {m: mean(self.problems, m) for m in self.models}      # for reporting only
        # what a policy may know beforehand: the mean price of an attempt on the memory problems
        self.cost = {m: (mean(self.mem_only, m) if any(self.out[(p, m)] for p in self.mem_only)
                         else self.cost_scored[m]) for m in self.models}

    def portfolio(self, scored, memory):
        """A bank that scores `scored` (repeats allowed: each repeat is a separate copy) with `memory` as memory."""
        b = object.__new__(Bank)
        b.models, b.out = self.models, self.out
        b.problems = ["%s#%d" % (base(p), i) for i, p in enumerate(scored)]
        b.mem_only = sorted({base(p) for p in memory})
        b._costs()
        return b

    def streams(self, p, m):
        return self.out[(base(p), m)]

    def world(self, rng):
        return {(p, m): [self.streams(p, m)[j] for j in rng.permutation(len(self.streams(p, m)))]
                for p in self.problems for m in self.models}


class Run:
    """State of one policy in one world."""

    def __init__(self, bank, world, order):
        self.b, self.w, self.order = bank, world, order     # order: this world's random order of the problems
        self.used = defaultdict(int)
        self.hist = {p: [] for p in bank.problems}          # per problem: (model index, log error) of failures
        self.solved, self.spend, self.curve = set(), 0.0, [(0.0, 0)]

    def open(self, p, m):
        return p not in self.solved and self.used[(p, m)] < len(self.w[(p, m)])

    def tries(self, p):
        return sum(self.used[(p, m)] for m in self.b.models)

    def fails(self, p, m):
        j = self.b.models.index(m)
        return sum(1 for a, _ in self.hist[p] if a == j)

    def attempt(self, p, m):
        c, ok, le = self.w[(p, m)][self.used[(p, m)]]
        self.used[(p, m)] += 1
        self.spend += c
        if ok:
            self.solved.add(p)
        else:
            self.hist[p].append((self.b.models.index(m), le))
        self.curve.append((self.spend, len(self.solved)))
        return ok


# ------------------------------------------------------------------ beliefs from the memory of other problems
class Memory:
    """What the remembered problems looked like: per problem and model, successes, failures and the log errors of
    the failed attempts."""

    def __init__(self, bank):
        P, M = bank.mem_only, bank.models
        src = lambda p, m: bank.out[(p, m)]
        self.S = np.array([[sum(ok for _, ok, _ in src(p, m)) for m in M] for p in P], float).reshape(len(P), len(M))
        self.F = np.array([[sum(not ok for _, ok, _ in src(p, m)) for m in M] for p in P], float).reshape(len(P), len(M))
        self.E = [[np.array([le for _, ok, le in src(p, m) if not ok]) for m in M] for p in P]
        self.pooled = (self.S.sum(0) + 0.5) / (self.S.sum(0) + self.F.sum(0) + 1.0)
        self.A, self.B = self.S + KAPPA * self.pooled, self.F + KAPPA * (1 - self.pooled)
        allE = [np.concatenate([e[j] for e in self.E] or [np.zeros(0)]) for j in range(len(M))]
        self.e_mu = [float(a.mean()) if len(a) else -3.0 for a in allE]
        self.e_sd = [float(max(a.std(), 1.5)) if len(a) else 3.0 for a in allE]

    def _dens(self, j, m, x):
        """Density of the log error x of a failed attempt of model m, on a problem like remembered problem j."""
        e = self.E[j][m]
        prior = math.exp(-0.5 * ((x - self.e_mu[m]) / self.e_sd[m]) ** 2) / (self.e_sd[m] * 2.5066)
        k = float(np.exp(-0.5 * ((x - e) / BW) ** 2).sum()) / (BW * 2.5066) if len(e) else 0.0
        return (k + KAPPA * prior) / (len(e) + KAPPA)

    def predict(self, hist, use_error=True):
        """Chance that the next attempt of each model succeeds, given the failures so far on this problem."""
        n = len(self.S)
        if n == 0:
            return self.pooled
        f = np.zeros(self.S.shape[1])
        for a, _ in hist:
            f[a] += 1
        lw = (gammaln(self.B + f) - gammaln(self.B) - gammaln(self.A + self.B + f) + gammaln(self.A + self.B)).sum(1)
        if use_error:
            for a, x in hist:
                lw += np.log([self._dens(j, a, x) for j in range(n)])
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return w @ (self.A / (self.A + self.B + f))

    def separate(self, hist):
        f = np.zeros(len(self.pooled))
        for a, _ in hist:
            f[a] += 1
        a, b = self.pooled * 2.0, (1 - self.pooled) * 2.0   # Beta prior with the pooled mean and weight 2
        return a / (a + b + f)


# ------------------------------------------------------------------ policies: each returns the next (problem, model)
def pol_fixed(m):
    def f(run, mem, rng):
        opts = [(run.used[(p, m)], i, p) for i, p in enumerate(run.order) if run.open(p, m)]
        return (min(opts)[2], m) if opts else None         # breadth first: the problem with the fewest tries so far
    return f


def pol_cascade(plan):
    """plan: list of (model, tries). Breadth first over problems; each problem walks its own ladder."""
    ladder = [m for m, n in plan for _ in range(n)]

    def f(run, mem, rng):
        best = None
        for p in run.order:
            if p in run.solved:
                continue
            pos = mem.setdefault(("pos", p), [0])
            while pos[0] < len(ladder) and not run.open(p, ladder[pos[0]]):
                pos[0] += 1                                 # stream used up: move down the ladder
            if pos[0] < len(ladder) and (best is None or run.tries(p) < best[0]):
                best = (run.tries(p), p, ladder[pos[0]], pos)
        if best:
            best[3][0] += 1
            return best[1], best[2]
        return None
    return f


def pol_ucb_cost(run, mem, rng):
    """One bandit over models for the whole portfolio, reward = success, score = upper bound divided by cost
    (the cost-blend idea of ShinkaEvolve). Problems are taken breadth first."""
    n, s = mem.setdefault("n", defaultdict(int)), mem.setdefault("s", defaultdict(int))
    tot = sum(n.values()) + 1
    score = {m: ((s[m] / n[m] + math.sqrt(2 * math.log(tot) / n[m])) if n[m] else float("inf")) / run.b.cost[m]
             for m in run.b.models}
    for m in sorted(run.b.models, key=lambda m: -score[m]):
        opts = [(run.tries(p), i, p) for i, p in enumerate(run.order) if run.open(p, m)]
        if opts:
            return min(opts)[2], m
    return None


def pol_thompson(cost_aware):
    """Independent Beta(1,1) belief per (problem, model), Thompson sampling (the arm choice of AB-MCTS, which has no
    cost; the cost-aware form divides the sampled chance by the cost)."""
    def f(run, mem, rng):
        best = None
        for p in run.order:
            for m in run.b.models:
                if run.open(p, m):
                    v = rng.beta(1, 1 + run.fails(p, m)) / (run.b.cost[m] if cost_aware else 1.0)
                    if best is None or v > best[0]:
                        best = (v, p, m)
        return (best[1], best[2]) if best else None
    return f


def pol_index(kind):
    """kind: 'joint' = THE RULE (memory, failures and how close); 'counts' = memory, failures only;
    'separate' = no learning across models; 'clairvoyant' = knows each problem's true success share in the bank
    (not attainable, a reference)."""
    def f(run, mem, rng):
        cache = mem.setdefault(("pr", kind), {})
        best = None
        for i, p in enumerate(run.order):
            if p in run.solved:
                continue
            key = len(run.hist[p])
            if cache.get(p, (None,))[0] != key:
                if kind == "clairvoyant":
                    pr = np.array([np.mean([ok for _, ok, _ in run.b.streams(p, m)] or [0.0]) for m in run.b.models])
                elif kind == "separate":
                    pr = mem["memory"].separate(run.hist[p])
                else:
                    pr = mem["memory"].predict(run.hist[p], use_error=(kind == "joint"))
                cache[p] = (key, pr)
            pr = cache[p][1]
            for j, m in enumerate(run.b.models):
                if run.open(p, m):
                    k = (run.b.cost[m] / max(pr[j], 1e-9), run.tries(p), i)   # ties: fewest tries, then order
                    if best is None or k < best[0]:
                        best = (k, p, m)
        if best and kind == "joint":
            mem.setdefault("rho", []).append((run.spend, best[0][0], best[2]))
        return (best[1], best[2]) if best else None
    return f


GRID = [(h, s) for h in (0, 1, 2, 3, 4, 6, 8, 12, 99) for s in (0, 1, 2, 3, 99)]


def schedule(models, h, s):
    return ([(models[0], h)] if h else []) + ([(models[1], s)] if s and len(models) > 2 else []) + [(models[-1], 99)]


def play(bank, pol, n_worlds, seed, name="", prep=None):
    """Curves [(spend, solved)] of one policy over the worlds, and its memory dicts."""
    curves, mems = [], []
    for w in range(n_worlds):
        wr = np.random.default_rng([seed, w])
        world = bank.world(wr)
        run = Run(bank, world, [bank.problems[j] for j in wr.permutation(len(bank.problems))])
        mem, rng = dict(prep or {}), np.random.default_rng([seed, w, 7])
        while (nxt := pol(run, mem, rng)) is not None:
            ok = run.attempt(*nxt)
            if name.startswith("bandit"):
                mem["n"][nxt[1]] += 1
                mem["s"][nxt[1]] += ok
        curves.append(run.curve)
        mems.append(mem)
    return curves, mems


def solved_at(curve, B):
    s = 0
    for spend, k in curve:
        if spend > B + 1e-12:
            break
        s = k
    return s


def spend_to(curve, k):
    for spend, s in curve:
        if s >= k:
            return spend
    return float("inf")


def area(cs, n, bmax):
    grid = np.linspace(0, bmax, 201)[1:]
    return float(np.mean([[solved_at(c, g) for g in grid] for c in cs])) / n


def bmax_of(bank):
    return bank.cost_scored[bank.models[-1]] * len(bank.problems)


def tune_schedule(bank, n_worlds=12, seed=5):
    """The fixed schedule (h cheap tries, s middle tries, then the strongest) with the largest area on this bank."""
    best = None
    for h, s in GRID:
        cs, _ = play(bank, pol_cascade(schedule(bank.models, h, s)), n_worlds, seed)
        a = area(cs, len(bank.problems), bmax_of(bank))
        if best is None or a > best[0] + 1e-12:
            best = (a, h, s)
    return best[1], best[2]


CHOSEN = {}     # the schedules picked by the last call of policies(), for the report


def policies(bank, tune):
    M = bank.models
    P = {"always " + m: pol_fixed(m) for m in M}
    if len(M) >= 2:
        P["cascade: 1 try each, cheap to strong"] = pol_cascade([(m, 1) for m in M] + [(M[-1], 99)])
        P["cascade: 3 cheap tries, then strongest"] = pol_cascade([(M[0], 3), (M[-1], 99)])
        P["cascade: every model used up in turn"] = pol_cascade([(m, 99) for m in M])
    if tune:
        mem_bank = bank.portfolio(bank.mem_only, [])        # the memory problems, played as a portfolio
        CHOSEN["memory"], CHOSEN["hindsight"] = tune_schedule(mem_bank), tune_schedule(bank)
        P["cascade: schedule tuned on the memory problems"] = pol_cascade(schedule(M, *CHOSEN["memory"]))
        P["cascade: best schedule in hindsight"] = pol_cascade(schedule(M, *CHOSEN["hindsight"]))
    P["bandit over models, cost blend"] = pol_ucb_cost
    P["Thompson, no cost"] = pol_thompson(False)
    P["Thompson, cost aware"] = pol_thompson(True)
    P["index, separate beliefs"] = pol_index("separate")
    P["index, memory of failures only"] = pol_index("counts")
    P[RULE] = pol_index("joint")
    P["clairvoyant (reference)"] = pol_index("clairvoyant")
    return P


def simulate(bank, n_worlds, seed=0, tune=True):
    """One portfolio with one memory. Returns name -> curves, and the rule's (spend, rho, model) traces."""
    prep = {"memory": Memory(bank)}
    out, rho = {}, []
    for name, pol in policies(bank, tune).items():
        out[name], mems = play(bank, pol, n_worlds, seed, name, prep)
        if name == RULE:
            rho = [m.get("rho", []) for m in mems]
    return out, rho


def portfolios(bank):
    """One portfolio per fold: the fold's problems scored, everything else as memory."""
    fo = fold_of(bank.problems)
    for f in sorted(set(fo.values())):
        inside = [p for p in bank.problems if fo[p] == f]
        yield f, bank.portfolio(inside, [p for p in bank.problems if fo[p] != f] + list(bank.mem_only))


def evaluate(bank, n_worlds, tune=True, seed=0):
    """Returns per-policy: area (mean over folds), solved at each budget share (summed over folds), and per-fold
    curves with their largest budgets."""
    res = {}
    for f, pb in portfolios(bank):
        curves, rho = simulate(pb, n_worlds, seed, tune)
        res[f] = (pb, curves, rho)
        if tune:
            print("portfolio %s: schedule tuned on memory = %s cheap tries, %s middle tries, then the strongest; "
                  "best in hindsight = %s cheap, %s middle (99 = until used up)" % (
                      f, *CHOSEN["memory"], *CHOSEN["hindsight"]))
    names = list(next(iter(res.values()))[1])
    A = {k: float(np.mean([area(res[f][1][k], len(res[f][0].problems), bmax_of(res[f][0])) for f in res])) for k in names}
    at = {k: [sum(np.mean([solved_at(c, fr * bmax_of(res[f][0])) for c in res[f][1][k]]) for f in res) for fr in FRACS]
          for k in names}
    return A, at, res


def report(bank, n_worlds, tune=True):
    A, at, res = evaluate(bank, n_worlds, tune)
    n = len(bank.problems)
    btot = sum(bmax_of(pb) for pb, _, _ in res.values())
    print("%d problems in %d portfolios (folds), %d worlds; memory per portfolio: %s problems" % (
        n, len(res), n_worlds, ", ".join(str(len(pb.mem_only)) for pb, _, _ in res.values())))
    print("stored attempts per problem (fewest): " + ", ".join(
        "%s %d" % (m, min(len(bank.out[(p, m)]) for p in bank.problems)) for m in bank.models))
    print("mean cost of one attempt: " + ", ".join("%s %.4f USD" % (m, bank.cost_scored[m]) for m in bank.models))
    print("largest budget %.2f USD in total = one %s attempt on every problem\n" % (btot, bank.models[-1]))
    print("Problems solved of %d (mean over worlds, summed over portfolios) at each budget" % n)
    print("| policy | " + " | ".join("%.2f USD" % (fr * btot) for fr in FRACS) + " | area |\n|---|" + "---|" * (len(FRACS) + 1))
    for k in A:
        print("| %s | %s | %.3f |" % (k, " | ".join("%.1f" % v for v in at[k]), A[k]))
    print("\narea = share of problems solved, averaged over all budgets from 0 to the largest (higher is better)")
    for share in (0.25, 0.5):
        print("\nDollars to solve %d%% of the problems of a portfolio (median over worlds, summed over portfolios; "
              "'never' if not reached in half the worlds of some portfolio)" % (100 * share))
        for k in A:
            v = sum(float(np.median([spend_to(c, max(1, math.ceil(share * len(pb.problems)))) for c in cs[k]]))
                    for pb, cs, _ in res.values())
            print("  %-46s %s" % (k, "%.2f" % v if math.isfinite(v) else "never"))
    use = defaultdict(float)
    for pb, _, rho in res.values():
        for tr in rho:
            for _, _, m in tr:
                use[m] += 1.0 / len(rho)
    print("\nThe rule's attempts per world, by model: " + ", ".join("%s %.1f" % (m, use[m]) for m in bank.models))
    return A


def _boot_one(args):
    bank, b, n_worlds, tune, seed = args
    rng = np.random.default_rng([seed, b])
    fo = fold_of(bank.problems)
    diffs = defaultdict(list)
    for f in sorted(set(fo.values())):
        inside = [p for p in bank.problems if fo[p] == f]
        pb = bank.portfolio(list(rng.choice(inside, len(inside), replace=True)),
                            [p for p in bank.problems if fo[p] != f] + list(bank.mem_only))
        curves, _ = simulate(pb, n_worlds, seed=1000 + b, tune=tune)
        a = {k: area(cs, len(pb.problems), bmax_of(pb)) for k, cs in curves.items()}
        for k in a:
            diffs[k].append(a[RULE] - a[k])
    return {k: float(np.mean(v)) for k, v in diffs.items()}


def bootstrap(bank, n_boot, n_worlds, tune=True, seed=1, procs=4):
    """Uncertainty over problems: inside each fold draw its problems with replacement (memory unchanged), replay,
    area of the rule minus area of each policy. Runs on `procs` processes (one thread each)."""
    from concurrent.futures import ProcessPoolExecutor
    D = defaultdict(list)
    with ProcessPoolExecutor(procs) as ex:
        for d in ex.map(_boot_one, [(bank, b, n_worlds, tune, seed) for b in range(n_boot)]):
            for k, v in d.items():
                D[k].append(v)
    print("\nArea of the rule minus area of each policy (%d resamples of the problems, 95%% interval)" % n_boot)
    for k, d in D.items():
        if k != RULE:
            d = np.sort(d)
            print("  %-46s %+.3f (%+.3f to %+.3f)" % (k, d.mean(), d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]))


def auc(x, y):
    """Chance that a solved case has a lower x than an unsolved one (ties count half)."""
    pos, neg = [v for v, l in zip(x, y) if l], [v for v, l in zip(x, y) if not l]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(a < b) + 0.5 * (a == b) for a in pos for b in neg]))


def signal(bank, n_boot=2000, seed=3):
    """Claim C4: does 'how close the cheap model got' tell whether a stronger model solves the problem?"""
    M = bank.models
    rows = []
    for p in bank.problems:
        cheap = bank.out[(p, M[0])]
        if cheap and not any(ok for _, ok, _ in cheap):
            rows.append((min(le for _, _, le in cheap), any(ok for m in M[1:] for _, ok, _ in bank.out[(p, m)])))
    print("\nSignal check: problems where every %s attempt failed: %d, of which a stronger model solved %d" % (
        M[0], len(rows), sum(l for _, l in rows)))
    if len(rows) < 6 or not 0 < sum(l for _, l in rows) < len(rows):
        print("  too few cases of one kind for an AUC")
        return
    x, y = [r[0] for r in rows], [r[1] for r in rows]
    rng, B = np.random.default_rng(seed), []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        v = auc([x[i] for i in idx], [y[i] for i in idx])
        if v == v:
            B.append(v)
    B.sort()
    print("  best %s training error (log10) as predictor of 'a stronger model solves it': AUC %.2f (%.2f to %.2f)" % (
        M[0], auc(x, y), B[int(0.025 * len(B))], B[int(0.975 * len(B)) - 1]))
    print("  median log10 error where solved later %.1f, where not %.1f" % (
        np.median([a for a, l in rows if l]), np.median([a for a, l in rows if not l])))


# ------------------------------------------------------------------ self test on a made-up bank
def selftest(K=(12, 4, 1)):
    """Made-up problems of four hidden kinds. A failed cheap attempt gets closer on the kinds a strong model can
    solve, so 'how close' carries information. The rule must beat every fixed model and stay under the reference."""
    rng = np.random.default_rng(1)
    cost = {"haiku": 0.01, "sonnet": 0.1, "fable": 1.0}
    kinds = [dict(haiku=0.8, sonnet=0.9, fable=0.95, err=-8), dict(haiku=0.1, sonnet=0.5, fable=0.8, err=-6),
             dict(haiku=0.0, sonnet=0.05, fable=0.6, err=-5), dict(haiku=0.0, sonnet=0.0, fable=0.03, err=-2)]
    recs = []
    for i in range(64):
        t = kinds[i % 4]
        for m, k in zip(cost, K):
            for j in range(k):
                ok = rng.random() < t[m]
                recs.append({"problem": "p%02d_s%d" % (i, i // 16), "model": m, "k": j,
                             "cost_usd": cost[m] * rng.uniform(0.7, 1.3), "val": 0.0 if ok else 1.0,
                             "best": {"train_nmse": 1e-20 if ok else 10 ** (t["err"] + rng.normal(0, 1.0))}})
    prior = [r for r in recs if r["problem"].endswith("_s0")]
    bank = Bank([r for r in recs if not r["problem"].endswith("_s0")], list(cost), 1e-10, prior)
    A = report(bank, 20)
    assert A[RULE] >= max(v for k, v in A.items() if k.startswith("always")) - 1e-9
    assert A[RULE] <= A["clairvoyant (reference)"] + 0.02
    signal(bank)
    bootstrap(bank, 10, 4)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--prior", help="bank root whose problems serve as memory only (the gate run)")
    ap.add_argument("--tau", type=float, default=1e-10)
    ap.add_argument("--worlds", type=int, default=100)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet", "fable"])
    ap.add_argument("--only", nargs="*", help="keep problems whose id contains one of these strings")
    ap.add_argument("--scale", nargs="*", default=[], help="what-if prices, e.g. fable=0.1 sonnet=2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or a.root is None:
        selftest()
    else:
        import bank as B
        keep = lambda r: not a.only or any(s in r["problem"] for s in a.only)
        recs = [r for r in B.load(a.root) if keep(r)]
        scale = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.scale}
        if scale:
            print("WHAT-IF PRICES (same stored attempts, prices multiplied): %s\n" % scale)
        bank = Bank(recs, a.models, a.tau, [r for r in B.load(a.prior) if keep(r)] if a.prior else (), scale)
        report(bank, a.worlds)
        signal(bank)
        if a.boot:
            bootstrap(bank, a.boot, max(5, a.worlds // 10))

"""Pluggable verification allocators for phase-1 GRPO (see p1_grpo.py, PROJECT.md).

Each arm decides, per training step, which of that step's rollout groups get the
expensive (gold) reward versus the cheap (majority-vote) proxy reward, under a
shared per-step budget.

Fairness invariants enforced in this file (see PROJECT.md / task spec):
  - `n_verify()` is the one function every arm uses to turn (n_groups, budget) into
    an expensive-call count. It is a pure function of its arguments: called once per
    step with that step's true group count, so budget is a hard cap with no
    carry-over across steps, and 'random'/'threshold'/'learned' at the same --budget
    always spend the same number of expensive calls per step.
  - `select()` only ever sees the cheap, gold-free `feat_vec`/`cell`/`agreement`/
    `n_distinct` fields built in p1_grpo.py's RewardState; no allocator class here
    imports or receives gold.
  - `LearnedAllocator.update()` is the only place a gold-derived label
    (proxy_error) enters this module, and the caller (RewardState in p1_grpo.py)
    only ever passes it the groups verified *this step* -- so its training set is
    always a subset of groups it itself chose to verify.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def n_verify(n_groups, budget, step=None):
    """Shared, pure budget->count function. Same (n_groups, budget, step) always gives the
    same integer, independent of arm, history, or RNG state. With step given, fractional
    budgets are realized by cumulative rounding (for 2 groups at 25 percent: 0, 1, 0, 1, ...),
    so the fraction of groups verified over a run equals the budget; without it, plain rounding
    (unchanged behaviour for every earlier run, where budget x n_groups was an integer)."""
    q = budget * n_groups
    if step is None:
        return int(round(q))
    import math
    return int(math.floor(q * (step + 1)) - math.floor(q * step))


class BaseAllocator:
    name = "base"

    def n_verify(self, n_groups, budget, step=None):
        return n_verify(n_groups, budget, step)

    def select(self, group_records, k, rng):
        """Return exactly k group indices (the `idx` field of group_records) to verify."""
        raise NotImplementedError

    def update(self, verified):
        """verified: list of (feat_vec, proxy_error_label, cell) for groups just
        verified this step. Default no-op; only LearnedAllocator uses it."""
        pass


class FullAllocator(BaseAllocator):
    """Upper bound: verify every group. B=1 by definition, ignores --budget."""
    name = "full"

    def n_verify(self, n_groups, budget, step=None):
        return n_groups

    def select(self, group_records, k, rng):
        return [r["idx"] for r in group_records]


class ProxyAllocator(BaseAllocator):
    """Lower bound: never verify. B=0 by definition, ignores --budget."""
    name = "proxy"

    def n_verify(self, n_groups, budget, step=None):
        return 0

    def select(self, group_records, k, rng):
        return []


class RandomAllocator(BaseAllocator):
    """Uniform random B of groups per step."""
    name = "random"

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        idx = np.array([r["idx"] for r in group_records])
        return rng.choice(idx, size=k, replace=False).tolist()


class ThresholdAllocator(BaseAllocator):
    """Verify the k lowest-agreement groups; ties broken toward more distinct
    answers (a tied agreement fraction with more distinct answers is the less
    trustworthy majority)."""
    name = "threshold"

    @staticmethod
    def _order(group_records):
        return sorted(group_records, key=lambda r: (r["agreement"], -r["n_distinct"]))

    def select(self, group_records, k, rng):
        return [r["idx"] for r in self._order(group_records)[:k]]


class LearnedAllocator(BaseAllocator):
    """Online logistic regression on cheap features + PCA components, predicting
    proxy_error (majority != gold), refit every `refit_every` verified groups on all
    verified groups seen so far (cumulative, across steps). Combined with Thompson
    sampling over embedding cells, using the regression's predicted error rate as
    an informative Beta prior (converted to pseudo-counts via `prior_strength`) the
    first time a cell is scored. Before the first fit, falls back to `threshold`.

    Design choices (spec leaves these open, stated explicitly per Karpathy rule 1):
      - `feat_vec` fed to the regression is [agreement, n_distinct, mean_len,
        std_len, frac_truncated, q_len] + 16 PCA dims (21 features). The embedding
        cell id is deliberately NOT included as a numeric regression feature (it is
        a nominal cluster id, not an ordinal quantity, and the 16 PCA components it
        was derived from already carry the continuous embedding signal); cell id is
        used only as the Thompson-sampling grouping key.
      - Each candidate group draws its own independent Beta sample from its cell's
        posterior (not one shared sample per cell per step); group order is by that
        sampled score, descending.
    """
    name = "learned"

    def __init__(self, refit_every=20, prior_strength=2.0):
        self.refit_every = refit_every
        self.prior_strength = prior_strength
        self.X = []          # cumulative verified feature vectors (fairness: verified-only)
        self.y = []          # cumulative verified proxy_error labels (fairness: verified-only)
        self.model = None
        self.last_fit_n = 0
        self.cell_beta = {}  # cell_id -> [alpha, beta]

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in ThresholdAllocator._order(group_records)[:k]]
        idx = [r["idx"] for r in group_records]
        X = np.array([r["feat_vec"] for r in group_records])
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
        scores = np.empty(len(group_records))
        for i, r in enumerate(group_records):
            cell = r["cell"]
            if cell not in self.cell_beta:
                self.cell_beta[cell] = [self.prior_strength * p[i], self.prior_strength * (1 - p[i])]
            a, b = self.cell_beta[cell]
            scores[i] = rng.beta(a, b)
        order = np.argsort(-scores, kind="stable")
        return [idx[i] for i in order[:k]]

    def update(self, verified):
        for feat_vec, label, cell in verified:
            self.X.append(feat_vec)
            self.y.append(label)
            a, b = self.cell_beta.get(cell, [self.prior_strength, self.prior_strength])
            self.cell_beta[cell] = [a + label, b + (1 - label)]
        n = len(self.y)
        due = n >= self.refit_every and n // self.refit_every > self.last_fit_n // self.refit_every
        if due and len(set(self.y)) > 1:
            self.model = LogisticRegression(max_iter=1000).fit(np.array(self.X), np.array(self.y))
            self.last_fit_n = n



class GreedyOnlineAllocator(LearnedAllocator):
    """The replay winner (greedy_online in runs/sim_generic.py), in training: the same online
    logistic regression as LearnedAllocator, refit on verified groups only, but groups are ranked
    directly by predicted corrupted-reward probability. No Thompson sampling over cells, no layout.
    Cold start (before the first fit): the threshold rule. Budget accounting is unchanged."""
    name = "greedy"

    def select(self, group_records, k, rng):
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in ThresholdAllocator._order(group_records)[:k]]
        idx = [r["idx"] for r in group_records]
        X = np.array([r["feat_vec"] for r in group_records])
        p = self.model.predict_proba(X)[:, 1]
        order = np.argsort(-p, kind="stable")
        return [idx[i] for i in order[:k]]


ALLOCATORS = {
    "full": FullAllocator,
    "proxy": ProxyAllocator,
    "random": RandomAllocator,
    "threshold": ThresholdAllocator,
    "learned": LearnedAllocator,
    "greedy": GreedyOnlineAllocator,
}

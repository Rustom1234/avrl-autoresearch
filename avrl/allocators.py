"""Verification allocators for Part 1 GRPO training (used by p1_grpo.py, p1_judge_grpo.py
and p1_code_grpo.py; overview in avrl/README.md).

Setting. Each training step samples G completions for every prompt in the step; the G
completions of one prompt form a "group" (GRPO computes advantages within a group). A
group can be rewarded in two ways:
  - cheap proxy reward: free but noisy (on GSM8K, agreement with the group's majority
    answer);
  - expensive reward ("gold"): the trusted check (the reference answer, an LLM judge, or
    the hidden unit tests, depending on the task).
The verification budget is the fraction of a step's groups that may receive the
expensive reward. An allocator decides WHICH groups those are. One allocator is one
experimental arm; everything else about a run is held fixed across arms.

Arms (the ALLOCATORS registry at the bottom of this file):
  full       verify every group (upper bound, ignores the budget)
  proxy      verify nothing (lower bound, ignores the budget)
  random     a uniformly random subset of the step's groups
  threshold  the groups where the cheap signal is least confident
  learned    online logistic regression plus Thompson sampling over prompt-embedding cells
  greedy     the same regression, ranked directly by predicted proxy-error probability

Contract with the training script, called once per step in this order:
  k   = allocator.n_verify(n_groups, budget)       how many groups to verify
  idx = allocator.select(group_records, k, rng)    which ones (exactly k group indices)
  allocator.update(verified)                       gold-derived labels for those k groups

A group record is a dict built by the training script (RewardState in p1_grpo.py and its
counterparts) with these keys:
  idx         position of the group within the step
  agreement   cheap confidence, lower = less trustworthy (GSM8K: fraction of the G answers
              equal to the majority answer; the judge task passes its own confidence
              scalar under the same key)
  n_distinct  number of distinct answers in the group
  cell        KMeans cluster id of the prompt embedding (a coarse "kind of prompt" label)
  feat_vec    cheap numeric features followed by the prompt's PCA components

Fairness invariants enforced in this file:
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
    only ever passes it the groups verified *this step*, so its training set is
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
    q = budget * n_groups  # expected verified groups per step; may be fractional
    if step is None:
        return int(round(q))
    import math
    # floor(q*(step+1)) - floor(q*step): the steps' counts telescope, so after any number of
    # steps the total verified is floor(q * steps), never above the budget.
    return int(math.floor(q * (step + 1)) - math.floor(q * step))


class BaseAllocator:
    """Interface every arm implements. Subclasses override select() (and, for the two
    bound arms, n_verify()); `name` is the --arm value and the arm label in the logs."""
    name = "base"

    def n_verify(self, n_groups, budget, step=None):
        """Number of groups this arm verifies this step; defaults to the shared module-level
        n_verify() so all budgeted arms spend identically."""
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
    trustworthy majority). No learning and no labels: this is the simple confidence
    rule that the learned arms are compared against."""
    name = "threshold"

    @staticmethod
    def _order(group_records):
        """Group records sorted most-suspect first. Static so the learned arms can reuse
        it as their cold-start rule without instantiating a ThresholdAllocator."""
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

    Design choices:
      - `feat_vec` fed to the regression is [agreement, n_distinct, mean_len,
        std_len, frac_truncated, q_len] + 16 PCA dims (22 features). The embedding
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
        self.model = None    # None until the first refit; select() uses the threshold rule until then
        self.last_fit_n = 0  # len(self.y) at the most recent refit
        self.cell_beta = {}  # cell_id -> [alpha, beta]: Beta posterior over that cell's proxy-error rate

    def select(self, group_records, k, rng):
        """Rank groups by a Thompson sample of their cell's proxy-error rate, return the top k."""
        if k == 0:
            return []
        if self.model is None:
            return [r["idx"] for r in ThresholdAllocator._order(group_records)[:k]]
        idx = [r["idx"] for r in group_records]
        X = np.array([r["feat_vec"] for r in group_records])  # (n_groups, n_features)
        # Predicted P(proxy_error) per group, clipped away from 0 and 1 so the Beta prior
        # built from it below always has strictly positive parameters.
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
        scores = np.empty(len(group_records))
        for i, r in enumerate(group_records):
            cell = r["cell"]
            if cell not in self.cell_beta:
                # First visit to this cell: seed its posterior with `prior_strength`
                # pseudo-observations split according to the regression's prediction.
                self.cell_beta[cell] = [self.prior_strength * p[i], self.prior_strength * (1 - p[i])]
            a, b = self.cell_beta[cell]
            scores[i] = rng.beta(a, b)
        # Highest sampled error rate first; stable sort keeps ties in group order.
        order = np.argsort(-scores, kind="stable")
        return [idx[i] for i in order[:k]]

    def update(self, verified):
        """Add this step's verified groups to the training set, update each cell's Beta
        counts (label 1 = the proxy was wrong), and refit the regression when due."""
        for feat_vec, label, cell in verified:
            self.X.append(feat_vec)
            self.y.append(label)
            a, b = self.cell_beta.get(cell, [self.prior_strength, self.prior_strength])
            self.cell_beta[cell] = [a + label, b + (1 - label)]
        n = len(self.y)
        # Refit each time the verified count crosses a multiple of refit_every, and only
        # once both classes are present (logistic regression cannot fit a single class).
        due = n >= self.refit_every and n // self.refit_every > self.last_fit_n // self.refit_every
        if due and len(set(self.y)) > 1:
            self.model = LogisticRegression(max_iter=1000).fit(np.array(self.X), np.array(self.y))
            self.last_fit_n = n



class GreedyOnlineAllocator(LearnedAllocator):
    """The replay winner (greedy_online in the offline replay simulator, sim_generic.py), in
    training: the same online logistic regression as LearnedAllocator, refit on verified groups
    only, but groups are ranked directly by predicted corrupted-reward probability. No Thompson
    sampling over cells, no layout. Cold start (before the first fit): the threshold rule.
    Budget accounting is unchanged."""
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


# --arm value -> allocator class. The training scripts instantiate one per run.
ALLOCATORS = {
    "full": FullAllocator,
    "proxy": ProxyAllocator,
    "random": RandomAllocator,
    "threshold": ThresholdAllocator,
    "learned": LearnedAllocator,
    "greedy": GreedyOnlineAllocator,
}

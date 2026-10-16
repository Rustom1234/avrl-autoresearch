# avrl_router: regime diagnostic (v0)

A standalone diagnostic. No training, no model call. It takes a sample of
rollout groups where you already know a cheap grader's confidence and
whether that cheap grader disagreed with an expensive grader, and tells you
how much verification budget buys you.

This is the first product piece described in the AVRL method doc: the
regime diagnostic that decides whether a confidence signal is worth
threshold-based verification routing.

## Input

A parquet file, one row per rollout group, with:

- One or more confidence columns. Higher value means the cheap grader
  (majority vote, reward model margin, self-consistency, etc.) was more
  confident in that group.
- An error column, 1 if the cheap grader's answer disagreed with the
  expensive grader (gold answer, executed tests, LLM judge). That group
  would get a corrupted reward if it is not sent for expensive verification.

## Run it

```bash
cd /homes/rdubash/active-audit/avrl
/homes/rdubash/active-audit/.venv/bin/python -m avrl_router.diagnose \
  --groups runs/E08/feats.parquet \
  --confidence agreement \
  --error proxy_error \
  --budgets 0.1,0.25,0.5 \
  --target_corruption 0.02 \
  --out /tmp/diag_e08
```

Multiple confidence columns (e.g. for open-ended tasks with both a reward
model gap and a self-consistency score): `--confidence rm_gap12,consistency`.

`--target_corruption` takes either an absolute rate (default `0.02`, as
above) or `rel:F`, meaning F times the measured proxy error rate for that
confidence column, e.g. `--target_corruption rel:0.25` targets a quarter of
the observed base error rather than a fixed absolute number.

Writes `diagnosis.json` to `--out` and prints the same numbers as a table.

### Realizable budgets (`--n_groups_per_step`, `--pick`)

The 0.05 grid above assumes any budget fraction can be hit exactly. A harness
that verifies `k` of `n_groups_per_step` groups per training step cannot: it
only ever realizes `k / n_groups_per_step`, and `round(budget * n_groups)`
silently snaps any other recommendation to the nearest one (see
`RESEARCH_LOG.md`'s E10e-rel entry: a recommended budget of 0.15 at
`n_groups_per_step=8` became 1 of 8 groups, i.e. 0.125, and corruption came
out at 7.5 percent against a 4 percent target).

Pass `--n_groups_per_step N` to restrict the recommendation to the
`N` candidates `k/N` for `k = 1..N`, and `--pick point` (default) or
`--pick upper` to require the target under the point estimate or the
bootstrap upper 95 percent bound. The report shows the candidate table and
the recommendation under both rules, so you can see where they diverge.
Omitting `--n_groups_per_step` leaves all existing output unchanged.

## Library use

```python
import pandas as pd
from avrl_router.diagnose import diagnose

df = pd.read_parquet("runs/E08/feats.parquet")
result = diagnose(
    df,
    confidence_cols=["agreement"],
    error_col="proxy_error",
    budgets=[0.1, 0.25, 0.5],
    target_corruption=0.02,
    n_boot=1000,
    seed=0,
)
```

## What each number means

- **proxy error rate**: fraction of groups where the cheap grader disagreed
  with the expensive grader, with a 95 percent bootstrap CI over groups.
  This is the corruption rate you would get with zero verification.
- **AUC**: how well (negative) confidence ranks groups by whether the cheap
  grader was wrong. Computed as `roc_auc_score(error, -confidence)`, so an
  AUC near 1 means the least-confident groups are reliably the wrong ones.
  AUC near 0.5 means confidence carries no information about correctness.
- **threshold corrupted, at a given budget**: if you verify the budget's
  fraction of groups with the LOWEST confidence (and trust the expensive
  label there), this is the fraction of ALL groups that still end up with a
  corrupted reward, counting only the unverified groups' errors.
- **random corrupted**: the same quantity if you verified a random budget
  fraction instead of the lowest-confidence fraction. Expected value is
  `(1 - budget) * proxy error rate`.
- **full-verification floor**: 0, the corruption you get at 100 percent
  budget. Always included as the reference bound.
- **relative reduction vs random**: `(random - threshold) / random`. How
  much of the random-allocation corruption the confidence-ranked threshold
  removes, at that budget.
- **recommended budget**: the smallest budget on a 0.05 grid (0.05 to 1.0)
  where threshold-based admitted corruption is at or below the effective
  target. Always reachable at budget 1.0, since full verification has zero
  corruption by definition.
- **target_mode / effective_target / base_error**: `target_mode` is
  `absolute` or `relative`, depending on whether `target_corruption` was a
  plain number or a `rel:F` string. `effective_target` is the number
  actually used to pick the recommended budget (the `target_corruption`
  value itself if absolute, or `F * base_error` if relative). `base_error`
  is that confidence column's measured proxy error rate (point estimate).
- **grid**: a compact 0.05-to-1.0 grid of threshold vs. random corruption
  (point estimates, no bootstrap CI), printed as a table so you can
  eyeball a different budget than the recommended one.
- **regime**: `strong` (AUC >= 0.85), `uninformative` (AUC < 0.65), `usable`
  (otherwise), with a one-line recommendation. The bands come from the
  project's in-training runs: vote agreement (AUC 0.87 to 0.91) and response
  consistency (AUC 0.67 to 0.69) both beat random allocation in training,
  reward-model margin (AUC 0.51 to 0.57) did not.
  - strong: use the confidence-ranked threshold at the recommended budget.
  - uninformative: this confidence signal is close to random, verification
    budget buys little per call, use a different confidence signal such as
    response consistency, or raise the budget.
  - usable: use the threshold; expect the relative reduction shown, not the
    strong-regime gain.

## Notes

- Bootstrap is over groups (resampling rows with replacement), seeded,
  1000 resamples by default (`--n_boot`). Ties in confidence are broken by
  original row order (stable sort), not randomly.
- If a resampled bootstrap draw has only one class in the error column, that
  draw is dropped from the AUC confidence interval rather than counted as a
  degenerate 0 or 1.
- This tool does not fabricate a confidence column. If your data does not
  have one, it will not run; that is intentional.

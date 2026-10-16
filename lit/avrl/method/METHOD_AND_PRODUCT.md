# Method and product synthesis for AVRL

Inputs: the six direction files (D1 to D6) in this folder and three independent judge passes over them. This file ranks the directions, proposes one coherent method combining the strongest pieces, sketches the paper and the product around it, and appends five new backlog entries.

## 1. Ranked table (averaged across three judges)

Each cell is the mean of the three judges' 1-10 scores. "Avg" is the mean of the four columns.

| Rank | Direction | Novelty | Feasibility | Publishability | Product value | Avg |
|---|---|---|---|---|---|---|
| 1 | D5: Reward-hacking defense (canary divergence alarm) | 5.3 | 7.3 | 5.7 | 6.7 | 6.3 |
| 2 | D6: Product framework (regime diagnostic + TRL/verl wrapper) | 2.3 | 7.7 | 3.0 | 8.7 | 5.4 |
| 3 | D2: Verifier cascade with online recalibration | 3.0 | 7.3 | 4.3 | 5.3 | 5.0 |
| 4 | D4: Spend budget at the GRPO learning frontier | 2.3 | 8.0 | 3.0 | 5.3 | 4.7 |
| 5 | D3: Structure-aware allocator (cell bandit + classifier + gate) | 3.7 | 5.3 | 4.7 | 4.3 | 4.5 |
| 6 | D1: Self-consistency gating | 2.3 | 8.7 | 3.0 | 3.7 | 4.4 |

Reading: nothing here scores high on novelty alone; every direction reuses published machinery (self-consistency, Trust or Escalate, VIP, ProbeLLM, Heteroskedastic Signals' HGA, DAPO/PCL/MoPPS, Baker et al.'s CoT monitor). D5 ranks first because no cited paper combines an unbiased canary sample with a trend-triggered budget reallocation checked against an injected ground-truth exploit; D6 ranks second almost entirely on product value, since it is packaging rather than research; D1 ranks last because it is the most completely pre-published idea with the lowest product value. The three lowest-scoring items on novelty (D1, D4, D6) are exactly the three the project's own ledger already treats as settled recipes. This argues for using D1, D2, D4, and D6 as commodity infrastructure inside a bigger system, and spending the actual research claim on D5's hacking-detection mechanism and D3's weak-signal allocator, which is also what the project's two measured regimes already demand.

## 2. THE METHOD: AVRL-Router

One system, not six papers. It is a regime-conditional router with a safety layer, built entirely from pieces the ledger has already measured: a regime detector (from D6) chooses between a recalibrated self-consistency gate (D1 + D2) in the strong-signal regime and a structure-aware cell-bandit allocator (D3) in the weak-signal regime, and a canary divergence alarm (D5) runs underneath both regimes as an override that can pull budget toward a cluster showing reward-hacking-shaped drift, regardless of which regime is active.

```
ALGORITHM AVRL-Router

INPUTS
  cheap(prompts, completions) -> (score, confidence_features)   # vote margin, RM score, etc.
  expensive(prompts, completions) -> score                       # judge, gold, metered
  budget_pct B                                                    # fraction of groups verified
  canary_frac C (e.g. 0.02)                                       # fraction of groups held out, never routed by policy
  recalibrate_every K (e.g. 25 steps)
  regime_check_every N (e.g. 100 steps)
  cell_embedder (frozen sentence embedder), num_cells k

STATE
  regime in {STRONG, WEAK}, initialized UNKNOWN
  tau (threshold on confidence_features), fit online
  cell_posteriors[c] = Beta(a_c, b_c) for c in 1..k           # D3
  classifier (logistic, cheap features -> corrupted-reward prob)  # D3
  canary_set: fixed groups, drawn once at init, stratified by cell, excluded from all routing decisions  # D5
  canary_baseline[c], canary_noise_band[c]                     # D5
  buffer: FIFO of (group, cheap_score, expensive_label) for verified non-canary groups  # D2

PER TRAINING STEP (one GRPO batch of rollout groups)
  1. Score every group with cheap(); compute confidence_features.
  2. REGIME CHECK (every N steps, or at step 0):
       auc = AUC(confidence_features -> cheap-vs-expensive disagreement, over buffer)
       nbr = neighbor_ratio(confident subset, cell_embedder)      # D6's phase-0 diagnostic
       regime = STRONG if auc >= 0.85 else WEAK
       (STRONG bar matches ledger: GSM8K/MBPP sit at 0.87-0.93; WEAK matches
        open-ended reward-model-vs-judge at 0.75-0.80 with neighbor ratio ~2.0
        in the confident subset)
  3. CANARY ALARM (every verification-eligible step):
       for each cell c with scheduled canary groups this step:
         verify them with expensive(); update divergence(c) = mean(cheap - expensive)
         if divergence(c) exceeds canary_baseline[c] + 2*canary_noise_band[c]:
            flag c as HOT for this window                         # D5
  4. BUDGET SPLIT: reserve C*batch_size calls for canary (step 3, always spent,
     never chosen by the policy); remaining (B-C)*batch_size calls allocated by regime:
       IF regime == STRONG:
         tau = online percentile of confidence_features such that exactly
               (B-C) fraction of this batch falls below tau            # D1+D2
         verify groups with confidence_features < tau
       IF regime == WEAK:
         sample theta_c ~ cell_posteriors[c] for each cell present this batch
         rank cells by theta_c; cells flagged HOT by step 3 are forced to the
         top of the ranking regardless of theta_c                      # D5 overrides D3
         within chosen cells, classifier ranks groups by corrupted-reward
         probability; spend remaining budget on top-ranked groups        # D3
  5. For unverified groups, reward = cheap score. For verified groups
     (canary + routed), reward = expensive score; log disagreement as a
     corrupted-reward-avoided count.
  6. Update buffer, cell_posteriors (Beta update from this step's labels),
     classifier (refit every M steps on a sliding window with inverse-
     propensity weighting by theta_c at selection time), and canary_baseline
     (slow EMA, canary groups only, so policy drift cannot bias it).
  7. Every K steps, independent of regime: if STRONG, recalibrate tau from
     the buffer (not from live routing decisions); if WEAK, refit cell
     posteriors' priors from the buffer to bound the self-reinforcement
     failure already measured in training (ledger P1-goldberg-s0: a pure
     online learner's corrupted admissions rose over training while the
     fixed threshold's fell).

BUDGET ACCOUNTING (reported every step)
  expensive_calls_spent, canary_calls (always C*batch), routed_calls (<= (B-C)*batch),
  corrupted_rewards_admitted_estimate (disagreement among unverified groups,
  estimated from the buffer's base rate), regime label, hot_cells this window.
```

Why this composition and not another: the regime detector is cheap and already validated (D6's smallest experiment reuses existing diagnostics with zero new cost). Gating strong-signal tasks with a recalibrated threshold rather than a learned allocator avoids the one failure the project has actually observed in training (P1-goldberg-s0: a self-training allocator admitted 2.7-3x more corrupted rewards than the threshold, and the gap grew). Routing weak-signal tasks to the structure-aware allocator uses the only regime where a learned component has shown a replay edge with intervals excluding zero (L9a, L9d: +19 to +31 percent). The canary alarm sits outside both regimes because reward hacking is a trajectory property, not a static regime property: a task could sit in the strong-signal regime today and still be hacked tomorrow if the policy finds an exploit the vote-margin signal does not see, so the alarm needs an unbiased, policy-blind sample regardless of which allocation policy is otherwise in charge.

## 3. The paper

**Title:** "Route, Don't Guess: A Regime-Conditional, Hacking-Aware Budget for Verification in RL Post-Training"

**Abstract (draft):** Reinforcement learning post-training of language models increasingly relies on two-tier reward signals: a cheap grader used on every rollout and an expensive grader (a strong judge, a human, ground truth) affordable on only a fraction of them. We show that the right allocation rule for this budget is regime-dependent and measurable before training starts: where the cheap grader's own confidence predicts its errors (self-consistency tasks like math and code, AUC 0.87-0.93), a fixed, periodically recalibrated threshold is near-optimal and a learned allocator actively hurts, admitting more corrupted rewards as the policy drifts. Where the cheap grader is weak (a reward model scoring open-ended text, AUC 0.75-0.80) and its errors cluster in prompt space (neighbor ratio ~2.0 among confident predictions), a structure-aware allocator that learns which prompt regions are dangerous beats the budget-matched threshold by 19-31 percent in replay. We combine both findings into AVRL-Router, a single system that diagnoses its own regime and switches policy accordingly, plus a canary-based divergence alarm that detects and reallocates toward reward hacking independent of regime. We release the router as a drop-in TRL reward wrapper.

**Three contributions:**
1. A validated, two-statistic regime diagnostic (cheap-feature AUC, confident-subset embedding neighbor ratio) that predicts, before training, whether a fixed or a learned verification policy wins, checked across nine task/proxy pairs in replay.
2. AVRL-Router: a regime-conditional allocator that composes a recalibrated self-consistency threshold with a structure-aware cell-bandit allocator, shown in replay to match the strong-signal regime's near-zero-loss optimum and the weak-signal regime's double-digit gain in a single system with no manual tuning.
3. A canary divergence alarm that detects an injected, ground-truth-labeled reward-hacking exploit from an unbiased held-out sample and reallocates budget toward the affected prompt cluster, tested against a static budget-matched threshold.

**Headline figure:** one scatter, cheap-grader AUC (x-axis) against allocator gain over the budget-matched threshold (y-axis), with the project's nine measured points (GSM8K, MBPP near zero at AUC 0.87-0.93; HelpSteer2 at +19 to +31 percent at AUC 0.75-0.80), AVRL-Router's own regime calls marked on each point, and an inset panel showing the canary alarm firing before and budget visibly migrating into the hacked cluster after, on the injected-exploit run.

**Baselines:** Full verification (upper bound), Random, fixed-once threshold, online-recalibrated threshold (no regime switch), the existing self-training LearnedAllocator, the existing CellBandit alone (no regime gate, no canary), Oracle (uses gold labels for free).

**Experiments mapped to existing harnesses:** regime-diagnostic validation runs on `avrl/p0_analyze.py` over saved rollouts (E01, E08, E09d) at zero new compute; the strong-signal arm reuses `avrl/p1_grpo.py` on GSM8K; the weak-signal arm reuses `avrl/p1_judge_grpo.py` on HelpSteer2 with the 7B AWQ judge; `avrl/allocators.py` gains `RecalibratedThresholdAllocator`, `StructureAwareAllocator`, and `DivergenceAllocator` as composable pieces rather than separate scripts; the hacking experiment reuses the marker-string injection already specified in D5 inside `p1_grpo.py`.

## 4. The product

**Package name options:** `avrl-router` (literal, searchable), `gradewise` (carried over from D6's suggestion, friendlier), `cheapjudge` (memorable, slightly cute), `verigate`. Recommendation: `avrl-router`, since it is the most literal and the project's own name already has recognition inside the lab.

**One-page API sketch (Python):**

```python
from avrl_router import Router, graders

cheap = graders.SelfConsistency(n=8)          # or graders.RewardModel(ckpt="...")
expensive = graders.Judge(model="qwen2.5-7b-instruct", rubric="...")  # or graders.GroundTruth(fn=...)

router = Router(
    cheap=cheap,
    expensive=expensive,
    budget_pct=0.25,
    canary_frac=0.02,
    recalibrate_every=25,
)

diagnosis = router.diagnose(sample_prompts)    # phase-0 diagnostic, no training
print(diagnosis.regime, diagnosis.policy)       # "strong_signal" / "threshold" or "weak_signal" / "cell_bandit"

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[router.as_trl_reward_fn()],
    ...
)
trainer.train()

router.dashboard()   # calls spent, corrupted-rewards-avoided estimate, regime, hot cells, hack alarm state
```

**Demo:** a single-GPU notebook running two tracks side by side on the project's own measured regimes: GSM8K (router diagnoses strong-signal, picks the threshold, matches the fixed-rule ledger numbers) and HelpSteer2 plus the 7B judge (router diagnoses weak-signal, picks the cell bandit, reproduces the L9d replay gain), with a third cell injecting the marker-string exploit on the GSM8K track to show the canary alarm firing.

**Install story:** `pip install avrl-router`; no new heavy dependency beyond what the harness already uses (sentence embeddings for cells, scikit-learn-scale logistic regression and Beta posteriors, both cheap on CPU). Works as a drop-in `reward_funcs` entry for TRL's `GRPOTrainer`, matching its existing weighted-sum, multi-function convention; a `verl` reward-manager wrapper is a stated future item, not v0.

**What it reuses from TRL:** the `reward_funcs` interface and its `log_metric`/`log_extra` hooks (confirmed against current GRPOTrainer docs, https://huggingface.co/docs/trl/main/en/grpo_trainer), and the existing LoRA/PEFT training loop already wired into `avrl/p1_grpo.py` and `avrl/p1_judge_grpo.py`.

**What the owner should build first:** the regime diagnostic and dashboard alone, as a standalone CLI (`avrl-router diagnose`), since it costs nothing new, it is the piece every judge scored highest on product value (D6), and it is useful even to someone who never adopts the full router. Wire it into the TRL reward wrapper second. Defer the canary alarm's UI and the verl wrapper until the in-training weak-signal result (the decisive hannah run) lands, since the paper's claim 2 and the alarm's claim both depend on it.

## 5. Backlog additions

Five new entries, M01 to M05, appended to `avrl/EXPERIMENT_BACKLOG.md` under the heading "Method and product experiments."

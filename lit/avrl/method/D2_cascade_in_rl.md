# D2: Verifier cascades inside the RL loop, recalibrated online

## The idea in plain language

Right now the harness picks between two graders per rollout group: a cheap one (majority vote, a small reward model) and an expensive one (a gold answer, a big judge). Add a third rung in between, a medium grader (a small judge, 3B-class), and two escalation gates instead of one. Most groups stop at the cheap grader. Groups it is unsure about go to the medium grader; groups the medium grader is still unsure about, or flags as likely wrong, go to the expensive grader. The two gate thresholds are not set once and left alone. They are refit on a schedule from a small buffer of groups that already got the expensive label, and a drift check watches whether the cheap grader's error pattern is still what it was when the thresholds were last set. If it has moved, the schedule forces an early refit instead of waiting for the next one.

This only matters in the regime already found in the ledger: self-consistency tasks (GSM8K, MBPP) where the cheap signal is strong (AUC 0.87 to 0.93) do not need any of this, a threshold set once is already near-optimal there (E01, E08, E06). The open-ended regime, reward model versus judge (AUC 0.75 to 0.80, confident-group neighbor ratio 2.0, E09d, L9d), is where a cascade with online recalibration has something to prove.

## The mechanism, precisely

1. Cheap grader scores every rollout group, every step.
2. A confidence feature is computed from the cheap score (vote margin for self-consistency, RM score spread for reward-model proxies, reused from `avrl/p0_analyze.py`).
3. Two thresholds, tau_1 (cheap to medium) and tau_2 (medium to expensive), partition groups into three bins.
4. Every K steps, both thresholds refit from a bounded FIFO buffer of the last N groups that reached the expensive tier, chosen to hit target escalation rates under a hard budget (medium calls <= B_m, expensive calls <= B_e).
5. A drift statistic, the error rate inside the cheap grader's most-confident bin, is tracked on a sliding window against its value at last calibration. A move past a fixed band triggers an early refit.
6. Budget is reported in expensive-call-equivalents (medium plus expensive calls), comparable to the existing ledger entries at the same spend.

## Prior art

- **Trust or Escalate** (arXiv 2407.18370, https://arxiv.org/abs/2407.18370). Escalates from a weak judge to a strong judge when a calibrated self-trust check (simulated annotators) says the weak verdict likely disagrees with a human. Calibrated once, offline, on a fixed evaluation set, against a frozen model being judged. Two tiers, no training loop, no drift. Project notes list the venue as ICLR 2025; that specific claim is UNVERIFIED here, only the arXiv record was checked.
- **Heteroskedastic Signals in Budgeted LLM Verification** (Jinlong Yang, arXiv 2606.15841, https://arxiv.org/abs/2606.15841, 2026). Shows that under heterogeneous cost and error regions a single global threshold is often beaten by cost-stratified thresholds, and proposes a pre-test (Heterogeneity-Gated Allocation) to decide whether stratifying is worth it, on MBPP and MATH. Calibration is done once, at evaluation time, on static models. Close kin to this project's two-tier finding (fancier allocation only helps where the signal is heterogeneous), but it never runs inside a training loop or revisits its calibration.
- **SOFisher** and **VIP**, already logged in `avrl/PROJECT.md` and the gap-check files, learn in-training budget allocation but neither targets a drifting proxy-versus-expensive-grader gap or a recalibration schedule.
- Closest prior art of all is this project's own ledger entry P1-goldberg-s0: the existing `LearnedAllocator` in `avrl/allocators.py`, run in training on GSM8K, admitted 2.7 to 3 times more corrupted rewards than the fixed threshold because it trains online on the biased sample of groups it chose to verify, and the gap widened as the policy improved. A buffer-based, periodically refit threshold avoids exactly that, by separating what gets verified from what sets the next threshold.

## The exact novelty claim that survives

Not "cascades help": Trust or Escalate already claims that, at evaluation time. Not "learned beats fixed": this project's own data says that is regime-dependent, and the learned allocator's self-reinforcement is actively harmful on at least one task. The claim that survives: in the weak-signal, clustered-error regime (open-ended, RM-versus-judge), a policy that keeps the simple shape of a threshold but refits it from a buffer logically separate from its own routing decisions can capture the adaptive allocator's replay-measured gain (L9a: +12 to +26 percent corrupted rewards caught; L9d: +19 to +31 percent) without the self-reinforcement failure seen on GSM8K, because recalibration is decoupled from exploitation. This is falsifiable in training, not just replay, the open question STATUS.md flags as decisive.

## Smallest experiment

Reuse the E09d configuration (HelpSteer2, 1.5B policy, 0.6B reward model cheap, 7B AWQ judge expensive, 25 percent budget): the best-powered weak-signal point in the ledger (225 error groups, L9d).

Files to change:
- `avrl/allocators.py`: add `RecalibratedThresholdAllocator`, a two-tier allocator (skip the medium rung for this smallest run) that refits its threshold every K=25 steps from the last N=50 verified groups, plus a drift trigger reusing the confident-subset error-rate statistic from `avrl/p0_analyze.py`.
- `avrl/p1_judge_grpo.py`: log (group, cheap_score, expensive_label) for every verified group into the buffer the allocator reads, and log each recalibration event (scheduled or drift-triggered) with the threshold value and step number.
- `avrl/exp/`: a new script cloning the E09d/P1-judge setup, three arms (fixed-once, `LearnedAllocator`, `RecalibratedThresholdAllocator`), 2 seeds, same step count as the queued hannah run.
- `avrl/RESEARCH_LOG.md`: pre-register pass and kill before running, matching existing format.

Pass test: recalibrated matches or beats learned on corrupted-reward admission and held-out judge score at equal budget, across both seeds. Kill test: if recalibrated does no better than fixed-once on both metrics, the schedule adds nothing and the two-tier fixed-threshold recipe stands as the deliverable. Run on goldberg first (fits the 16 GB card with the existing AWQ judge) before competing for the contended hannah queue.

## Publication shape

Venue: a 2027 workshop on efficient or budgeted RL for language models (NeurIPS or ICML workshop track); main track only if the result generalizes past this one task, which is UNVERIFIED and untested. Working title: "When to Recalibrate: Budgeted Verifier Cascades Under Policy Drift in RL Post-Training." Headline figure: cumulative corrupted rewards admitted versus training step, three lines (fixed-once, learned, recalibrated), vertical ticks marking recalibration events, faceted by regime (self-consistency task where all three tie, open-ended task where recalibrated tracks learned's gain without its blowup).

## Product angle

What a user would call: bring your own cheap grader (a vote rule or small reward model) and expensive grader (a judge API, test suite, or human queue), set a budget fraction, and the framework runs the escalation and recalibration: `avrl-cascade --cheap rm.ckpt --expensive judge-api --budget 0.25 --recalib-every 25`. What they would see: a running log showing the current threshold, a timeline of recalibration events, spend so far against the budget, and a warning like "confident-group error rate rose from 4 percent to 9 percent, recalibrating early" whenever drift fires. The pitch is narrower than "replaces your judge": stop guessing one threshold at the start of a multi-day run, and stop trusting a learned router that quietly overfits to what it chose to check.

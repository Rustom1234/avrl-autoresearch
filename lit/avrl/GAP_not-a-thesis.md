# GAP: Not a Thesis

## Verdict

Even in its best-case outcome (Phase 1 kill test cleared), AVRL as scoped in THESIS_PROPOSAL.md is an engineering result, not a thesis. It shows that one learned allocator beats one fixed threshold on one task, one model, one proxy, at one budget. It does not establish why, under what conditions the result would reverse, or how to predict the outcome on an untested task without rerunning the experiment.

## The argument

**1. It is a substitution, not a principle.** SYNTHESIS.md's own risk analysis (section 4, risk 3) states the defensible novelty "narrows specifically to substituting verification cost and reward corruption for rollout/token cost as the optimized resource." The mechanism, a learned, budget-capped, in-training allocator, already exists for rollout count (VIP, Nguyen et al., ICLR 2026, https://arxiv.org/abs/2602.01601; SARA, Nomand et al., 2026, https://arxiv.org/abs/2607.26253) and for evaluation-time judge cascades (Trust or Escalate, Jung, Brahman, Choi, ICLR 2025, https://arxiv.org/abs/2407.18370; FrugalGPT, Chen, Zaharia, Zou, 2023, https://arxiv.org/abs/2305.05176). Relabeling the budgeted resource from "rollouts" to "verifier calls" while reusing the same allocation machinery is a systems contribution, not a new scientific claim.

**2. The named contributions are a swap, a metric, and a measurement, not a theory.** THESIS_PROPOSAL.md's "Three Contributions" section lists: (1) substituting verifier choice for rollout count as the optimized resource, (2) a new evaluation metric (corrupted-reward admission rate), and (3) an empirical test of whether known proxy-error structure (RewardBench, Lambert et al., https://arxiv.org/abs/2403.13787; Gao, Schulman, Hilton, 2022, https://arxiv.org/abs/2210.10760) survives inside a moving-policy training loop. None is a general, falsifiable principle that predicts behavior on an untested task. Compare to genuinely theory-shaped neighbor work: Skalse, Howe, Krasheninnikov, Krueger (2022, https://arxiv.org/abs/2209.13085) prove a general theorem about when any imperfect proxy is hackable, independent of task; Gao, Schulman, Hilton (2022, https://arxiv.org/abs/2210.10760) give a scaling law relating the gold-proxy gap to KL divergence, predictive across model and proxy sizes. AVRL's proposal offers no analogous lawful relationship, only a task-specific comparison.

**3. The proposal's own kill criteria concede this.** THESIS_PROPOSAL.md states that if Phase 1 fails to beat the threshold arm beyond the spread across seeds, the deliverable becomes "the fixed-threshold recipe," explicitly an engineering artifact, not a scientific finding. Even the positive branch is framed as "a system that beats X at equal budget," a benchmark claim, not an explanation of a mechanism.

**4. Transfer is deferred, not central.** Cross-task-family transfer, the one test that would show the method or finding generalizes, is listed in THESIS_PROPOSAL.md as a Phase 2 item contingent on "Phase 1 surviving" (months 10-11), not a primary test. A result not yet shown to transfer beyond GSM8K plus majority vote cannot claim generality.

## What would make it a thesis

1. A general, falsifiable account of when and why proxy error clusters in prompt space, stated in measurable task properties (for example a proxy-error autocorrelation or entropy statistic), that predicts before the experiment runs whether a learned allocator beats a fixed threshold on a new task and proxy pair.
2. Evidence that the fitted allocator, or the principle behind it, transfers zero-shot or near-zero-shot across task families and proxy types, made the central test rather than a contingent late-stage ablation.
3. A mechanistic account of how adversarial, online training changes the qualitative allocator-versus-threshold comparison relative to the static cascades already in the literature (Trust or Escalate, https://arxiv.org/abs/2407.18370; Heteroskedastic Signals, Yang, 2026, https://arxiv.org/abs/2606.15841), stated as a decision rule usable without retraining from scratch on each new task.

Without one of these three, the most honest description of the finished project is: a working, budget-efficient verification router for RLVR, useful, but an engineering trick.

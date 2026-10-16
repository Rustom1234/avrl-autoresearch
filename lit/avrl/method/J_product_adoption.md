# Judge pass: product-adoption lens

Scope: the six method directions in this folder (D1 to D6). Lens: would a team actually install and keep using this as a product or open framework, not whether it is a good research result. Scoring is 1 to 10 on novelty, feasibility, publishability, product value. Per the brief, a direction whose core mechanism is already published counts novelty 2 or less, even if the combination around it is new.

## D1: Consistency gating (self-consistency as a verifier gate)

- Novelty: 2. The direction's own text says it: "no new estimator is being proposed; agreement-as-confidence is self-consistency" (Wang et al., 2022, https://arxiv.org/abs/2203.11171; TTRL, 2025, https://arxiv.org/abs/2504.16084). This is an application of a known trick, not a new one.
- Feasibility: 9. Reuses rollouts already sampled for GRPO, no new generation cost, smallest experiment is a few lines against data already on disk.
- Publishability: 3. The direction itself calls it "not a standalone paper, an ablation inside the main AVRL submission."
- Product value: 4. As a packaged feature it is a free alternative confidence signal with no infrastructure cost, but a product cannot be built around it alone: it only matters if it beats the reward model's AUC on free text, an open question, so it ships as a config flag inside a bigger framework, not something to adopt on its own.

## D2: Verifier cascades with online recalibration

- Novelty: 3. The cascade shape is published (Trust or Escalate, https://arxiv.org/abs/2407.18370; the stratified-threshold idea in arXiv 2606.15841). The buffer-decoupled, drift-triggered refit is a real addition but a modest one layered on an existing pattern.
- Feasibility: 7. More moving parts than D1 (two thresholds, a FIFO buffer, a drift statistic) but still sits inside the existing allocator and harness files.
- Publishability: 4. Workshop-level, and the direction admits the main-track case needs generalization past one task, which is unverified.
- Product value: 6. This is a sellable, concrete pain point: "your threshold goes stale over a multi-day run and you will not notice." A console line like "confident-group error rate rose from 4 percent to 9 percent, recalibrating early" is the kind of thing a user remembers and tells a colleague about. Still a feature inside a framework, not a product on its own.

## D3: Structure-aware allocator (cell bandit plus classifier plus self-disabling gate)

- Novelty: 5. The individual pieces are published (VIP for in-training allocation, https://arxiv.org/abs/2602.01601; ProbeLLM for failure clustering, https://arxiv.org/abs/2602.12966; the regime-gate idea echoed in arXiv 2606.15841) but the specific combination, a bandit and classifier that can turn themselves off, was not found assembled elsewhere.
- Feasibility: 5. Three cooperating components, inverse-propensity correction, and a regime detector that has to be right about disabling itself; the project's own ledger shows a simpler learned allocator already self-reinforcing into more corrupted rewards (P1-goldberg-s0), so this design has more surface area for the same failure mode to hide in.
- Publishability: 5. Workshop-shaped, "a measurement-backed design choice plus one combination," by the direction's own description.
- Product value: 4. From an adoption standpoint this is the worst-engineered pitch of the six: a product wants one lever a user understands, and Thompson sampling over embedding cells plus a propensity-weighted online classifier is not something most teams can debug when it misbehaves at 2 a.m. The self-disabling gate is a good safety idea but it is a second thing that can be wrong on top of the first.

## D4: Spend the budget at the GRPO learning frontier

- Novelty: 2. The direction states this plainly: "mostly a reframing in the self-consistency regime... should reproduce E01/E08, not beat them," stitching together DAPO (https://arxiv.org/abs/2503.14476), PCL (https://arxiv.org/abs/2510.01135), and MoPPS (https://arxiv.org/abs/2507.04632), each already published.
- Feasibility: 8. Pass-rate bucketing from scores already computed, and the DAPO-style resample step is a contained addition to two existing scripts.
- Publishability: 3. "Not a standalone paper on these numbers," by its own account, one ablation figure inside the main submission.
- Product value: 7. The best pure-simplicity pitch in the set: a verification rule derived from the optimizer's own zero-advantage condition removes a tuning knob (tau) that every other direction here still needs fit or recalibrated. "We derive your threshold from GRPO, you do not tune it" is a one-line sell, and it composes with rollout-side savings (PCL, MoPPS) for a double efficiency story that plays well in a README.

## D5: Budgeted verification as a reward-hacking defense (canary divergence alarm)

- Novelty: 6. The closest published work (Baker et al., https://arxiv.org/abs/2503.11926) penalizes hacking with a constant-rate monitor on a separate signal; using the cheap-versus-expensive reward gap on an unbiased canary sample as both the alarm and the budget-reallocation trigger, inside RL training, was not found elsewhere in the cited set. The pieces (overoptimization scaling laws, https://arxiv.org/abs/2210.10760; Spurious Rewards, https://arxiv.org/abs/2506.10947) are diagnostic, not this mechanism.
- Feasibility: 7. The injected, ground-truth-labeled exploit is cheap to add (a marker string and a bonus term), and the canary-set and divergence tracking reuse the project's existing cluster code.
- Publishability: 6. Reward hacking and monitorability are a live topic right now; a safety or oversight workshop is a realistic, receptive target, more so than the cost-framing in D1 through D4.
- Product value: 7. This is the strongest emotional hook of the six for a buyer: "get told when and where your model starts gaming the cheap grader, before you burn the rest of your expensive budget finding out." That fear (silent reward hacking) is something teams already worry about, unlike an abstract cost-per-dollar optimization, which makes this easier to sell even though it ships as one callback inside the same framework as everything else.

## D6: The product framework itself (phase-0 diagnostic plus both allocators as a TRL/verl reward wrapper)

- Novelty: 3. The direction says so itself: "not learned beats fixed in general," and the regime-diagnostic idea has a close published cousin, Heterogeneity-Gated Allocation (arXiv 2606.15841, confirmed via OpenAlex and arXiv). What is being judged here is packaging, not a new mechanism; by the brief's own rule, a mostly-already-measured result wrapped in a library does not earn high novelty just because nobody shipped the wrapper.
- Feasibility: 8. Every piece already exists in this repo (`allocators.py`'s threshold and self-training allocators, the phase-0 AUC and neighbor-ratio statistics, TRL's documented `reward_funcs` signature); the work is interface and dashboard, not new science.
- Publishability: 4. "A workshop paper... a measurement result, not a new mechanism," by its own account; it gets a paper mention, not a main-track claim.
- Product value: 9. This is the only one of the six that is actually shaped like a product: a two-function grader interface, a one-call diagnostic that tells the user which policy to run, a named package (gradewise, tentative), drop-in hooks for both TRL and verl, and a demo that is literally "the tool picks the right policy in both of this project's own measured regimes without the user hand-tuning anything." For product-adoption, that pitch, decide for me, do not make me learn the allocator theory, beats every other direction's pitch, because D1 through D5 are each one feature this wrapper would contain, not a reason to install something separate.

## Ranking under this lens

1. D6 (product value 9) is the deliverable: it is the only direction that is a thing a stranger would install rather than a paper section.
2. D5 (product value 7) is the best feature to put inside D6 first, because its pitch (catch hacking before you pay to find out) sells itself independent of whether the allocator-vs-threshold research question resolves either way.
3. D4 (product value 7) is the best second feature: free, no-tuning, composes with rollout savings.
4. D2 (product value 6) is a good third feature once the framework has users running multi-day jobs long enough for drift to matter.
5. D1 (product value 4) and D3 (product value 4) are the weakest from this lens for opposite reasons: D1 is too thin to be more than a config flag, D3 is too complex and too close to the self-reinforcing failure this project already caught once in `avrl/allocators.py`'s learned allocator.

Caveat: publishability and product value pull in different directions on purpose here. The two highest-novelty directions (D5, D3) are not the two highest product-value directions (D6, D5, D4); a judge scoring for a paper would rank D3 and D5 above D4 and D6, which this lens does not. UNVERIFIED: no numbers in this file are new measurements; all cited AUC, neighbor-ratio, and replay-gain figures are restated from `avrl/STATUS.md` and `avrl/PROJECT.md`, and all prior-art citations are restated from the D1 to D6 files themselves, not independently re-fetched in this pass.

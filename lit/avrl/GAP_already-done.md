# GAP: Already Done? (Adversarial Check)

**Verdict: Not fully closed, but substantially narrower than THESIS_PROPOSAL.md assumes. No single retrieved paper performs learned, online, budget-capped verifier-choice allocation inside RLVR reward scoring. Two independent halves of the mechanism are each already published at production scale, which is the strongest available already-done argument.**

## The two halves, each already shipped

**Half 1, the allocator mechanism.** VIP (Nguyen et al., ICLR 2026, https://arxiv.org/abs/2602.01601) already trains, online and jointly with a GRPO policy, a learned allocator (Gaussian process over per-prompt success probability, convex optimization under a hard compute budget) that beats uniform and heuristic allocation on the exact RLVR training loop AVRL targets. Its action space is rollout count, not verifier choice, but the learned-versus-fixed comparison AVRL's Phase 1 proposes is structurally the comparison VIP already ran and won.

**Half 2, the decision target.** Heteroskedastic Signals (Yang, 2026, https://arxiv.org/abs/2606.15841) and Trust or Escalate (Jung, Brahman, Choi, ICLR 2025, https://arxiv.org/abs/2407.18370) already frame "expensive verifier or cheap proxy, under budget" as the decision, though both fix the rule at evaluation time rather than learning it inside a training loop, as the existing SYNTHESIS.md already notes.

A third paper closes more distance than SYNTHESIS.md credited. OpenFC (Wang, Qiu, Lin, Lv, Chen, Wang, Yang, Zhang, 2026, https://arxiv.org/abs/2609.33579) post-trains Qwen3-8B with RL under what it literally calls "Verification-Aware Reinforcement Learning (VA-RL)" and "budget-aware tool rewards," to decide online when to invoke an external verifier. This is agentic tool-budgeting inside open-search fact-checking, not reward-scoring for RLVR; I could not confirm from the abstract whether the decision is per-instance, per-group, or under a hard cap (UNVERIFIED). But the name "verification-aware RL" is no longer an open term, and a defense committee would expect this paper to be cited and distinguished explicitly, not just added to a list.

## Why this does not fully close the gap

No retrieved source combines both halves inside the RLVR reward-scoring loop specifically. VIP never touches verifier identity. Heteroskedastic Signals and Trust or Escalate never touch online training. OpenFC touches a different loop (tool calls during generation, not reward computation on completed rollouts) and a different task family (fact-checking, not RLVR). The "corrupted reward admission" metric was not found in any source; S01-S14, SYNTHESIS.md, and THESIS_PROPOSAL.md all already confirm this, and nothing surfaced in this pass closes it either.

## Why the gap is thinner than the proposal argues

THESIS_PROPOSAL.md frames "substituting verifier choice for rollout count" as the research contribution. Given that at least nine 2026 preprints (per S07: VIP, SARA, PAIR, Pilot-Commit, GPS, DARE, Kalman-Curriculum, ThinkPrior, LEEPS) already prove "learned beats fixed, budget-capped, in-training allocator" on one resource axis, and OpenFC already proves a learned, RL-trained verification-budget gate on an adjacent axis, the remaining novel surface is narrow: port a known allocator mechanism to a known decision, inside one specific loop, plus one new bookkeeping metric. A hostile reviewer would ask why VIP's Gaussian-process-plus-convex-optimization machinery should not simply be re-applied with "verify: yes/no" as the allocated resource instead of rollout count. THESIS_PROPOSAL.md's current design (logistic regression plus Thompson sampling) does not run that simpler transfer as a baseline anywhere in Phase 0 through Phase 2, which weakens the case that the chosen method, not just the chosen axis, is the source of any measured edge.

## Surviving, undisputed gap

No retrieved paper does verifier-choice allocation, learned online, inside the reward-scoring step of an RLVR loop, evaluated on a corrupted-reward-admission metric. That specific combination remains open. The defensible claim narrows from "novel mechanism" to "novel combination of two known mechanisms plus one novel metric," which is real but weaker than the proposal's framing.

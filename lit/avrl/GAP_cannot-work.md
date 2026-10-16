# Adversarial Gap Check: Cannot-Work

## Verdict

The claim survives, but barely, and only with the right comparison. The literature does not prove a learned allocator cannot beat a fixed threshold, but it supplies an evidenced mechanism by which the margin collapses toward zero, one the current experimental plan does not address.

## The cannot-work argument

**1. The closest empirical analog says the fixed rule already captures most of the gain.** Heteroskedastic Signals in Budgeted LLM Verification (Yang, 2026, https://arxiv.org/abs/2606.15841) reports that a static, non-learned cost-stratified threshold recovers most of the achievable gain on MBPP/MATH, and frames structural heterogeneity as something that "limits optimization gains" for any allocator. That is a ceiling argument: if heterogeneity caps how much any policy can extract, a learned allocator's extra degrees of freedom have little headroom left to exploit.

**2. The allocator inherits a selection-bias blind spot a recalibrated threshold does not have.** THESIS_PROPOSAL.md's method section flags this: the online regression is refit only on groups the allocator itself chose to verify, so "an allocator that stops looking where it predicts low error could blind itself to drift there." A threshold recalibrated on stratified or random samples has no such feedback loop, since its calibration set is not filtered by its own past decisions. Over training, this should erode, not grow, the learned allocator's information advantage.

**3. The trained policy can learn to game the allocator, not just the verifier.** Baker et al. (OpenAI, 2025, https://arxiv.org/abs/2503.11926) show a policy that can infer a monitor's pattern learns to hide misbehavior from it rather than stop, an "obfuscated reward hacking" result under CoT monitoring. Applied here, the rollout-generating policy observes which outputs get escalated and can drift cheap-feature statistics (majority-vote agreement, length, truncation) toward whatever region the allocator currently under-escalates, since that region is defined by the allocator's own learned, updatable parameters. A fixed or slowly-recalibrated threshold is a smaller, less observable target.

**4. Classical precedent from a cousin field.** Tomanek and Hahn (2010), cited in S06, report that learned cost-prediction models for active learning do not reliably beat simpler heuristics in practice, a precedent for "the learned allocator does not beat the threshold" as normal, not a strawman.

**5. Where learned budget allocators do work, the margin over a tuned heuristic is modest.** SARA (Nomand et al., 2026, https://arxiv.org/abs/2607.26253) matches dynamic-sampling accuracy with 22% fewer rollouts; matching is not beating by a meaningful margin. If the easier adjacent problem (rollout count, immediate signal) yields only modest gains over heuristics, the harder one here (verifier choice, delayed label, adversarial pressure per point 3) is unlikely to do better.

## Closest work already making this case

Heteroskedastic Signals (arXiv 2606.15841) is the paper that would need to be wrong for AVRL's headline claim to hold; Baker et al. (arXiv 2503.11926) supplies the mechanism for why the margin should shrink specifically during training, which static-evaluation papers like Heteroskedastic Signals cannot observe, since they never let a policy adapt to the allocator.

## What would settle it

Compare the learned allocator against a periodically recalibrated cost-stratified/HGA threshold (not a single fixed-once rule; UNVERIFIED whether Heteroskedastic Signals tests periodic recalibration), at equal expensive-call budget, tracking the accuracy and corrupted-reward-admission gap across training steps. A flat or shrinking gap relative to early-training values is direct evidence of point 3 and confirms cannot-work. A gap that holds or grows beyond 3-seed variance at the final step refutes cannot-work on this task.

## Surviving gap

No retrieved source runs this recalibrated-threshold-versus-learned-allocator race inside an active RL loop with the policy free to adapt; it is untested, not resolved either way.

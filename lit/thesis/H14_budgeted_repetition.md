# H14: Budgeted Repeated Exposure in Data-Constrained Pretraining

## 1. Hypothesis and SOFisher mapping

When high-quality pretraining data runs out, labs repeat the corpus for several epochs, but uniform repetition wastes budget because some regions carry more remaining learning value than others at any given point in training. A policy that spends each additional (second, third, fourth) exposure on the corpus regions with the most favorable cheap signals (loss trend, forgetting rate), exploiting autocorrelation of that value across neighboring regions the way SOFisher exploits spatial autocorrelation of rare targets across tissue, should reach lower held-out loss than uniform epoching or static quality-weighted repetition at the same total token count.

## 2. Prior art

1. Muennighoff et al. 2023, "Scaling Data-Constrained Language Models," https://arxiv.org/abs/2305.16264 — repeating data up to roughly 4 epochs gives negligible loss degradation versus unique tokens; meaningful gains extend to roughly 16 epochs before returns end near 40 epochs. This is the standard uniform-repetition null model.

2. Hernandez et al. 2022, "Scaling Laws and Interpretability of Learning from Repeated Data," https://arxiv.org/abs/2205.10487 — repeating just 0.1% of training tokens 100 times measurably degrades test loss via a double-descent effect. Shows repetition harm is highly concentration-dependent, not just epoch-count-dependent.

3. Xue et al. 2023, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis," https://arxiv.org/abs/2305.13230 — systematic study finds dropout is the one regularizer that reliably mitigates multi-epoch degradation; most other regularization techniques do not help. Exact repetition-count thresholds reported in the paper are UNVERIFIED here (not independently re-confirmed this session).

4. DataComp-LM (DCLM), https://arxiv.org/abs/2406.11794 — DCLM-Baseline reaches 64% 5-shot MMLU on a 7B model trained on 2.6T tokens, illustrating that filtered high-quality pools are still small enough that multi-epoch training is often required in practice.

5. Phi-4 Technical Report, https://arxiv.org/abs/2412.08905 — raising synthetic-data epochs from 4x to 12x produced further MMLU gains with no observed overfitting, directly contradicting a hard 4-epoch ceiling for some data types.

6. Llama 3 Herd of Models (Meta, 2024) — pretrained on roughly 15T tokens total. Per-category repetition/epoch counts were not found in this search pass; treat any specific repetition schedule for Llama 3 as UNVERIFIED.

7. "When to Review: Spaced Repetition for Continual Pre-Training of Language Models," Aug 2026, https://arxiv.org/abs/2608.17530 — uses the SuperMemo-2 (SM-2) spaced-repetition algorithm to decide which previously-seen examples get replayed during continual pretraining, to fight catastrophic forgetting. Closest published mechanism in spirit to H14, but it schedules replay of old data after a distribution shift, not budgeted repeat-exposure of the current corpus during first-pass data-constrained pretraining.

8. Zuo et al. 2026, "Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Training," https://arxiv.org/abs/2607.04969 (ICLR 2026 DATA-FM workshop) — a "Memorization Window" signal derived from loss-retention dynamics decides when and how much to reuse data; their preliminary experiments show performance keeps improving with repetition "far beyond" the commonly cited 4-epoch limit. This is the most direct existing attempt at adaptive, signal-driven repetition scheduling and the strongest adjacent baseline.

9. "Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs" (TREC), https://arxiv.org/abs/2509.25380 — predicts per-example loss trajectories from a partial run to place high-quality data optimally within a single pass. Targets ordering within one epoch, not the repeat-count-per-region decision H14 targets.

10. Mindermann et al. 2022, "Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt" (RHO-Loss) — selects training points by "reducible loss" (current loss minus an irreducible-loss proxy) to prioritize learnable, not-yet-learned examples. Closest general mechanism for "value of revisiting a point is not fixed and can be estimated cheaply." Exact arXiv identifier UNVERIFIED this session.

11. "Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics," 2026, https://arxiv.org/html/2601.21698v2 — holds the data fixed and varies only ordering, isolating curriculum/ordering effects from data selection effects, useful methodological template for isolating H14's repetition-scheduling effect from data-mixture effects.

12. "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining," Nov 2025, https://arxiv.org/abs/2511.18903 — curriculum ordering substantially beats random shuffling under a constant learning rate, but that advantage shrinks sharply under a standard LR decay schedule. Important caution: any repetition-scheduling gain found in H14 must be checked under a realistic decayed LR schedule, not just a constant one.

## 3. What is genuinely open as of 2026

No paper found in this pass runs an RL or bandit policy that treats "which corpus region gets its Nth exposure next" as a sequential decision problem under a fixed total-token budget, compared explicitly against uniform epoching as the null, the way SOFisher compares learned FOV selection against raster scanning. The two closest works split the difference: "When to Review" (item 7) solves a related but distinct problem (post-hoc replay against forgetting in continual learning, not first-pass data-constrained scheduling), and "Train Smarter, Not Longer" (item 8) uses a hand-built memorization-window heuristic, not a learned policy, and reports only preliminary results from a workshop paper. Whether the SOFisher precondition of spatial (here, topical/semantic) autocorrelation in cheap signals like loss trend or forgetting rate actually holds across neighboring corpus regions has not been directly measured anywhere found. A year is needed because this requires, in sequence: (a) an offline measurement of whether that autocorrelation exists at all, (b) building and tuning an actual policy (not just a heuristic) once autocorrelation is confirmed, and (c) checking whether gains hold or grow across model scale, since the relevant null models (Muennighoff's scaling curves) are themselves multi-point curves that require several training runs, not one.

## 4. Three-condition check

**Targets spatially autocorrelated.** Partially supported, not directly verified. Ordering/curriculum work (items 11, 12) shows position and sequence effects on training are real and structured, implying corpus regions are not independent, but no paper directly measures autocorrelation of loss-trend or forgetting-rate signals across topically or semantically neighboring shards. UNVERIFIED, flagged as the first thing Phase 1 must check.

**Expensive measurement costs far more than cheap features.** Supported, arguably more strongly than in SOFisher itself. An extra epoch is a full additional forward-backward pass over a region (compute-expensive at the scale implied by DCLM/Phi-4/Llama-3-sized runs, items 4-6), while the "cheap features" (per-region running loss, a forgetting-rate statistic from checkpoints) are near-free byproducts of training already in progress. Cost asymmetry holds essentially by construction.

**Cheap features alone cannot rank targets.** This is the weakest leg for H14, and cuts against fit. Item 8 (Train Smarter, Not Longer) shows a simple heuristic built purely from a cheap loss-retention signal already captures a meaningful chunk of the achievable gain, unlike SOFisher's imaging setting where cheap state features are explicitly insufficient. This is evidence that a full RL policy may buy less over a well-tuned heuristic than SOFisher's setting would predict, and should be treated as a live risk, not a resolved condition.

## 5. Year plan

**Phase 1 (months 1-3).** Offline check of the autocorrelation precondition using existing training logs from a small run. Concrete first experiment, runnable in under a week on 4 A100s: a 125M-parameter decoder-only model, corpus split into roughly 200-500 regions by shard or topic, total budget fixed at 4 epochs' worth of tokens. Compare three conditions at matched total tokens: (a) uniform round-robin epoching (every region seen exactly 4 times), (b) static quality-weighted repetition (fixed weights from a one-time quality/perplexity score), (c) an adaptive scheduler that reallocates repeat-budget across regions using per-region loss trend and forgetting rate, recomputed periodically. Metric: held-out loss at fixed total tokens, plus a small downstream eval subset in the style of DCLM's suite (item 4). Strongest existing baseline: Muennighoff's uniform 4-epoch repetition curve (item 1), with Zuo et al.'s memorization-window heuristic (item 8) as the strongest adaptive baseline to beat.

**Phase 2 (months 4-8).** If the simple adaptive scheduler beats both baselines at 125M, replace the hand-tuned scheduler with a learned policy (bandit or small DQN, matching SOFisher's own approach) and test at 350M-1B parameters, checking whether the gain persists or grows with scale relative to a fitted Muennighoff-style scaling curve.

**Phase 3 (months 9-12).** Scale to a more production-realistic corpus size and mixture, ablate the compute overhead of running the policy itself against the compute saved, and write up.

## 6. Kill criteria

- If Phase 1's offline check finds no meaningful autocorrelation in loss-trend or forgetting-rate signals across neighboring regions, kill the project: without structure to exploit, no policy can beat a well-tuned bandit or even uniform allocation.
- If the Phase 1 adaptive scheduler does not beat static quality-weighted repetition by a nontrivial margin (for example, under roughly 0.5% relative held-out loss improvement at matched tokens), kill the RL framing and, at most, keep a cheap heuristic.
- If a heuristic in the style of item 8 captures most of the measured gain and a learned policy adds little on top, downgrade the project from "learned policy" to "adopt the cheap heuristic," consistent with the condition-3 tension flagged in section 4.

## 7. Impact on a lab's stack

Nearly every current frontier-adjacent effort (Muennighoff, DCLM, Phi-4, Llama 3, items 1, 4, 5, 6) is already operating in a data-constrained regime and already repeats data. A working scheduler would be a drop-in addition to existing dataloader and sharding infrastructure, reordering and reweighting repeat exposure without touching model architecture, optimizer, or the data-filtering pipeline itself, and would directly extend the effective high-quality token budget at fixed compute, the same lever DCLM- and Phi-4-style labs already lean on manually via static quality weighting.

## SCORECARD
novelty: 6
feasibility_1yr: 7
lab_impact: 6
sofisher_fit: 6
strongest_baseline: Memorization-window guided reuse (Zuo et al. 2026, arXiv:2607.04969)
one_line_pitch: Learn a policy that spends each extra pretraining epoch on the corpus regions whose cheap loss/forgetting signals predict the most remaining value, instead of repeating everything uniformly.

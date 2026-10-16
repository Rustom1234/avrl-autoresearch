# Twenty SOFisher-inspired thesis hypotheses for LLM training: ranked

Compiled 2026-09-22 in about 40 minutes. Method: 20 Sonnet literature reviews (H01 to H20, each with prior art, three-condition check, year plan, kill criteria, scorecard) followed by 20 adversarial Sonnet kill-tests (K01 to K20, each hunting for the paper that already did it). Web-search quota ran out partway; later agents verified via arXiv and Semantic Scholar fetches, so coverage is real but not exhaustive. Anything an agent could not confirm is marked UNVERIFIED in its file. Scores are the agents' own 1 to 10 ratings; "already done" is the kill-tester's rating where 10 means fully published.

## Top 5

1. **H9. Budgeted verification inside RL post-training.** Decide which GRPO or RLAIF rollouts earn the expensive scorer (judge, sandbox, human) using cheap proxies, exploiting autocorrelation of reward-model error across prompts and rollout groups. Kill-test rated it the least already-done (3). The bar to beat is not uniform sampling but a cost-stratified static threshold (arXiv 2606.15841 shows that captures most gains at evaluation time); nobody has run that baseline inside a live RL loop or measured reward-model-error autocorrelation across rollout groups. Phase 1: GRPO on a 1.5B model, math, expensive judge simulated by a larger model, metric = accuracy gain per expensive call. Lab change: the reward-computation step in RL post-training.
2. **H13. Budgeted defensive audit of pretraining data for poisoning.** Spend expensive poison tests (influence on trigger behaviour, retraining checks) on corpus regions cheap features and earlier hits flag. Already-done 3; the defining threat paper (arXiv 2510.07192, near-constant 250 poison documents, Anthropic and UK AISI) proposes no defence and calls for one; Delta-Influence is vision-only. Load-bearing risk: dispersed fluent poison may not cluster, so phase 1 is a clustering measurement with injected poison in a 125M corpus before any policy is built.
3. **H5. Continuous memorisation monitor across checkpoints.** A budgeted extraction-probe policy that learns which training-data regions to probe and persists across checkpoints. Already-done 5; the Hubble model suite (arXiv 2510.19811) already provides injected memorisation targets and checkpoints, which removes three months of infrastructure. Bar to beat: static duplication-rank probing with a Min-K%++ prefilter. Connects directly to the existing sensitive-data work and to today's finding that judge-flagged documents are high-loss.
4. **H10. Sequential annotation for pretraining filters.** Refit-as-you-go annotation instead of one-shot random annotation before distilling a filter. We already have a 2x result on FineWeb sensitivity, and a running codebase. The kill-test found SIEVE (arXiv 2410.02755, Oct 2024) already runs this loop for DCLM-style quality filtering, so the honest framing is an extension: sub-1% safety and legal labels (sensitivity, toxicity, licensed text), annotation-dollar-matched baselines, and downstream safety evaluation of the filtered corpus. Lowest risk, highest continuity, narrowest novelty.
5. **H16. Active monitoring of training runs.** A policy that allocates a fixed diagnostic budget across layers, steps and shards to catch instabilities earlier. Already-done 3 and the only hypothesis the kill-test marked "survives: yes". Weakest feasibility (5): needs an instability-induction recipe at 125M to 350M (Wortsman et al. 2023 give one) and a cheap latent-signal threshold baseline from arXiv 2601.13160 that must be beaten.

Umbrella that ties 1, 2, 3 and 4 together for an advisor: **budgeted audit policies across the LLM training stack**, one method family (sequential decision about where to spend an expensive measurement), instantiated on data poisoning, memorisation, sensitivity filtering, and RL reward computation, with the honest lesson from the September work that the gain comes from learning as you go, not from spatial navigation.

Alternates: H19 unlearning verification (novelty 7, already-done 5, a cheap heuristic in PreUnlearn must be ruled out first), H18 budgeted semantic dedup (already-done 3, but SPOKES shows diversification objectives beat dedup outright), H8 targeted synthesis (feasibility 8, but per-example active generation exists at arXiv 2512.00884).

Dropped as crowded or already published: H6 scaling-law experiment design (SL2, PLES, ADMIRE-BayesOpt), H20 active distillation (five on-point papers 2022 to 2025), H17 pairwise evaluation (SELECT-LLM, and theory caps the gain at logarithmic), H15 RL prompt selection (DAPO, PCL, PPM), H2 cluster curriculum (Group-MATES), H3 red-teaming (ProbeLLM occupies the mechanism), H14 repetition (a heuristic already captures the value), H4 preferences (a 2026 negative result on the phase-1 ablation), H1, H7, H11, H12 (each has a real but narrow gap and a strong static baseline).

## Full table
| id | topic | novelty | feasible/yr | lab impact | SOFisher fit | already done (kill-test) | survives | closest prior work |
|---|---|---|---|---|---|---|---|---|
| H1 | active evaluation | 5 | 7 | 6 | 7 | 6 | maybe | ATLAS: Adaptive Testing for LLM Evaluation, 2026, https://arxiv.org/abs/2511.046 |
| H2 | active curriculum | 6 | 6 | 5 | 7 | 7 | maybe | Group-Level Data Selection for Efficient Pretraining (Group-MATES), 2025, https: |
| H3 | active redteaming | 6 | 7 | 6 | 8 | 7 | maybe | ProbeLLM: Automating Principled Diagnosis of LLM Failures, 2026, https://arxiv.o |
| H4 | active preferences | 5 | 7 | 6 | 8 | 6 | maybe | Random Is Hard to Beat: Active Selection in online DPO with Modern LLMs, 2026, h |
| H5 | memorisation audit | 6 | 7 | 6 | 7 | 5 | maybe | Hubble: A Model Suite to Advance the Study of LLM Memorization, 2025, https://ar |
| H6 | active scaling experiments | 4 | 7 | 7 | 8 | 8 | maybe | Power-Law Entropy Search (PLES), arXiv:2609.01431, https://arxiv.org/abs/2609.01 |
| H7 | active acquisition | 5 | 6 | 5 | 6 | 6 | maybe | Crawl4LLM, ACL Findings 2025, https://arxiv.org/abs/2502.13347 |
| H8 | targeted synthesis | 5 | 8 | 7 | 8 | 6 | maybe | Towards Active Synthetic Data Generation for Finetuning Language Models, 2026, h |
| H9 | budgeted verification | 6 | 7 | 7 | 8 | 3 | maybe | Heteroskedastic Signals in Budgeted LLM Verification, 2026, https://arxiv.org/ab |
| H10 | sequential annotation | 6 | 7 | 6 | 8 | 7 | maybe | SIEVE, "GPT-4o as the Gold Standard: A Scalable and General Purpose Approach to  |
| H11 | active interp probing | 6 | 6 | 7 | 6 | 4 | maybe | MechRL, 2026, https://arxiv.org/abs/2605.26343 |
| H12 | contamination search | 6 | 7 | 6 | 7 | 6 | maybe | Beyond Surface-Level Similarity: Hierarchical Contamination Detection for Synthe |
| H13 | poison audit | 6 | 6 | 6 | 8 | 3 | maybe | Poisoning Attacks on LLMs Require a Near-Constant Number of Poison Samples, 2025 |
| H14 | budgeted repetition | 6 | 7 | 6 | 6 | 7 | maybe | Train Smarter, Not Longer: Memorization-Guided Data Reuse for Efficient LLM Trai |
| H15 | active rl prompts | 4 | 8 | 6 | 7 | 6 | maybe | Small Generalizable Prompt Predictive Models (PPM), Feb 2026, https://arxiv.org/ |
| H16 | active run monitoring | 6 | 5 | 6 | 7 | 3 | yes | Towards Robust LLM Post-Training: Automatic Failure Management for Reinforcement |
| H17 | active pairwise eval | 4 | 8 | 7 | 8 | 7 | maybe | Large Language Model Selection with Limited Annotations (SELECT-LLM), 2026, http |
| H18 | budgeted dedup | 7 | 7 | 6 | 7 | 3 | maybe | SEDD: Scalable and Efficient Dataset Deduplication with GPUs, 2025, https://arxi |
| H19 | unlearning verification | 7 | 6 | 6 | 7 | 5 | maybe | PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unl |
| H20 | active distillation | 6 | 7 | 7 | 8 | 8 | maybe | LLM on a Budget: Active Knowledge Distillation for Efficient Classification of L |

## Surviving gaps

- **H1**: Whether exploiting checkpoint-to-checkpoint score autocorrelation beats a strong per-checkpoint ATLAS baseline under a calibrated winner's-curse correction on judge-based benchmarks, a combination with zero located prior work.
- **H2**: Whether an online sequential policy with a rationed expensive-probe budget beats a strong static cluster-level selector like Group-MATES, not just domain-level online mixers like Aioli, has not been tested by anyone found.
- **H3**: Porting that exploration/exploitation-over-embeddings mechanism specifically to adversarial safety red-teaming with a real cheap-proxy-vs-expensive-judge/human cost asymmetry, and closing the loop into measured downstream safety-training effect, is still unshown.
- **H4**: Whether a learned, embedding-clustering-aware selector that adapts across 3-5 iterative DPO rounds beats the strongest active baselines (DRTS/DeltaUCB) and random on a capability-preserving margin, untested by any cited work.
- **H5**: an online policy that updates on observed extraction hits and persists across checkpoints as a continuous, budget-bounded monitor, evaluated against static duplication-rank probing, has not been published
- **H6**: transfer of the learned active-selection policy itself across architectures/tokenizers plus calibrated uncertainty validated against a genuinely independent held-out large run, neither of which any found paper claims
- **H7**: No paper benchmarks a sequential yield-updating crawl policy against Crawl4LLM's static influence score under matched scoring compute for LLM pretraining.
- **H8**: Whether an online region-level bandit/RL policy using only cheap probes beats per-example active-learning selection (not merely static/uniform) at equal verified-example budget is untested in any paper found here.
- **H9**: Jointly training a verification-budget policy inside a live GRPO/RLVR loop, tested against a cost-stratified static threshold rather than uniform-random, with direct measurement of RM-error autocorrelation across rollout groups, remains untested by any paper found.
- **H10**: Closed-loop sequential annotation has not been tested at true sub-1% base rates on safety/legal labels (toxicity, contamination, licensed text) against an annotation-dollar-matched random baseline with downstream safety evals.
- **H11**: whether sequential state (prior probe outcomes plus checkpoint-adjacency, not one-shot raw SAE activation) makes cheap signals predictive of expensive-intervention outcomes, which neither MechRL nor MIB tested.
- **H12**: Whether corpus contamination is strongly autocorrelated by host/domain at a granularity an adaptive policy can exploit to beat the 2025 static cascade at matched expensive-call budget.
- **H13**: No adaptive/sequential budget-allocation policy for expensive poison tests has been demonstrated on LLM pretraining text; it survives only if Phase 1 shows real clustering strength before further scaling.
- **H14**: Whether a learned sequential policy beats the tuned memorization-window heuristic specifically by exploiting measured cross-region autocorrelation, tested at matched tokens against both uniform and heuristic baselines, remains untested anywhere found.
- **H15**: A controlled, single-harness ablation isolating whether explicit embedding-neighborhood transfer beats a scalar per-prompt adaptive tracker (PCL/MoPPS/PPM-style) has not been confirmed to exist.
- **H16**: no paper trains a sequential policy allocating a fixed diagnostic budget over (layer, step, shard) triples to catch precursors early
- **H17**: No found system does online low-rank active pairwise ranking with cold-start checkpoint arms injected mid-training, though the search for this specific claim was time-boxed and incomplete (Semantic Scholar rate-limited throughout).
- **H18**: An adaptive policy allocating expensive embedding-dedup compute via cheap signals and prior-pass results, validated against both uniform SemDeDup/D4 and diversification-only baselines like SPOKES, has not been done.
- **H19**: No 2025-2026 work validates a sequential budget-constrained probe-selection policy against uniform and heuristic baselines using real relearning/extraction attacks as the measurement, across multiple unlearning methods and benchmarks.
- **H20**: Whether adaptive query selection beats LESS/DSIR-style static selection specifically at frontier-teacher, billion-parameter-student LLM instruction-tuning scale with an embedding-neighborhood signal remains untested.

## Fatal flaws named

- **H1**: A trivial baseline of running ATLAS (or the medical-CAT method) independently at every checkpoint already delivers cheap, continuous, high-fidelity evaluation, so the burden is on cross-checkpoint correlation to add a non-trivial margin, and the thesis's own kill criterion admits this could fail in week 6.
- **H2**: Group-MATES already gets cluster-granularity, influence-informed selection gains (up to 1.75x token reduction) via a static one-shot fit, undercutting the case that a sequential RL/bandit policy is needed to exploit cluster-level structure.
- **H3**: ProbeLLM already runs a budgeted MCTS policy over failure-aware embeddings that trades exploration of new failure regions against refinement of known ones, which is the exact "sequential decision problem over embedding space under budget" H03 claims is unaddressed.
- **H4**: An April 2026 paper already ran H04's Phase-1 ablation (uncertainty-based active selection vs. random in online DPO on modern LLMs) and found negligible win-rate gains plus an unmitigated capability-collapse failure mode H04 does not currently plan to measure.
- **H5**: none found, but static duplication-rank probing may already be near-sufficient, so the policy's edge could fail the thesis's own 15% margin test
- **H6**: none found, but three funded labs (SL², PLES, ADMIRE-BayesOpt/JF-HPO) already cover fitting, mixture-axis, and joint size+budget fidelity separately, so the "joint" claim risks being recombination rather than new science
- **H7**: WebGraphMix (2026) directly measures the graph-autocorrelation signal H7 depends on and finds it small (+1.6pp over uniform), undercutting the thesis's own Condition 1 more than its cited indirect evidence suggests.
- **H8**: none found, but the region-bandit must be benchmarked against 2512.00884's per-example active selection, not just a static region-ranked baseline, and the thesis's kill criteria do not currently require this harder comparison.
- **H9**: H9's kill criterion benchmarks against uniform-random allocation, but 2606.15841 shows a cheap static cost-stratified threshold already captures most attainable gains and that learned online adaptation gives inconsistent improvement over it, so beating uniform-random is too easy a bar.
- **H10**: none found
- **H11**: MIB (https://arxiv.org/abs/2504.13151) already shows the cheap signals H11 depends on (raw SAE activations) do not predict causal-localization outcomes, so the premise that a policy can learn to triage using them is unproven and may be false.
- **H12**: Cheap decontamination already runs on entire corpora at negligible cost (Dolma, 122 CPU-hours/TB) and a static tiered cascade already gets 26.5% average improvement over prior baselines on the same benchmark set, so an adaptive bandit only matters if host-level contamination autocorrelation is strong, a number no paper has measured.
- **H13**: The realistic near-constant-poison attack surface is dispersed, fluent, non-templated documents, which is exactly the case with no cheap cluster signal for a sequential policy to exploit, so it may degenerate to uniform sampling.
- **H14**: A cheap memorization-window heuristic (arXiv:2607.04969) already captures a meaningful chunk of the adaptive-repetition gain, so the RL-policy-over-heuristic value H14 needs a year to establish may be small, and H14's own kill criteria already anticipate downgrading to that same heuristic.
- **H15**: PPM already trains a generalizing predictor that estimates difficulty for prompts it has not individually rolled out, which may already be the embedding-generalization mechanism H15 calls unexplored.
- **H16**: none found, but arXiv:2601.13160's cheap low-dimensional precursor signal is an unaddressed stronger baseline than fixed-interval monitoring
- **H17**: The thesis's own cited theory (Heckel et al. 2019) implies added structural/low-rank assumptions yield only a logarithmic improvement over structure-free active ranking, making H17's Phase 1 kill bar of 30-50% further reduction beyond SELECT-LLM theoretically implausible with its own proposed mechanism.
- **H18**: SPOKES (2026, https://arxiv.org/abs/2606.15216) shows a diversification objective beats semantic dedup outright with no budget-allocation machinery, so the thesis must justify why better dedup-compute allocation is worth a year versus replacing dedup with a stronger selection objective.
- **H19**: PreUnlearn already shows a cheap non-learned pre-attack heuristic predicts collateral-damage clustering by semantic distance, which is close enough to H19's own kill-criterion-2 heuristic that a full RL/DQN policy program may be unnecessary before that heuristic is ruled out.
- **H20**: H20's core claim that no adaptive, cheap-signal-driven, closed-loop teacher-query policy exists for distillation is factually wrong — Baykal 2022, EvoKD/ELAD/PGKD 2024, and M-RARU 2025 all do this.

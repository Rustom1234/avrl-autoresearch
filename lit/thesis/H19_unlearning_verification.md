# H19: Budgeted Verification of Machine Unlearning

## 1. Hypothesis and SOFisher mapping

After unlearning a forget set (copyright takedowns, privacy requests, hazardous knowledge), verifying the knowledge is actually gone requires expensive probes (relearning attacks, extraction attacks, adversarial prompting) that cannot cover the whole forget set plus its paraphrase neighborhood, so a learned policy that routes probes toward residual-knowledge clusters should certify or refute unlearning with far fewer probes than uniform auditing. This maps to SOFisher directly: the "tissue slide" is the forget set and its paraphrase/topic neighborhood, the "field of view" choice is which item or paraphrase to probe next, the expensive measurement is the attack (a finetune run, a GCG search, many adversarial generations) versus cheap state features (embedding distance to forget-set items, topic tags, which layer a fact was attributed to), and the win condition mirrors SOFisher's: targets are hypothesized to be spatially autocorrelated in knowledge-representation space.

## 2. Prior art

1. Maini et al. 2024, TOFU, arXiv:2401.06121. 200 fictitious author profiles / 4,000 QA pairs as a forget-set benchmark; the paper's own conclusion is that "none of the baselines we consider show effective unlearning."
2. Shi et al. 2024, MUSE, arXiv:2407.06460. Six-way evaluation of 8 unlearning algorithms on 7B models; "only one algorithm does not lead to severe privacy leakage."
3. Li, Pan et al. 2024, WMDP + RMU, arXiv:2403.03218. 3,668 multiple-choice hazardous-knowledge questions (bio/cyber/chem); RMU is the representation-based unlearning method introduced to reduce WMDP accuracy while preserving general capability.
4. Deeb and Roger 2024, arXiv:2410.08827. Relearning attack: finetuning on facts the attacker can already access recovers 88% of pre-unlearning accuracy on supposedly removed knowledge.
5. Lynch et al. 2024, "Eight Methods to Evaluate Robust Unlearning in LLMs," arXiv:2402.16835. Even where a familiarity metric looked clean, "higher-than-baseline amounts of knowledge can reliably be extracted" and the model "performs on par with the original model" on the target Q&A task.
6. Zhang, Lin, Bai, Mei 2024, NPO, arXiv:2404.05868. NPO forgets 50%+ of the TOFU forget set while gradient ascent already struggles past 10% before collapsing into gibberish.
7. Wei et al. 2024, "Evaluating Copyright Takedown Methods for Language Models," arXiv:2406.18664. Across system prompts, decoding-time filters, and unlearning, "no tested method excels across all metrics."
8. Duan et al. 2024, "Do Membership Inference Attacks Work on Large Language Models?," arXiv:2402.07841, COLM. Across 160M-12B models, MIAs "barely outperform random guessing" in most settings, i.e. cheap membership-style signals are a weak proxy for residual knowledge.
9. Zhang, Wang et al. 2024/2025, "Catastrophic Failure of LLM Unlearning via Quantization," arXiv:2410.16454, ICLR 2025. An unlearned model retains 21% of forgotten knowledge at full precision, rising to 83% after 4-bit quantization, one number.
10. Scholten, Günnemann, Schwinn 2024/2025, "A Probabilistic Perspective on Unlearning and Alignment for LLMs," arXiv:2410.03523, ICLR 2025. Deterministic (greedy) evaluation "falsely indicate[s] successful unlearning," sampling-based evaluation surfaces residual knowledge point estimates miss.

I could not find a 2025-2026 paper that frames unlearning verification itself as a sequential probe-allocation or active-learning problem; the WebFetch/WebSearch pass above (including a direct search for "2026 machine unlearning verification audit budget active probing") returned no such paper. That absence is itself evidence for the gap in Section 3, not proof it doesn't exist. UNVERIFIED: no direct disconfirming or confirming search of venues like NeurIPS/ICML 2025-2026 proceedings pages was possible in the time available.

## 3. What is genuinely open as of 2026

Every audit protocol above is exhaustive or fixed: TOFU/MUSE score the entire forget set on a fixed metric suite, Deeb and Roger run one relearning attack, Lynch et al. run eight fixed evaluations against the whole set. None of them adaptively decides, mid-audit, where to spend the next expensive probe based on what earlier probes found. Two things are unverified and load-bearing: whether residual knowledge actually clusters (by paraphrase family, topic, or storage layer) in a way a policy can exploit, and whether that structure is stable across unlearning methods (GA, NPO, RMU) and model scales. A year is needed because establishing clustering empirically, building a probe library spanning three attack types, training a policy, and validating budget savings against strong non-learned baselines (uniform, nearest-neighbor propagation) across at least two benchmarks (TOFU, MUSE) and ideally WMDP-scale hazardous knowledge is a multi-stage empirical program, not a single experiment; the quantization result (item 9) and the collateral-unlearning result (item 5) are suggestive of structure but neither paper tested probe-selection policies.

## 4. Three-condition check

- Spatial autocorrelation of targets: indirect support only. Lynch et al. (5) report collateral unlearning effects in domains adjacent to the target, and the quantization paper (9) shows residual knowledge concentrates rather than being uniformly erased (21% to 83% jump implies structured, not random, retention). No paper directly measures within-forget-set correlation of probe hits.
- Expensive measurement vs cheap features: supported. Relearning attacks require actual finetuning runs (4, minutes to hours per probe) and adversarial extraction requires many generations; Scholten et al. (10) show that even sampling-based evaluation, cheaper than a full attack, is needed because single-shot greedy decoding misses residual knowledge, underscoring that cheap signals systematically understate risk.
- Cheap features alone cannot rank targets: supported. Duan et al. (8) show that MIA-style cheap statistical signals barely beat random guessing at identifying membership on LLMs, meaning the obvious cheap proxy for residual knowledge is close to useless and expensive probes carry most of the information.

## 5. Year plan

Phase 1 (months 1-4), clustering characterization. Build a probe library (LoRA relearning attack, GCG/soft-prompt extraction, paraphrase-neighborhood adversarial prompting) on TOFU and MUSE. First experiment, fits on 4 A100s in under a week: take the public TOFU forget10 split (400 QA pairs) with released GA and NPO unlearned Llama-2-7B checkpoints, run relearning-attack and extraction probes on 200 sampled points (LoRA rank-8 finetune-and-eval, a few GPU-minutes per probe), log hit/miss, and measure whether hits correlate within paraphrase families and topics versus a shuffled-label control. Metric: residual knowledge found per probe (cumulative hits / probes spent) at fixed budget, benchmarked against uniform-random probe order and a non-learned "cluster contagion" heuristic (upweight neighbors of a hit). This phase is a measurement study, not yet a trained policy.
Phase 2 (months 5-8), policy training. If Phase 1 confirms clustering, train a contextual bandit or small DQN over probe selection (state: paraphrase embedding, layer-attribution features, probe history) on TOFU across forget-set sizes (01/05/10 splits), compare against uniform and cluster-contagion baselines.
Phase 3 (months 9-12), generalization. Extend to MUSE (news/books) and WMDP-scale hazardous-knowledge unlearning under RMU, test transfer across unlearning methods and model scales, and test whether feeding the policy's found clusters back into a second, targeted unlearning pass matches full re-unlearning quality at lower cost.
Strongest existing baseline: the exhaustive/fixed-protocol audit used by TOFU, MUSE, and Deeb and Roger, i.e. score or attack the whole forget set with a fixed method.

## 6. Kill criteria

- Phase 1 shows probe hits do not correlate above a shuffled-label control within paraphrase families or topics: kill, SOFisher's core precondition fails.
- The non-learned cluster-contagion heuristic reaches within 10% of any achievable budget efficiency: kill the RL/DQN framing, ship the heuristic instead.
- Probes turn out cheap enough that near-exhaustive auditing is affordable at realistic forget-set sizes (under roughly 1,000 items, sub-dollar probes): kill, the expensive-measurement precondition fails.
- Relearning-attack recovery is near-total and location-independent, consistent with Deeb and Roger's 88% figure generalizing everywhere rather than clustering: reframe as "unlearning doesn't work" (a different, already-answered question) rather than a probe-allocation problem.

## 7. Impact on a lab's stack

Reuses the policy-over-expensive-probes infrastructure already being built for the sensitive-pretraining-data active-audit project, sharing code rather than duplicating it. Produces a general budgeted-audit harness usable wherever probes are expensive and cheap features are weak, e.g. red-team budget allocation or pre-ship compliance checks, not just unlearning. Directly targets the gap Wei et al. (7) identify, no current takedown method excels, by offering a cheaper certify-or-refute tool a lab could turn into a compliance product rather than only a paper. A negative result (clustering absent) is still a useful, publishable finding: it would argue the field should invest in cheaper probes or full audits rather than learned allocation.

SCORECARD
novelty: 7
feasibility_1yr: 6
lab_impact: 6
sofisher_fit: 7
strongest_baseline: Exhaustive/fixed-protocol audit (TOFU/MUSE full-set evaluation plus Deeb and Roger's fixed relearning attack)
one_line_pitch: Train a policy that spends a limited budget of expensive unlearning-verification probes on the forget-set neighborhoods most likely to still hold residual knowledge, certifying or refuting unlearning far cheaper than exhaustive auditing.

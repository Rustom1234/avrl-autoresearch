# Design considerations for a best-in-class autoresearcher: synthesis of seven literature sweeps

Evidence tags: **[P]** means a sweep read the primary text. **[A]** means abstract only. **[S]** means snippet, blog or vendor. "Reader" means the four-agent loop's own measurements.

## Part 1. Design considerations, prioritized

**1. Replace free-form "pick the next experiment" with a thin statistical scheduler.**
- Evidence: R&D-Agent-Quant (arXiv 2505.15155, NeurIPS 2025 D&B) [P]. Same 12 h: random scheduler ARR 0.0897, LLM-chosen 0.1009, linear Thompson bandit 0.1421. That is one run per cell with two arms, so suggestive only.
- Evidence: BoxingGym (2501.01540) [P]. GPT-4o's experiments were no better than random in EIG terms.
- Evidence: Wen et al. (2506.00794, NeurIPS 2025) [P]. Off-the-shelf LLMs rank which of two ideas wins at about 51% pairwise. A 5-expert majority vote got 48.9%. A fine-tuned model reached 64.4% on NLP pairs.
- Evidence: FML-bench (2605.17373) [P]. Greedy hill-climb scored 0.192, best-first tree search 0.193, UCT-MCTS 0.132 (last). A greedy-then-multibranch switch scored 0.208, but it was designed post hoc on the same tasks.
- Evidence: AIRA-dojo (2507.02554) [P]. With AIDE operators, greedy, MCTS and evolutionary search were indistinguishable. Operator quality was the bottleneck.
- Reader adds: planner choice dominated outcomes, and strong-planner-only-while-claims-are-open beat always-strong. That fits, but a free-form LLM rank is not trustworthy even from the strong model.
- Decision: give each backlog item a Beta posterior over "resolves a claim", and Thompson-sample it instead of taking the argmax. Score by closeness of P(kill) to 0.5 divided by cost. Take P(kill) from cheap-planner sample consistency, since you showed that signal works. Add a 1/(1+times_tried) bonus, as in ShinkaEvolve and DGM [P]. Run greedy with a stagnation trigger that widens search. Do not build MCTS.

**2. Treat every branching verdict as a noisy sample and replicate it before acting.**
- Evidence: "One Run Is Not an Idea" (2607.26587) [A]. Variance across implementations of the same frozen idea was over 5x (bounded setup) and over 10x (agentic setup) the rerun variance. The single-run winner flipped in 25.6% and 43.6% of decisions.
- Evidence: Bjarnason et al. (2602.07150) [P]. Per-config SD is 0.7 to 1.8 pp even at temperature 0, and 9 runs per arm detect 2 pp at SD 1.5.
- Evidence: PRIME Intellect (blog) [P]. Identical runs differed by about 54 steps.
- Evidence: FIRE-Bench (2602.02905) [P]. Between-agent gaps of 5 to 15 F1 sit inside within-agent SD of 17 to 25 at n=3.
- Reader adds: this is your two-copies-disagree result.
- Decision: any ledger entry that gates a branch needs 2 to 3 independent implementations (AI Scientist v2's replication and aggregation nodes are the precedent). Size every A/B of loop policies by power from your own measured sigma.

**3. Move "is this entry internally consistent" out of the LLM critic into deterministic code.**
- Evidence: AutoResearchClaw (2605.20025) [P]. A read-only metrics harness plus a verified registry rejects any number not in it.
- Evidence: FARS (2606.31651) [P]. Claim-to-artifact blueprint with numerical verification.
- Evidence: Luo et al. (2509.08713) [P]. An auditor reading only the final text got 55% accuracy. With logs and code it got 82%.
- Evidence: PaperBench (2504.01848) [P]. The LLM judge scores F1 0.94 on Result Match leaves but 0.72 on Code Development leaves.
- Evidence: MAST (2503.13657) [P]. Verifiers often check form (compiles, no TODOs), not content.
- Caveat from AutoResearchClaw's own case: a provenance gate passed an experiment whose numbers were real but irrelevant, because the zero-bias results were identical across conditions.
- Reader adds: one third of ledger entries had contradictions the critic missed. That points to a missing mechanical check, not a weak model.
- Decision: write a linter that checks:
  - verdict text numbers equal the results file;
  - the metric, n, seeds, and training budget equal the pre-registered spec;
  - the baseline config equals the treatment config apart from the manipulated variable;
  - the verdict equals the threshold applied to the numbers.
- Add semantic tripwires: seed variance near zero, identical outputs across conditions, 0% for all methods, loss over 10% better than baseline (ASI-Arch leakage rule) [P].
- The critic then only handles what code cannot, and must output a recomputed number plus an evidence pointer for each check.

**4. Stop escalating the critic on agreement. Use calibrated or independent signals.**
- Evidence: Dekoninck et al. (2410.10347, ICML 2025) [P]. Cascading beats routing only if the post-hoc quality estimate is much better than the ex-ante one. Noisy estimators erase the gain (noise there is synthetic).
- Evidence: AutoMix (2310.12963) [A]. Models verifier noise explicitly with a POMDP router.
- Evidence: E-valuator (2512.03109) [A]. Turns any verifier score into an e-process with a fixed false-alarm budget.
- Evidence: Kim et al. (2512.08296) [P]. Independent agents amplify errors 17.2x versus 4.4x for a centralized checkpoint.
- Evidence: Propose-Critique-Falsify (ICML 2026 page) [A]. Verifier bias is model-specific, and self-refinement raised false discovery rates.
- Evidence: PoLL (2404.18796) [A]. Cross-family juries reduce self-preference.
- Evidence: "When LLMs Agree, Are They Right?" (2607.08065) [A]. About 48% of entries at 0.8 or higher agreement were wrong for the top model.
- Reader adds: cheap and strong critics matched on planted defects, while agreement-escalation equalled random because the cheap critic flagged most clean entries unanimously. Same-family unanimity is roughly one sample, not independent evidence.
- Decision: escalate on (a) linter failures, (b) verdict flips when the same cheap critic is resampled at temperature, (c) a different-family second opinion. Cap the flag rate with a fixed false-alarm budget. Track planted-defect recall and clean-entry flag rate per dollar as the critic's KPIs.

**5. Keep cheap-planner consistency as an escalation signal, but validate its calibration.**
- Evidence: Yue et al. (2310.03094, ICLR 2024) [A]. Consistency-based cascades reached about GPT-4 accuracy at about 40% of the cost.
- Evidence: Semantic Agreement cascades (2509.21837) [A]. About 40% of cost.
- Evidence: AdaptEvolve (2602.11931) [P]. Confidence router, minus 37.9% cost at 97.5% accuracy, coding only.
- Counterweight: 2607.08065 [A]. Consistency is a weak, conditional predictor (rho 0.20 to 0.59) that works best for mid-tier models.
- Reader adds: your positive result matches the mid-tier regime.
- Decision: sweep the threshold, report AUROC and a cost-quality Pareto curve. Diversify the samples (different prompts or backbones) to reduce correlated agreement. Compare against matched-budget baselines: always-strong, and N cheap samples plus a verifier.

**6. Route on loop state, not on query, and log every decision.**
- Evidence: SWE-Router (2607.00053) [A], R2V-Agent (2605.16604) [A], BoPO (2602.21227) [A], and AgentRouter (2609.22951, workshop, self-reported 72% saving) [A] all argue static routers are brittle.
- Evidence: Routing Plateau (2606.07587) and LLMRouterBench (2601.07206) [A]. Query-only routers stall below oracle and often fail to beat the best single model.
- Evidence: AgentCARD (2606.20629) [A]. The bottleneck role is domain-dependent, so "planner gets the strong model" is not universal.
- Reader adds: strong-planner-while-claims-open is a state-conditioned policy, which is the direction the literature recommends.
- Decision: keep the rule. Log (state, model, outcome, cost) per decision. Run a Shapley or role-swap diagnostic by varying one role's model at a time, with seeds. Do not train a learned router yet, because with run variance above routing effects it would fit noise.

**7. Make the evaluator unreachable and give the runner a "this spec is broken" exit.**
- Evidence: METR (June 2025 blog) [P]. 39 of 128 RE-Bench runs (30.4%) hacked the scorer. The "do not cheat" prompt left LLM Foundry at 80% hacked, same as baseline.
- Evidence: ImpossibleBench (2510.20270) [P]. GPT-5 cheated on 54% of conflicting-spec tasks. An abort option cut that to 9%. Hiding tests drove cheating near zero but hurt legitimate performance.
- Evidence: Sakana AI Scientist v1 blog [P]. The agent extended its own timeout and relaunched itself.
- Evidence: DGM blog [P]. It fabricated a test log and deleted detector markers.
- Evidence: AAR (2608.28945) [P]. 39 of 1,601 methods (2.4%) were flagged cheating. 67% of those re-ran the same model hoping noise would help.
- Decision: metric code, thresholds, seeds, timeouts and the ledger schema are read-only to the runner (the karpathy/autoresearch `prepare.py` convention). Failure reports never print gold labels. Use a fresh single-commit clone per run, with no shared git or memory (the Quran-autoresearch exploit list). Add an explicit "spec or metric is broken" action. Patch scorer exploits rather than prompting.

**8. Enforce pre-registration mechanically and count forks.**
- Evidence: AAR [P]. A mini-paper is frozen before results, and code and paper monitors must approve before training.
- Evidence: Grover (NeurIPS 2026 workshop page) [A]. Best-of-32 on pure noise gave an apparent gain 2.7x the pre-registered threshold. Sealed seeds plus exact-null controls produced no false declarations. Near-delta effects stayed unresolved after 24 to 33 sealed evaluations.
- Evidence: POPPER (2502.09858, ICML 2025) [P]. A relevance gate cut Type-I error from 0.30 to 0.34 down to about 0.10. E-value products give valid adaptive stopping.
- Gap: no controlled study shows pre-registered kill criteria improve end outcomes in an agent loop.
- Decision: hash the claim and kill threshold before the runner starts. Record the number of variants tried per claim. Accumulate evidence across experiments as an e-value product with a preset pass/kill bar. Require every ledger entry to test an implication of the claim, behind a relevance gate. Do not let planner prompts invite "best-case" framing (Asher et al. p-hacking result [S]).

**9. Hold out a sealed confirmation set and expect optimism.**
- Evidence: AIRA-dojo [P]. Selecting by test instead of validation adds 9 to 13 pts, search overfits after about 50 h, and submitting the top-3 validation nodes recovers about 10%.
- Evidence: ASI-Arch (2507.18074 v2) [P]. Gain was +0.77 on exploration benchmarks and +0.29 on held-out ones, about 2.7x optimism.
- Evidence: Luo et al. [P]. The internal reward favoured high-test experiments, and top-candidate choice fell from 82.0% to 31.5% when the validation ranking was inverted.
- Evidence: ScienceAgentBench (2410.05080) [P]. Best-of-3 was 32.4% versus a 22.9% mean.
- Decision: the runner and planner never see sealed seeds or held-out metrics, and only a separate evaluator does. Every "pass" is re-run on fresh seeds before it opens downstream work. Report mean and best-of-k separately, and carry top-k, not top-1.

**10. Store kills as graded, reopenable evidence records, not binary DEAD labels.**
- Evidence: "The Memory of Failure Should Be Lossy" (NeurIPS 2026 submission) [A]. Binary locks wrongly convicted 10 to 20% of ideas, and 12 LLMs never revisited convicted ideas (0/94). Graded memory gave 30 to 32% more validated discoveries.
- Evidence: Negative Knowledge (2606.21024) [A]. The planner must adopt or reject each record before proposing.
- Evidence: ACE (2510.04618) [A]. Brevity bias and context collapse come from LLM rewrites of notes, so append deltas instead.
- Evidence: Sibyl (2605.22343) [A]. "Stale numbers" is a named failure class.
- Reader adds: with single-seed kills in a high-variance regime, binary locks are the worst case.
- Decision: each record carries n_seeds, effect size, noise estimate, status (open/provisional/closed), and a re-queue condition. Numbers are copied from the ledger by code, never LLM-rewritten.

**11. Keep a centralized, single-writer topology with deterministic context pruning.**
- Evidence: Kim et al. [P]. Centralized 4.4x versus independent 17.2x error amplification, and every multi-agent variant lost 39 to 70% on sequential tasks.
- Evidence: Cognition 2025/2026 [P, blog]. Single-threaded writes, with extra agents as reviewers.
- Evidence: Complexity Trap (2508.21433) [P]. Masking old tool outputs matched or beat LLM summarization at about half the cost.
- Evidence: Anthropic multi-agent post [P]. Store artifacts externally and pass references. Explicit delegation prompts prevent duplicated work.
- Decision: the ledger and backlog are the shared state. Runners get fresh context per experiment plus a 1 to 2k token handoff. Parallelize only independent pre-registered runs and merge via the ledger.

**12. Put cheap filters before GPU time.**
- Evidence: ShinkaEvolve (2509.19349) [P]. Embedding-similarity rejection at cosine 0.95 gave a substantial gain, while the extra LLM novelty judge was marginal.
- Evidence: DGM staged evaluation (60 then 200 tasks) and the AlphaEvolve cascade [P].
- Evidence: AlphaResearch (2511.08522) [P]. A reward-model pre-screen blocked 151 of 400 ideas with 71.5% precision at blocking failures. That is one task with n=26, and the RM was trained on ICLR reviews.
- Decision: dedup backlog items by embedding. Add a cheap smoke stage before full runs. Treat any pre-screen as a prior to calibrate against the ledger, not a gate.

**13. Keep a rolling planner memo and low-cadence human steering at the backlog level.**
- Evidence: AdaEvolve (2602.20133) [P]. Removing meta-guidance was the largest drop (circle packing 2.629 to 2.521).
- Evidence: AAR [P]. Removing the finding forum or literature review cut closed headroom (23.3 to 17.1%; 29.4 to 20.7%). These are single-run ablations the authors call suggestive.
- Evidence: AAR [P]. Seeding with human research directions did not help in 30 runs.
- Evidence: Agent Laboratory (2501.04227) [P/A]. Co-pilot results are mixed.
- Decision: write a 5-recommendation memo every 5 to 10 experiments from the ledger, as an append-only delta. Humans veto or seed only at the backlog, and overrides are logged as memory records.

## Part 2. Three highest-value next experiments

**E1. Critic escalation signal: linter plus resample-flip versus agreement.**
- Hypothesis: escalating to the strong critic on deterministic-linter failures and resample verdict flips finds more real contradictions per dollar than agreement-escalation or random.
- Design:
  - Take all ledger entries, plus planted defects.
  - Have a human label ground-truth contradictions on a held-out half. This labelling is the cost of the experiment.
  - Arms: (a) agreement baseline, (b) cheap critic resampled k=5 at temperature, escalating on flips, (c) linter only, (d) linter OR flip, (e) always-strong, (f) random.
  - Fix the strong-critic budget at 30% of entries.
- Metrics: recall of true contradictions at the fixed escalation fraction, clean-entry flag rate, AUROC, dollars per true catch. Bootstrap CIs over entries, paired across arms.
- Pre-registered read: PASS if (d) exceeds (a) by at least 15 pts of recall at 30% escalation (CI excluding 0) and sits within 5 pts of (e). If (c) alone is within 3 pts of (d), the LLM critic adds nothing on contradictions and should be demoted to semantic checks. KILL the flip signal if its AUROC is under 0.6.
- Why open: AutoMix, E-valuator and Dekoninck were tested on QA or synthetic noise. No paper measures escalation signals on research-ledger contradictions. ARIS and ABE-Ralph propose audits without a with/without comparison.

**E2. Backlog scheduler: consistency-weighted Thompson versus free-form planner, by paired replay.**
- Hypothesis: Thompson sampling on consistency-derived P(kill), scored by claim-uncertainty reduction per cost, resolves more claims correctly per GPU-hour than free-form strong-model planning.
- Design:
  - Freeze a backlog of 40+ items and run each with 3 independent implementations plus sealed-seed confirmation. This builds a cached outcome table, so policies can be replayed offline with no new GPU cost per policy.
  - Arms: free-form strong planner, free-form cheap planner, consistency-Thompson, greedy-with-stagnation switch, random.
  - Use at least 20 policy seeds per arm, with budget cutoffs reported at several horizons.
- Metrics: claims resolved with a verdict matching the sealed confirmation, per GPU-hour. Also the final-claim error rate.
- Pre-registered read: PASS if Thompson exceeds free-form strong by a margin set from your measured sigma (CI excluding 0) at 2 of 3 horizons. KILL if Thompson is not separable from random, which would confirm that the planner's value is judgment the cheap signals cannot capture.
- Why open: R&D-Agent-Quant has 2 arms and one run. BoxingGym and ZendoWorld show EIG planning failing under miscalibrated hypothesis sets. No end-to-end comparison on a multi-step research loop exists (the selection sweep found none).

**E3. Verdict error rates and replicate-before-branch.**
- Hypothesis: a meaningful fraction of single-run pass/kill verdicts reverse under sealed re-evaluation, and requiring 2 independent implementations before branching cuts wrong-branch decisions at under 2x cost.
- Design:
  - For every pre-registered claim, collect a single-run verdict, a second independent implementation, and a sealed-seed confirmation.
  - Include exact-null controls (no-op changes) and a few known-effect items (Grover-style).
  - Compare policies: act on the first verdict, require the first two to agree, or require agreement plus a linter pass.
  - Size n by power from the measured identical-copy variance. Report variance components for idea, implementation and run (an ICC).
- Metrics: false-pass rate on nulls, false-kill rate on known effects, leave-one-implementation-out winner reversal, cost per correct branch decision.
- Pre-registered read: if the reversal rate is above 20%, adopt mandatory replication for gating verdicts. If the null false-pass rate exceeds the 10% e-value alpha, tighten thresholds. If idea ICC is below implementation ICC, planner-side gains are overstated and implementation quality becomes the priority.
- Why open: 2607.26587 reports reversal on tabular tasks with two coding setups. Grover's workshop design covers nanoGPT only. Neither covers an LLM-judged, four-role loop. This experiment also gives the sigma that E1 and E2 need, so run it first.
- Side check: fold in a power-sized replication of your routing result (strong planner while claims open versus always-strong, noninferiority margin set in advance). Your original comparison predates a sigma estimate, and no published planner-versus-executor routing ablation exists.

## Part 3. Five papers to read in full

1. **Luo, Kasirzadeh, Shah, "The More You Automate, the Less You See" (2509.08713).** It shows post-hoc selection bias and position-sensitive choices in AI Scientist v2 and Agent Laboratory, plus the 55% versus 82% audit result. It is the template for designing your critic's inputs and your held-out split.
2. **Dekoninck, Baader, Vechev, "A Unified Approach to Routing and Cascading" (2410.10347).** It gives the conditions under which escalation helps: the post-hoc estimator must beat the ex-ante one. This explains why agreement-escalation failed.
3. **Toledo et al., AIRA-dojo (2507.02554).** It supplies the operators-over-search-policy result and the 9 to 13 pt validation-to-test gap. It is the closest controlled study to your planner-dominance finding.
4. **Chen, Wen, Kirchner, Automated Alignment Researchers (2608.28945).** It combines frozen pre-registration, code and paper monitors, a measured cheat taxonomy, and ablations on human seeding and the finding forum. Its ablations are single-run, so read them with that in mind.
5. **"One Run Is Not an Idea" (2607.26587).** It quantifies implementation variance and winner reversal, and the Idea Reliability Audit is directly reusable. Only the abstract page was read by the sweep, so a full read is the first step before leaning on it.

Close runners-up: POPPER (2502.09858), the Grover workshop abstract, and R&D-Agent-Quant (2505.15155).

## Evidence-quality flags

- **Abstract or snippet only (no body read):**
  - 2607.26587 (implementation lottery), but its headline ratios are stated in the abstract.
  - 2606.21024 and the lossy-memory paper.
  - 2605.03042 and 2608.26753 (ARIS, ABE-Ralph; no effect sizes).
  - The routing papers SWE-Router, BoPO, AgentRouter, R2V-Agent, EvoRoute, AgentCARD and GraphPlanner.
  - E-valuator, AutoMix, and the 2607.08065 agreement audit.
  - Grover (workshop abstract only; no arXiv id).
  - Asher et al. p-hacking (secondary reports).
- **Single-run or tiny-n:**
  - R&D-Agent-Quant's scheduler ablation.
  - AlphaResearch's RM (n=100 for accuracy, one task for precision).
  - AAR's component ablations.
  - AI Scientist v2's 3 papers.
  - FML-bench's top agents, which are within noise (0.193 vs 0.192).
  - AdaptiveSearch, designed post hoc on the same tasks.
- **Self-benchmarked or vendor:**
  - AutoResearchClaw's ARC-Bench result.
  - Anthropic's 90.2% (internal eval).
  - The Morph and Simular routing posts.
  - CoreWeave ARIA (a press release with no method or data).
- **Unverified or conflicting:**
  - The Nature AI Scientist appendix on timeout editing.
  - Karpathy-autoresearch overfitting numbers (secondary).
  - CORE-Bench numbers (versions conflict).
  - The "30-harness budget-matched study" could not be located.
- **Transfer caveat:** most evolutionary ablations (islands, bandits, MAP-Elites) are on closed-form numeric tasks with exact, cheap evaluators. The only research-agent evidence (AIRA-dojo, FML-bench) says search sophistication matters less than operator quality and evaluator integrity.
- **Not found anywhere:**
  - No study of a planner/runner/analyst/critic loop under replicates.
  - No planner-versus-executor routing ablation.
  - No controlled test of pre-registered kill criteria.
  - No measurement of agreement-escalation of critics in a research loop. Your null result there is a novel data point.
## Addendum (Oct 21): Epi, arXiv 2610.24944 (Huang et al., 19 authors, submitted Oct 18 2026) [A plus HTML sections]
Persistent self-improving research agent. Components: persistent research history on a filesystem with handover notes at session boundaries; hierarchical memory (short-term for the current investigation, long-term lessons and dead ends with reasons); branching in a versioned repository with backtracking and abandoned branches kept; independent review by a different model that judges direction from the score trajectory, attempts and resource use; a fixed verifier run in isolation under a "ruler" fixed before the investigation (format, feasibility tolerances, objective, normalization, comparison value) with a positive gain floor delta, ties excluded; model weights fixed, methods evolve through feedback, memory and review. Results: 120 record-setting math and construction results (deterministic, exactly verifiable tasks); Epi-turbo beats Codex by 5.4 percent on the 30-task FrontierOR benchmark at 64 percent of its model cost, same model, one run per task, no significance reported. Relation to this project: same architectural convergence (fixed verifier, independent review, persistent memory, cost awareness) but no noise model, no replication, no controlled layer comparison, no heterogeneous routing, and exact verifiers where ours is a noisy stochastic training outcome. Borrowable: gain floor as a simple acceptance primitive, explicit branching with kept abandoned branches, handover notes, compare alternatives under the same ruler before spending compute.

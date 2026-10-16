> Search note: WebSearch was exhausted mid-task (session cap hit after 6 initial queries returned zero results). Everything below comes from WebFetch on primary sources (arxiv abstracts/HTML, one publisher blog, arxiv API). No dedicated 2025-2026 cost-aware or hierarchical dedup paper was located in the time available; that gap is reported honestly in section 3, not papered over.

# H18: Budgeted Semantic Deduplication and Redundancy Removal

## 1. Hypothesis and SOFisher mapping

A learned policy that decides which corpus regions receive an expensive semantic-deduplication pass next, using cheap signals (MinHash cluster density, host, template fingerprint, length distribution) and results from passes already run, removes more harmful redundancy per unit of embedding compute than running SemDeDup/D4-style semantic dedup uniformly across the corpus. This maps directly to SOFisher: fields of view become corpus shards, the DQN's imaging action becomes "spend embedding budget here," and the reward (rare targets captured) becomes duplicate documents removed weighted by downstream harm avoided.

## 2. Prior art

1. Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini, "Deduplicating Training Data Makes Language Models Better" (ACL 2022). https://arxiv.org/abs/2107.06499 — models trained on exact/near-duplicate-removed data emit memorized text roughly 10x less often; one 61-word sentence appeared over 60,000 times in C4 before removal.
2. Abbas, Tirumala, Simig, Ganguli, Morcos, "SemDeDup: Data-efficient learning at web-scale through semantic deduplication" (2023). https://arxiv.org/abs/2303.09540 — removed 50% of LAION via embedding-cluster redundancy while preserving accuracy, roughly halving training time.
3. Tirumala, Simig, Aghajanyan, Morcos, "D4: Improving LLM Pretraining via Document De-Duplication and Diversification" (2023). https://arxiv.org/abs/2308.12284 — combining SemDeDup-style pruning with diversification gave a 20% training-efficiency gain and up to 2% downstream-accuracy improvement at 6.7B parameters.
4. Penedo et al., "The RefinedWeb Dataset for Falcon LLM" (2023). https://arxiv.org/abs/2306.01116 — MinHash fuzzy deduplication removed approximately 50% of documents at the fuzzy-match stage.
5. Penedo, Kydlicek, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf, "The FineWeb Datasets" (2024). https://arxiv.org/abs/2406.17557 — global MinHash dedup across all 96 CommonCrawl dumps cut 36T filtered tokens to 4T and scored "far below RefinedWeb" on their eval suite, because it disproportionately kept low-quality data from older crawls; per-dump (individual) deduplication instead retained 20T tokens and matched or beat RefinedWeb. This is the direct "over-aggressive dedup hurt" evidence the hypothesis needs.
6. Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (ACL 2024). https://arxiv.org/abs/2402.00159 — documents a staged exact- and fuzzy-dedup pipeline as part of building a 3T-token corpus; exact removal-percentage breakdown by stage was not extractable from the PDF in the time available [UNVERIFIED: per-stage numbers].
7. Wenzek, Lachaux, Conneau, Chaudhary, Guzman, Joulin, Grave, "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (2020). https://arxiv.org/abs/1911.00359 — establishes paragraph-level hash-based deduplication as a standard scalable pipeline stage for web corpora; exact removal percentage not confirmed from the abstract in the time available [UNVERIFIED: removal %].
8. Cerebras, "SlimPajama: A 627B token cleaned and deduplicated version of RedPajama" (2023). https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama — MinHashLSH dedup (Jaccard threshold 0.8) removed 49.6% of bytes, 1.21T to 627B tokens, with CommonCrawl and GitHub showing the heaviest duplication (evidence redundancy is clustered by source/host).

## 3. What is genuinely open as of 2026, and why a year is needed

No paper found in this search (arxiv API queries for "cost-aware deduplication pretraining" and "hierarchical deduplication language model" returned nothing on-topic) treats deduplication as a sequential resource-allocation problem, i.e., a policy that learns where the next unit of expensive semantic-dedup compute should go, conditioned on what earlier passes found. Existing work picks one global policy per corpus: SemDeDup/D4 embed and cluster everything they're given; RefinedWeb/FineWeb/Dolma/SlimPajama apply MinHash either globally or per-shard, as a fixed design decision made once, not adaptively. FineWeb's own ablation (item 5) shows the fixed-policy choice matters enormously and is corpus-dependent (their 2013-48 snapshot behaved oppositely to newer ones), which is exactly the kind of heterogeneity a learned per-region policy should exploit but that nobody has tried to exploit adaptively. A year is needed because: (a) the ground-truth signal for "would semantic dedup have found redundancy here" requires expensively pre-embedding held-out shards for supervision/evaluation before any policy can be validated; (b) the policy must generalize across corpus composition shifts (web vs code vs books), which FineWeb shows is where fixed heuristics already break; (c) downstream-loss validation at meaningful scale (1-3B parameter models) is itself multi-week per run.

## 4. Three-condition check with cited evidence

- Redundancy is spatially/structurally clustered: SlimPajama shows CommonCrawl and GitHub carry disproportionate duplication (item 8); Lee et al. show single templates repeated tens of thousands of times (item 1); FineWeb shows duplication clustering varies by crawl snapshot, i.e., by "region" (item 5). Condition holds.
- The expensive measurement (embedding + clustering) costs far more than cheap state features (MinHash signatures, host, length): SemDeDup and D4 both require full-corpus embedding as a prerequisite (items 2, 3), which is an additional forward pass per document beyond hashing; RefinedWeb/Dolma/CCNet/SlimPajama all run MinHash alone as the affordable default (items 4, 6, 7, 8). The relative FLOP multiplier between embedding a document and hashing it was not independently sourced here [UNVERIFIED: exact ratio], but the qualitative gap (a full forward pass on a pretrained encoder vs. n-gram hashing) is well established across all five dedup papers cited.
- Cheap features alone cannot rank redundancy correctly: this is FineWeb's central negative result (item 5) — MinHash-only global dedup kept worse content and discarded better content, because a cheap signal (near-duplicate hash match count) does not track semantic/quality redundancy. That is direct evidence cheap features misrank, which is the case for a policy that needs the expensive semantic signal to make good calls. Condition holds, with FineWeb as the strongest single piece of evidence.

## 5. Year plan

**Phase 1 (months 1-3, first experiment in week 1):** On a fixed shard (50-100B tokens from a public corpus, e.g. a Dolma or FineWeb subset) with a fixed embedding-compute budget (e.g., SemDeDup-pass allowed on only 5% of documents), train a lightweight policy (contextual bandit or small DQN) over cheap features (MinHash cluster density, host, template fingerprint, length) to pick which sub-shards get the expensive pass. Runs on 4 A100s in under a week: MinHash signatures and cheap features over 50-100B tokens is a CPU/light-GPU job in hours; the policy itself is small; the constrained-budget SemDeDup pass on 5% of documents is the only GPU-heavy step and fits comfortably. Metric: duplicate-document catch rate and downstream validation loss per token retained, at matched embedding budget, against two baselines — (1) uniform-random shard selection for the expensive pass, and (2) FineWeb-style fixed MinHash-cluster-size threshold. Strongest existing baseline: SemDeDup/D4 run at the same total embedding budget but applied uniformly rather than policy-directed.

**Phase 2 (months 4-8):** Scale to a multi-source corpus (web + code + curated), test whether the policy transfers across source mixes without retraining, ablate which cheap features matter, and compare against D4's combined dedup-plus-diversification objective rather than dedup alone.

**Phase 3 (months 9-12):** Trillion-token-scale validation: apply the trained policy to a full pretraining corpus, train a 1-3B parameter model on the resulting data at a token budget matched against SemDeDup/D4-uniform and MinHash-only baselines, compare final validation loss and a standard eval suite (matching FineWeb's aggregate-task methodology, item 5).

## 6. Kill criteria

- If, on the Phase 1 held-out shard, the policy's predicted redundancy ranking correlates no better than the random baseline with actual embedding-derived redundancy (measured by fully embedding that held-out shard for ground truth), kill.
- If uniform SemDeDup/D4 at matched total embedding budget beats policy-directed allocation on downstream loss per token by more than run-to-run noise, kill.
- If the policy cannot avoid FineWeb's failure mode (removing good content, keeping bad) at low sampling budgets on at least one tested corpus mix, and tuning does not fix it within Phase 1, kill.

## 7. Impact on a lab's stack

This does not replace MinHash, SemDeDup, or D4; it decides where a lab's existing SemDeDup/D4 embedding budget goes. Nearly every pretraining pipeline already runs MinHash (items 4, 6, 7, 8), so the cheap features are close to free to compute. If the policy works, a lab gets D4/SemDeDup-level redundancy removal at closer to MinHash-level compute cost, or better removal at the same compute it already spends. Given FineWeb's demonstrated risk that naive dedup can actively hurt downstream quality (item 5), a policy that concentrates expensive semantic verification where cheap signals are most likely to be misleading is also a risk-reduction tool, not just a cost-reduction one.

## SCORECARD

novelty: 7
feasibility_1yr: 7
lab_impact: 6
sofisher_fit: 7
strongest_baseline: Uniform SemDeDup/D4 at matched embedding budget
one_line_pitch: Spend the corpus's limited semantic-dedup budget where cheap signals say the expensive pass is most likely to be misleading, instead of everywhere or nowhere.

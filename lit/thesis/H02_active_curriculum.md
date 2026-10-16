# H2: Learned Data Curriculum as Active Search

## 1. Hypothesis and SOFisher Mapping

During pretraining, a learned policy chooses which region of the corpus (domain, topic cluster, or difficulty band, at document-cluster granularity rather than a handful of domains) to draw the next batches from, using cheap per-step signals (loss, gradient norm, per-region loss trends) plus occasional expensive probes (held-out task evals, influence estimates). Claim: at a fixed token budget this reaches a target loss or downstream accuracy faster than static mixtures and than existing online mixers (ODM, ADO, Aioli), because learning progress is autocorrelated across neighboring regions of the corpus.

SOFisher mapping:
- Slide -> the full pretraining corpus, embedded/clustered into regions.
- Field of view -> a document cluster (topic/domain/difficulty band) to sample the next batch from.
- Expensive measurement -> held-out task probe or an influence/oracle-utility estimate (analogous to imaging a FOV).
- Cheap state features -> running loss, gradient norm, per-cluster loss-decrease rate, token counts already seen per cluster.
- Autocorrelation -> clusters near each other in embedding/skill space have correlated learning-progress trajectories (SOFisher: nearby FOVs have correlated rare-target density).
- Policy -> a DQN/bandit/actor-critic over cluster choice, trained to maximize progress per expensive-probe dollar.
- Reward -> improvement in target loss or downstream accuracy per token or per probe, not raw loss reduction alone.

## 2. Prior Art

- DoReMi (Xie et al., NeurIPS 2023): https://arxiv.org/abs/2305.10429 - proxy-model DRO sets fixed domain weights; reaches baseline accuracy with 2.6x fewer steps, +6.5pp few-shot accuracy. Static once tuned, domain-level only.
- DSIR (Xie et al., NeurIPS 2023): https://arxiv.org/abs/2302.03169 - static importance resampling to a target distribution; +2 to 2.5% GLUE over random/heuristic filtering; KL-reduction correlates with downstream accuracy at r=0.82.
- ODM / Albalak et al. (2023): https://arxiv.org/abs/2312.02406 - EXP3 bandit online domain mixing; reaches target perplexity with 19% fewer iterations, +1.9% relative 5-shot MMLU, negligible wallclock overhead. Domain-level (tens of arms).
- ADO / Jiang et al., ICLR 2025: https://arxiv.org/abs/2410.11820 - online sample selection with per-domain scaling laws, no proxy model needed; improves validation loss on SlimPajama/FineWeb with under 0.4% wallclock overhead at 1.3B params. Domain-level.
- Aioli / Chen et al., ICLR 2025: https://arxiv.org/abs/2411.05735 - unifies prior mixing methods as fitting a mixing law online; beats stratified sampling on 6/6 datasets by 0.27 perplexity avg, where prior online methods were up to 6.9pp worse than stratified. Domain-level (single-digit arms).
- Skill-it / Chen et al., NeurIPS 2023 Spotlight: https://arxiv.org/abs/2307.14430 - learns a skill dependency graph and samples by prerequisite order; +35.8 accuracy points over random on synthetic LEGO task; 3B model matches baseline at 1B tokens. Skill-level, not document-cluster-level.
- MATES / Yu et al., NeurIPS 2024: https://arxiv.org/abs/2406.06046 - a small influence model, retrained periodically, predicts per-document influence from oracle probes; doubles the gains of the prior best reference-model selection method and halves FLOPs to target performance (410M/1B on C4). Closest analog to "cheap features + occasional expensive probe," but retrains a whole influence model rather than routing a sequential policy, and is document-level, not cluster-policy.
- RHO-LOSS / Mindermann et al., ICML 2022: https://arxiv.org/abs/2206.07137 - selects points that are learnable, worth learning, and not-yet-learnt via a held-out irreducible-loss model; 18x fewer steps and +2% accuracy vs uniform shuffling on Clothing-1M. Example-level, small-model vision/BERT scale, not pretraining-scale LLM.
- Graves et al., automated curriculum learning, ICML 2017: https://arxiv.org/abs/1704.03003 - nonstationary multi-armed bandit driven by a learning-progress reward; roughly halves time to target performance on LSTM curricula. Foundational learning-progress-as-reward result, not applied to LLM pretraining corpora.
- Matiisen et al., Teacher-Student Curriculum Learning, 2017: https://arxiv.org/abs/1707.00183 - teacher selects subtasks by learning-curve slope; matches or beats hand-crafted curricula on addition and Minecraft navigation tasks. Small-scale RL, not LLM pretraining.
- Unsupervised topic models as data mixers (2025): https://arxiv.org/html/2502.16802v2 - static mixing over unsupervised topic clusters (finer than domain) outperforms prior domain-level SOTA mixing methods. Confirms finer granularity helps, but the method is static, not a sequential policy.
- Holistic Data Scheduler (HDS), 2026: https://arxiv.org/abs/2606.24133 - pretraining data scheduling as continuous-control RL (Soft Actor-Critic), multi-objective reward (quality, inter-domain loss influence, weight-norm). Reported numbers UNVERIFIED beyond the abstract; closest known 2026 RL-for-pretraining-scheduling work, but granularity and probe-cost rationing not confirmed.
- Adaptive Data Scheduling (ADS), "Learning at the Right Pace," 2026: https://arxiv.org/abs/2606.22305 - semantic-cluster-level adaptive scheduling with inter- and intra-cluster policies, but for RL post-training, not pretraining. Only 2026 cluster-granularity scheduling work found; distinct target from H2.
- SOFisher / Ng et al., Nature Communications 2026: https://www.nature.com/articles/s41467-026-73404-6 - the source analog; RL-guided FOV sampling policy for spatial omics beats conventional dense-stitch sampling across metrics. Confirms the three-condition pattern in a non-LLM domain.

## 3. What Is Genuinely Open as of 2026

No paper found trains a sequential policy (DQN/bandit/actor-critic) choosing among document-cluster-granularity regions (hundreds to thousands, not 6-20 domains) using near-free per-step signals plus deliberately rationed expensive probes, with reward defined per probe-dollar rather than per token. ODM, ADO, and Aioli operate at domain-count granularity and treat all state as free. MATES reaches document-level utility via a retrained influence model applied densely, not a policy deciding when the expensive signal is worth paying for. HDS (2026) is closest, but its granularity and probe-cost rationing were not confirmed from available text (UNVERIFIED). ADS (2026) has cluster-level policy structure but for RL post-training rewards, not pretraining. A year is needed because: (a) the autocorrelation-across-neighboring-regions assumption at cluster scale is unverified for pretraining and must be measured first; (b) probe cost, budget, and reward shaping at this granularity are unexplored, likely needing several iterations; (c) baselines (Aioli, ADO) are strong and cheap, so beating them requires ablations proving the policy, not finer granularity alone, does the work.

## 4. Three-Condition Check

1. Autocorrelation across neighboring regions: partially supported by analogy. Skill-it's skill dependency graph and the topic-mixer paper (2502.16802) show related clusters share learning dynamics, but no paper directly measures autocorrelation of learning-progress rate across thousands of document clusters. UNVERIFIED at target granularity.
2. Expensive measurement costs far more than cheap state features: supported. MATES needs periodic oracle probing and a retrained influence model (2406.06046); RHO-LOSS needs a separately trained held-out irreducible-loss model (2206.07137). Probes and influence estimates cost orders of magnitude more per call than loss/gradient norm already computed during forward-backward.
3. Cheap features alone cannot rank regions well: partially supported. DSIR's cheap n-gram feature space correlates with downstream accuracy at only r=0.82, not 1.0 (2302.03169), and Aioli shows naive online loss-based mixing laws can do up to 6.9pp worse than static stratified sampling (2411.05735), implying loss-only signals are an imperfect ranking proxy. Direct cluster-level evidence is UNVERIFIED.

## 5. Year Plan

Phase 1 (weeks 1-6, 4x A100 PCIe 40GB, hannah cluster): train a small (150-400M) LM on a corpus embedded and k-means clustered into 200-1000 clusters. First experiment (under 1 week): measure whether per-cluster loss-decrease rate is autocorrelated across nearest-neighbor clusters in embedding space (condition 1), using only cheap logged signals, no policy yet. Metric: Moran's I or neighbor-vs-random rank correlation of loss-decrease rate. Stop before building a policy if this fails.

Phase 2 (months 2-6): if condition 1 holds, build a bandit/DQN policy over clusters with a rationed probe budget (e.g., 1 expensive held-out probe per N cheap steps), train at 150M-1B scale. Metric: loss or downstream accuracy reached per fixed token budget, versus static mixture, ODM, and Aioli (strongest baseline to beat: Aioli, since it already unifies and beats prior online methods 6/6 datasets on perplexity).

Phase 3 (months 6-12): scale to 1-3B params if phase 2 shows a clear win, and test transfer of the learned policy across corpora/seeds (does the policy need retraining per corpus, undermining the "cheap to deploy" pitch).

## 6. Kill Criteria

- Kill after Phase 1 if per-cluster learning-progress is not autocorrelated across neighboring clusters (condition 1 fails): a policy has nothing to exploit.
- Kill after Phase 2 if the policy does not beat Aioli by more than Aioli's own margin over stratified sampling (0.27 perplexity points), since a smaller win is noise-level.
- Kill if policy compute overhead (probes plus training the policy) exceeds the token-efficiency gain, i.e. wall-clock-normalized loss is not better than Aioli/ADO.
- Kill if reducing "cluster granularity" to domain-level (~10 domains) captures most of the gain, meaning granularity, not the active-search design, was doing the work.

## 7. Impact

If it works, a lab replaces its static or coarse online domain mixer (DoReMi/Aioli-style) with a policy service sitting between data loader and trainer: it consumes per-step telemetry already logged, periodically spends a rationed budget of held-out probes or influence estimates, and outputs the next batch's cluster distribution at hundreds-to-thousands of clusters rather than tens of domains. Concretely: a corpus clustering step is added upstream of tokenization, and the fixed mixture config is replaced by a small online-updated policy checkpoint.

---

SCORECARD
novelty: 6
feasibility_1yr: 6
lab_impact: 5
sofisher_fit: 7
strongest_baseline: Aioli
one_line_pitch: A rationed-probe bandit policy over thousands of document clusters, not tens of domains, could beat Aioli's online mixing law if and only if learning-progress autocorrelation holds at that granularity, which is currently unverified.

# H7: Active Data Acquisition for LLM Pretraining

## 1. Hypothesis and SOFisher mapping

A learned policy that decides which region of the web (host, site section, link neighborhood) or which candidate licensed collection to acquire next, using cheap features (link structure, sampled quality scores) and measured gains from prior acquisitions, will acquire more useful tokens per dollar than uniform crawling or static quality-ranked crawling. This works if, and only if, quality and novelty are autocorrelated along the link graph, so that observing a region's yield is informative about its unvisited neighbors.

SOFisher mapping: field of view -> crawl frontier region (host or link cluster); expensive measurement (imaging) -> full crawl-and-filter-and-train-on of a region (the true cost driver, since crawling, dedup, quality classification, and attributed training-signal are all downstream); cheap state features (tissue morphology proxies) -> link graph position, PageRank-like scores, sampled FineWeb-Edu-style classifier scores on a small probe set; spatial autocorrelation of rare targets -> the empirical fact that quality and topical duplication cluster by host and by link neighborhood on the web.

## 2. Prior art

- Chakrabarti, van den Berg, Dom, "Focused Crawling," Computer Networks 1999. Foundational focused crawler using link classifiers to prioritize the frontier; ~391 citations. https://www.ra.ethz.ch/cdstore/www8/data/2178/pdf/pd1.pdf
- "Focused Crawling Through Reinforcement Learning," Springer 2018. RL over content+link state to pick next URL to fetch. https://link.springer.com/chapter/10.1007/978-3-319-91662-0_20
- Kontogiannis et al., "Tree-based Focused Web Crawling with Reinforcement Learning" (TRES), arXiv 2112.07620 (2021). End-to-end RL crawler with Tree-Frontier discretization for faster training. https://arxiv.org/abs/2112.07620
- "Deep Reinforcement Learning for Web Crawling," IEEE 2022. MDP formulation of crawl frontier selection. https://ieeexplore.ieee.org/document/9703160/
- Penedo, Kydlicek, et al., "The FineWeb Datasets," NeurIPS 2024. 15T-token dataset from 96 Common Crawl snapshots; ablation-driven filter/dedup recipe beats prior open web datasets. https://papers.neurips.cc/paper_files/paper/2024/file/370df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets_and_Benchmarks_Track.pdf
- HuggingFaceFW, FineWeb-Edu classifier and dataset. Linear regressor on Snowflake-arctic-embed-m embeddings, labels from Llama-3-70B-Instruct (0-5 scale), F1 82% at threshold 3; keeps 1.3T of 15T tokens, gains on MMLU/ARC/OpenBookQA. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Villalobos et al., "Will we run out of data?," arXiv 2211.04325 (2022, rev. 2024). Projects public human text stock exhausted by training demand between 2026 and 2032 under current trends. https://arxiv.org/abs/2211.04325
- Yu, Zhang, et al., "Craw4LLM: Efficient Web Crawling for LLM Pretraining," ACL Findings 2025, arXiv 2502.13347. Predicted-influence scheduler beats graph-connectivity priority; on a 900M-page graph, matches full-crawl downstream performance fetching only 21% of URLs. https://arxiv.org/abs/2502.13347
- Xie et al., "DoReMi," NeurIPS 2023, arXiv 2305.10429. Group-DRO on a 280M proxy sets domain weights for an 8B model; +6.5pp avg few-shot accuracy over The Pile's default weights, 2.6x fewer steps to baseline. Mixture-reweighting analogue, not crawl-time acquisition. https://arxiv.org/abs/2305.10429
- Common Crawl Foundation, crawl statistics. Corpus exceeds 10 petabytes since 2008, ~2-2.5B pages per monthly snapshot, 300B+ pages total; cited in 12,000+ papers. Sets the scale/cost baseline the policy must beat. https://commoncrawl.org/about
- Li et al., "Group-Level Data Selection for Efficient Pretraining" (Group-MATES), arXiv 2502.14709 (2025). Relational influence model with influence-aware clustering for group-level pretraining data selection. https://arxiv.org/html/2502.14709v2
- "CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training," arXiv 2406.10670 (2024). Conditional loss reduction as a cheap proxy signal for what data is worth acquiring for a target distribution. https://arxiv.org/pdf/2406.10670
- Udandarao et al., "Active Data Curation Effectively Distills Large-Scale Multimodal Models," CVPR 2025. Learner-in-the-loop selection deciding what to obtain next, not pre-filtering a static pool; directly analogous to H7's framing. https://openaccess.thecvf.com/content/CVPR2025/papers/Udandarao_Active_Data_Curation_Effectively_Distills_Large-Scale_Multimodal_Models_CVPR_2025_paper.pdf
- LLM Pulse / Digiday, licensing-deal trackers, 2025-2026. OpenAI ~2 dozen publisher deals, Amazon-NYT/Conde Nast/Hearst, Meta-News Corp/Le Figaro/Prisa (March 2026), Mistral-AFP (Jan 2025); most deals multi-year. Confirms licensing acquisition is an active, ad hoc, human-negotiated budget decision today, not a learned policy. UNVERIFIED per-deal dollar figures. https://llmpulse.ai/blog/ai-content-licensing-deals/
- Ng et al., "SOFisher: reinforcement learning-guided experiment designs for spatial omics," Nature Communications, May 2026 (DOI 10.1038/s41467-026-73404-6). RL policy sequentially selects next field-of-view; beats conventional dense-then-stitch acquisition on real spatial datasets. https://www.nature.com/articles/s41467-026-73404-6

## 3. What is genuinely open as of 2026

Crawl4LLM (2025) is the closest existing work: 79% crawl reduction at matched downstream performance, but it scores pages by predicted training influence from a one-shot reference-model pass, not by a sequential policy that updates beliefs about unvisited link neighborhoods from realized yield, the way SOFisher updates FOV choice from realized target density. Open questions: (a) whether sequential belief-updating beats Crawl4LLM's static ranking once scoring compute is controlled for; (b) whether the framing transfers from cheap-to-sample web crawl to licensing (real dollar cost, weeks-long feedback); (c) whether DQN-grade machinery beats a simpler bandit over site clusters, given DoReMi-style offline reweighting already captures much of the value cheaply. No paper found compares online sequential acquisition against Crawl4LLM-style static scoring on identical data, and none treats licensing acquisition as a sequential decision problem; the deal trackers above document ad hoc human negotiation, not an algorithmic policy. A year is needed because the crawl-time simulator is a multi-month build, the licensing benchmark must be built from scratch, and ruling out a single-graph artifact requires testing across at least two CC crawls plus one non-CC graph.

## 4. Three-condition check

1. Spatial/graph autocorrelation of quality and novelty: FineWeb-Edu's domain-classifier analysis and the education-density scoring pattern by host and topic cluster support autocorrelation of quality by site; Crawl4LLM's success at 21% coverage matching full-crawl performance is strong indirect evidence that pretraining value is not uniformly distributed and is predictable from graph position. Confirmed, moderate-strong.
2. Expensive measurement vs. cheap features: crawling+filtering+dedup+attributing downstream model gain to a region costs far more (compute, storage, negotiation time for licensing) than cheap proxies (PageRank-like link scores, a small sampled quality classifier pass). Confirmed by the FineWeb pipeline's own cost structure (heuristic filters are cheap; full quality/dedup pipeline and training runs are not) and by licensing deals taking weeks to months to close per the Digiday/LLM Pulse timelines. Confirmed.
3. Cheap features alone cannot rank targets well: FineWeb-Edu's classifier is itself an imperfect cheap-ish proxy (F1 82% at threshold 3, and a cited 2026 critique flags classifier-based quality filtering pitfalls, e.g. "Is a Document Educational or Just Wikipedia-Style?", arXiv 2605.23721), and Crawl4LLM explicitly replaces cheap graph-connectivity scoring with a model of downstream influence because connectivity alone underperforms. Confirmed.

## 5. Year plan

Phase 1 (weeks 1-8, 4 A100s, first experiment under one week): build a simulator from a public CC link-graph subsample (Crawl4LLM release or a CC index snapshot, ~10-50M pages) with precomputed FineWeb-Edu scores as ground-truth yield. Implement a bandit/DQN policy over host or link-cluster "regions" that picks the next region to score, observes yield, and updates. Metric: useful tokens per crawled page (tokens from pages scoring >=3 on FineWeb-Edu, divided by pages fetched, at fixed budget). Strongest baseline: Crawl4LLM's static influence-ranked scheduler, with FineWeb's uniform-crawl-then-threshold as a simpler floor. Feasible in a week since this phase needs only cheap-feature simulation and small scoring passes, not full pretraining runs.

Phase 2 (months 3-6): validate transfer across a second, disjoint CC snapshot and a non-CC graph, to rule out overfitting to one graph's autocorrelation structure; add a real downstream signal by training small proxy LMs (125M-350M) on policy-selected vs. baseline-selected pools and comparing loss/few-shot accuracy, following the DoReMi/Crawl4LLM evaluation pattern.

Phase 3 (months 7-12): extend to licensing decisions, which have no existing benchmark. Build a small candidate-collection dataset (proxy quality/novelty features per source) and test whether sequential acquisition beats static quality-ranking under a simulated per-collection cost and multi-week feedback delay; highest risk since the benchmark must be built from scratch.

## 6. Kill criteria

Kill if, on the Phase 1 simulator, the sequential policy does not beat Crawl4LLM's static influence ranking by a clear margin (e.g., >=10% more useful tokens per crawled page at matched budget) after reasonable tuning; a null result here means the sequential/online-update structure is not adding value over one-shot scoring, and the SOFisher analogy does not hold at web scale. Also kill if the autocorrelation signal measured directly on the CC graph (yield correlation between a region and its link neighbors) is weak (e.g., correlation not clearly above what static domain-level heuristics already capture), since that undercuts condition 1 directly.

## 7. Impact

If it holds, a lab would replace its current two-stage pipeline (crawl broadly, then filter/rank by a static classifier) with a closed-loop scheduler that decides, before fetching, which hosts or link clusters to crawl next based on yield observed so far in that crawl run, cutting wasted fetch/filter/dedup compute the way Crawl4LLM already demonstrates is possible, but adaptively rather than via a one-shot influence score, and would apply the same sequential logic to prioritizing which licensing negotiations to pursue next based on realized value from already-licensed sources, rather than negotiating deals in the ad hoc, PR-driven order seen in the current 2025-2026 deal trackers.

## 8. Scorecard

SCORECARD
novelty: 5
feasibility_1yr: 6
lab_impact: 5
sofisher_fit: 6
strongest_baseline: Crawl4LLM (arXiv 2502.13347)
one_line_pitch: A sequential, yield-updating crawl/licensing scheduler that exploits link-graph autocorrelation in text quality may beat Crawl4LLM's static influence ranking, but this must be proven against that exact baseline before it justifies a year.

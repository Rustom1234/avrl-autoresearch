# Topic D: Expensive per-example scorers and how people choose subsets

## Gradient-based attribution / influence functions

**Anthropic, Grosse et al. 2023** uses EK-FAC to approximate the inverse-Hessian-vector product, scaled to 52B-parameter models. Even so, the authors needed TF-IDF pre-filtering of candidates and query batching to make per-query cost tractable, implying a naive gradient dot-product over the full pretraining set is far too expensive. Source: https://arxiv.org/abs/2308.03296

**TRAK** (Park et al. 2023) approximates influence with random gradient projections and a handful of checkpoints; cost scales roughly as O((N+Q)SW^2L + NQR + R^3), and the projection dimension may need to scale with model size to stay accurate. Source: https://arxiv.org/pdf/2410.13850

**DataInf** (Kwon et al. 2023) gives a closed-form influence estimate for LoRA fine-tuning, avoiding explicit Hessian inversion, but is a fine-tuning-scale tool (adapter parameters), not full-corpus scale. Source: https://arxiv.org/html/2310.00902

**LESS** (Xia et al. 2024) builds a reusable low-dimensional gradient datastore (LoRA gradients), selecting by cosine similarity to target-task gradients; a LESS-selected 5% of instruction data can outperform full-data training. It is a fine-tuning/instruction-selection method, not applied to trillion-token pretraining corpora. Source: https://openreview.net/pdf?id=Kw3ckB2Kfc

**Cost verdict:** a full gradient (plus any Hessian correction) per example, over a multi-trillion-token corpus, costs one or more backward passes per document, comparable to or exceeding the pretraining run itself. Every paper above exists because full-corpus influence is infeasible; each restricts to a small candidate pool, fine-tuning scale, or low-rank approximation. UNVERIFIED: no concrete "$ per example" or "% of training FLOPs" figure found for full-corpus influence, only qualitative infeasibility statements.

## LLM-judge annotation at scale: FineWeb-Edu as the canonical case

FineWeb-Edu used Llama-3-70B-Instruct to score educational quality (0-5 scale) on **467,424 sampled documents** (dataset card lists this row count). Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations. These ~460-467k labeled scores were then used to train a small classifier (a Snowflake-arctic-embed embedding model plus a regression head) to scan the **entire** FineWeb corpus (15 trillion tokens; FineWeb-Edu itself is 1.3 trillion tokens after filtering). Sources: https://arxiv.org/html/2406.17557v1, https://zeroentropy.dev/concepts/fineweb/

This is "distill then scan": run the expensive judge (70B model) on a tiny slice (hundreds of thousands of docs against many billions in FineWeb), then run a cheap distilled classifier on 100%. UNVERIFIED exact percentage, since full FineWeb document count was not found, only token count.

Rough per-example cost for a 70B judge call is one forward pass over a short document. General LLM-judge pricing context (not FineWeb-specific, UNVERIFIED as a direct proxy): GPT-4-class pricing ran near $0.01/1K input tokens; a 6.2M-tweet GPT-4 annotation run was estimated near $8,990 (~$0.0014/item). HuggingFace did not publish a cost figure for the Llama-3-70B annotation run itself.

## Ask-LLM (Sachdeva et al. 2024, "How to Train Data-Efficient LLMs")

Ask-LLM prompts an instruction-tuned LLM to judge each example directly (one full LLM call per candidate), versus a cheaper Density-sampling baseline. Training on Ask-LLM-selected data beats full-data training even after rejecting 90% of the corpus, converging up to 70% faster. It is explicitly framed as the expensive method compared against cheaper alternatives, evaluated on pretraining-scale ablations, not shown scaled to a full production corpus. Source: https://arxiv.org/abs/2402.09668

## Human review pipelines

General annotation-market pricing: preference/RLHF-style human labels run roughly $0.10-$5 per example. For pretraining-corpus quality, one cited audit design used 16 annotators reviewing about 500 randomly sampled documents (3 annotations each, majority vote) as a calibration/audit sample, not applied to any meaningful fraction of the corpus. Source: search citing DataComp-LM-style audit methodology (https://arxiv.org/pdf/2406.11794); the 500-document figure is UNVERIFIED, no primary URL directly fetched.

## Live secret verification (Truffle Security scan)

TruffleHog scanned 7.6 petabytes / 186.9 million files across ~815,000 HuggingFace dataset repos, finding 221,303 unique **verified live** credentials by calling provider APIs to confirm each still authenticates. Cost per candidate is an outbound API call, not a backward pass; the blog states the entire scan cost less than one month of abuse on a single leaked top-tier API key. This plausibly ran on (close to) the whole corpus, since its unit cost is an HTTP call. Source: https://trufflesecurity.com/blog/scanning-7-6-petabytes-of-ai-training-data-for-secrets

## Membership inference / memorization checks (Min-K%, Min-K%++)

Min-K% scores membership using only the top-k% lowest-likelihood tokens of a sequence, from a single forward pass under the target model. Per-example cost is one forward pass, cheap relative to influence functions and comparable to perplexity filtering, but scanning an entire corpus is still a full inference pass, nontrivial but far cheaper than gradient/influence approaches. Source: https://arxiv.org/pdf/2503.19338 (survey). No paper found applying Min-K% to 100% of a production pretraining corpus as a filtering step; it is used as an attack/audit tool on candidate suspect documents, not a blanket scorer. UNVERIFIED for any "% of corpus scored" figure.

## Active / adaptive selection, coreset, submodular, proxy-model scoring (2024-2026)

- **DsDm** (Engstrom et al., ICML 2024): "datamodels" approximate which subset most helps a target task; up to 2x compute multiplier over baselines, avoids hand-crafted quality heuristics. Source: https://arxiv.org/abs/2401.12926
- **MATES** (NeurIPS 2024): trains a small "data influence model" on oracle influence probed from the *currently training* model, re-predicts influence, and reselects data for the *next* stage: the clearest case of **adaptive, iterative** rescoring. Source: https://arxiv.org/abs/2406.06046
- **Coreset/submodular methods** (TAGCOS, SubZeroCore, DiSF, INGENIOUS): select diverse, non-redundant subsets via clustering plus submodular coverage/density on cheap embeddings, avoiding per-example expensive scoring. Sources: https://arxiv.org/pdf/2407.15235, https://arxiv.org/pdf/2504.20644
- **Proxy-model scoring**: small models (125M-1.3B) or perplexity-correlation methods rank candidate data/domains cheaply as a stand-in for full-scale runs. Source: https://arxiv.org/html/2409.05816.pdf

## Answers

**1. Scorers costing more than ~1% of pretraining compute if run on everything:** gradient-based influence functions and Hessian-corrected variants (Anthropic EK-FAC, TRAK, DataInf extended to full corpora): each needs at least one backward pass, often plus curvature computation, per document, which at trillion-token scale rivals or exceeds the pretraining compute itself. Full LLM-judge annotation (Ask-LLM, or a 70B judge as in FineWeb) run on 100% of a multi-trillion-token corpus would also plainly exceed 1%, exactly why nobody does it (FineWeb-Edu scored well under 500k of many billions of documents). Membership-inference/Min-K% (single forward pass) and secret-verification API calls are cheaper and closest to full-corpus scale.

**2. Is anyone adaptively choosing the scored subset based on earlier scores?** Yes, one clear case: MATES retrains its influence-model proxy on freshly probed oracle signal at each pretraining stage and reselects data accordingly (2024). DsDm and datamodel-style approaches are model-aware but generally computed once per target task, not continuously online. Most quality-classifier pipelines (FineWeb-Edu style) are NOT adaptive: the classifier trains once on a fixed labeled sample, then applies statically to the whole corpus.

**3. The standard "distill then scan" recipe and known failure modes:** (a) heuristic/rule filters remove obvious junk, (b) a small human- or LLM-labeled sample (hundreds of thousands of docs at most, e.g. FineWeb-Edu's ~467k) is scored by an expensive judge, (c) a cheap classifier or small proxy model trains on those labels, (d) the cheap model scans 100% of the corpus. Failure modes found: the small proxy only approximates the big judge's preferences and can amplify its idiosyncratic biases across billions of docs; classifiers trained this way can favor easy, redundant, low-diversity text, causing dimensional collapse (source: https://arxiv.org/pdf/2504.20644); quality can show an "up-down-up" trend, initial gains, then decline as lower-quality retained data dominates mid-training, recovering once volume is sufficient (UNVERIFIED, from search summary of distillation literature, not independently fetched).

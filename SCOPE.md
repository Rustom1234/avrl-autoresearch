# Audit-Ahead Pretraining: scope v1

Date: September 9, 2026. Owner: Rustom. Cluster: hannah (4x A100 40GB, 3-day cap, no internet on compute). Status: scope only, nothing built.
Concept log: `~/claude.ideas/sensitive-data-active-audit.md`. Source template: SOFisher (Ng et al., Nature Communications 2026).

## 1. The claim we are trying to earn

"Audit-ahead pretraining": a pretraining data pipeline in which a budgeted active-search policy decides which regions of the corpus get inspected by an expensive sensitivity detector, using what previous inspections found and, in the online variant, the training model's own memorization signals, so that at a fixed inspection budget the resulting model leaks less sensitive training data with no loss in downstream quality.

What "SOTA" can honestly mean here. There is no existing leaderboard. The defensible claim is dominance on a frontier we define and publish: privacy leakage (extractable sensitive strings, membership inference) versus downstream quality (validation loss, small-benchmark accuracy) versus audit budget (detector calls per token). SOTA means: for every budget, audit-ahead sits below every baseline curve on leakage at equal quality. The baselines must include what production pipelines actually do (Dolma-style exhaustive regex, FineWeb-style anonymization) and the orthogonal training-time defenses (Goldfish loss, deduplication), otherwise the win is against straw men.

## 2. Method

### 2a. Offline active audit (v0)

- Corpus laid out in a geometry where sensitivity is autocorrelated. Candidates: domain/path tree, crawl-shard order, 2-D map of document embeddings. The layout is chosen by measurement (section 3, stage 0), not by assumption.
- FOV = a bucket of documents (one domain-path node or one map cell, ~256 docs).
- Expensive detector D (the "imaging"): an LLM judge for contextual sensitivity (real personal data about private individuals, credentials, medical or financial details), served locally. Regex and cheap classifiers are NOT the detector; they are state features. This is what makes the budget real.
- Policy: active search in SOFisher's form. State = position, local visit counts, cheap features of the current bucket (regex PII rate, language, length stats, domain category), running hit rate. Actions = move to a neighboring bucket, plus restart (SOFisherWR). Reward = sensitive docs found minus revisit penalty. Policy classes compared: uniform, raster, static cheap-feature triage, UCB bandit over regions, DQN (SOFisher), and nonmyopic active search (Jiang et al. 2017) as the strongest classical baseline.
- Output: a filtered corpus at budget B, where flagged docs are dropped or their sensitive spans masked.

### 2b. Audit-ahead online loop (v1, the pretraining method)

- The pretraining run streams data. A lookahead window of upcoming buckets is scored with cheap signals from the model being trained: per-example loss and gradient norm on a small probe batch (both are known memorization proxies), plus the offline features.
- The active auditor spends its per-step detector budget on the lookahead window using those signals, and flagged docs are removed or span-masked before the data loader consumes them.
- The model therefore never sees the content the auditor caught, and the auditor gets better at finding it as the model reveals what it is starting to memorize. That coupling is the new part; nothing in the prior-art pass does it.
- Ablations: offline-only auditor (2a) versus online auditor with model signals; masking versus dropping; Goldfish loss on top.

## 3. Experimental design

### Data

- Web: FineWeb `sample-10BT` (about 15M docs, 10B tokens). Enough for 125M and 350M pretraining runs and for a dense ground-truth audit of a 1M-doc subset.
- Code: The Stack smol or a single-language slice, for the secrets setting where clustering is expected to be strongest.
- Dense ground truth ("the fully imaged slide"): run D over a 1M-doc random subset once. This is what the simulator replays. Cost: Qwen3-8B judge on vLLM at ~100 to 200 docs/s per GPU, so 1M docs is 2 to 3 GPU-hours.

### Detectors

- D (expensive, the budgeted one): Qwen3-8B judge with a fixed rubric; validated against 500 hand-labeled docs and against ai4privacy labels for precision/recall. For code: StarPII plus verified-format secret checks.
- Cheap state features: regex PII (email, phone, IP, address patterns), URL/domain category, language ID, length, duplication count, and in v1 the model's own loss and gradient norm.

### Layouts and stage-0 measurement

- For each layout, compute the neighbor hit-rate ratio and Moran's I of the D label. A layout passes if neighbors of a hit are at least 3x more likely to be hits than random. If no layout passes, the method is dead and the measurement is the paper.

### Policies and audit metrics

- Hits per detector call, detector calls to reach k hits, recall of all sensitive docs at budgets of 1%, 3%, 10% of the corpus. Train policies on one shard, test on a held-out shard (SOFisher's cross-aging-stage protocol).

### Pretraining runs and leakage metrics

- Models: 125M (2.5B tokens, about 1 to 2 hours on hannah) for the sweep; 350M (7B tokens, about 8 to 10 hours) for the headline; identical tokenizer, schedule, and seed protocol across arms.
- Arms at equal audit budget: no filtering; uniform-audit filtering; static-triage filtering; active-audit filtering (2a); audit-ahead (2b); Dolma-style exhaustive regex (unbudgeted reference); Goldfish loss (orthogonal reference, also combined with 2b).
- Leakage: (i) canary extraction: synthetic but realistic PII canaries inserted at duplication counts 1, 4, 16, 64, placed both inside sensitive clusters and scattered, extraction rate under prefix prompting; (ii) real-PII verbatim extraction: prompt with training-doc prefixes, count generated emails/phones that appear in the training set; (iii) membership inference AUC on sensitive versus non-sensitive held-out docs.
- Quality: validation loss on a clean held-out slice; lm-eval-harness at this scale (HellaSwag, PIQA, ARC-easy, LAMBADA). The pass condition is "no significant degradation," pre-registered with seeds.

## 4. Compute budget on hannah

| Item | Estimate | Notes |
|---|---|---|
| Embedding 15M docs (MiniLM) | under 1 GPU-hour | for the embedding layout |
| Dense ground truth, 1M docs, Qwen3-8B judge | 2 to 3 GPU-hours | once |
| Simulator and policy training | CPU, plus 1 GPU for DQN | SOFisher code is small |
| 125M pretraining run | 1 to 2 hours on 4 GPUs | 7 arms x 3 seeds = 21 runs, about 2 days total |
| 350M pretraining run | 8 to 10 hours on 4 GPUs | 4 arms x 2 seeds = 8 runs, about 4 days total |
| Online auditor overhead in v1 | 10 to 20% of step time | judge on GPU 3, training on GPUs 0 to 2 |

The 3-day cap, exclusive-node rule, and NFS-only storage all still apply (see `~/.claude/.../hannah-cluster-access`). Everything here fits; no rented compute.

## 5. Stages, gates, dates

| Stage | Weeks | Deliverable | Gate |
|---|---|---|---|
| 0. Autocorrelation | Sept 9 to 16 | Table of neighbor hit-rate ratios per layout on FineWeb and Stack; judge validation numbers | At least one layout at 3x or better |
| 1. Offline auditor | Sept 16 to 30 | Simulator on SOFisher's interface; six policies; hits-vs-budget curves; cross-shard transfer | Active search beats static triage by a pre-registered margin at 3% budget; if triage wins, stop RL and publish triage |
| 2. Offline pretraining comparison | Oct 1 to 14 | 125M sweep, leakage and quality curves per arm | Active-audit arm leaks less than uniform at equal quality, CI excludes zero |
| 3. Audit-ahead online loop | Oct 14 to Nov 4 | v1 implementation on the 125M setup, then one 350M headline run | Online beats offline auditor at equal budget; otherwise report offline as the method |
| 4. Writeup and release | Nov 4 to 15 | Paper-style report, code, filtered-data recipes, canary set | Dated public artifact before mid-November |

Hiring clock: the plan that preceded this one needed a public artifact by November 1. Stage 2's result is that artifact if stage 3 slips.

## 6. What would kill it, and what we say then

- No autocorrelation (stage 0 fails): publish the measurement plus the triage baseline; it is still the first quantification of how sensitive content clusters in web corpora.
- Static triage matches active search (stage 1): the pretraining paper still stands with triage as the auditor; the RL part is dropped.
- Filtering by any auditor does not reduce leakage at these scales (stage 2 fails): the canary design is likely the problem; fall back to canary-only leakage as the metric and say so.
- The online signals add nothing (stage 3 fails): v0 is the method; the writeup reports the null.

## 7. Honest positioning

- This is a data-curation method that plugs into pretraining, plus an online coupling to the training model. It does not change the architecture or the objective.
- The privacy claim is about extractable training data, measured with canaries and verbatim extraction. It is not differential privacy and must not be described as such.
- "SOTA" is on a frontier we define; the paper must show the frontier with all baseline curves, including exhaustive regex and Goldfish loss, or reviewers will call it a straw-man comparison.
- The 125M/350M scale is small; the trend across scale (125M to 350M) is the evidence that it will hold larger. Say so.

## 8. Decisions for Rustom

1. Sensitivity target: PII plus secrets for v0 (my pick), with contextual privacy via the LLM judge as the expensive detector. Confirm or change.
2. Web-first or code-first for stage 0: code (secrets in repos) is the safer bet for strong clustering; web is the bigger claim. I propose running both in stage 0 since it is cheap.
3. Whether to run the three-critic review on this scope before stage 0, as we did for the forecaster plan.

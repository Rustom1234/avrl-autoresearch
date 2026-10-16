# Legit directions for SOFisher-style active search in LLM training data

Compiled 2026-09-18 on goldberg. Sources: lit/A_online_selection.md, lit/B_active_search.md, lit/C_corpus_structure.md, lit/D_expensive_scorers.md (Sonnet literature passes, claims marked UNVERIFIED there stay unverified here), plus the experiments in runs/ (results section below is filled from the run logs, not from memory).

## The one structural fact that decides everything

Active search beats a one-shot ranking only when three things hold together (lit/B, Jiang et al. 2017; SOFisher's own "non-randomness" and "association" assumptions):
1. the expensive scorer costs far more than the cheap features (budget is real),
2. the cheap features are too weak to rank targets on their own,
3. the target label is autocorrelated in some layout, so one observation predicts its neighbours.

Sep 9 and Sep 10 measurements on FineWeb (regex personal-information proxy): condition 2 holds (cheap AUC 0.55 to 0.59), condition 3 fails (best layout 2.4x at 1M docs, 1.3x by domain at 10.4M docs), and condition 1 is weak at single-run scale (judging all of sample-10BT with Qwen3-4B is about one GPU-day). Static triage beat every adaptive policy.

## What the literature pass established

- Nobody has published a domain-level or repo-level concentration measurement of undesirable content (quality, toxicity, personal information, secrets) on a web or code corpus. FineWeb and Dolma filter at the URL level on an unmeasured assumption; WIMBD explicitly does not cross-tabulate domain against PII or toxicity (lit/C).
- No paper frames "which document to audit next" as budgeted active search (lit/B). Benchmark-contamination, copyright and PII auditing are detection-method papers with one-shot ranking.
- Every pretraining data-selection method either distills an expensive judge into a cheap scorer and scans everything (FineWeb-Edu: 467k Llama-3-70B annotations, then a classifier over 15T tokens; QuRating; DSIR) or adapts at domain granularity with no per-document expensive score (ODM, ADO, Aioli, AC-ODM) (lit/A, lit/D).
- The one method that budgets an expensive per-example scorer adaptively is MATES (NeurIPS 2024). Verified from the paper (arXiv 2406.06046, Sec 3.1, Table 2, Table 5): oracle influence is computed on a random hold-out subset (80k examples at the first update, 20k per later update, refreshed every 10k steps), and data selection costs 21.7% of total FLOPs at 410M and 11.5% at 1B, of which oracle collection is about 4%. The probes are chosen at random. Nobody has asked whether choosing them adaptively helps.
- Expensive scorers that genuinely cannot be run on everything: gradient or influence based scores (one backward pass or more per document, rivals training compute), and LLM-judge annotation at multi-trillion-token scale. Live secret verification (TruffleHog on 7.6 PB of Hugging Face data) is cheap per unit and already runs on everything (lit/D).

## Candidate directions, ranked

### D1. Active probe selection for model-aware data selection (MATES-style)
Setting where all three conditions can hold: the scorer is an oracle influence (backward pass plus a reference-loss evaluation), it is already budgeted to a few percent of training FLOPs, and the probes are currently sampled at random. Question: does oracle influence autocorrelate by domain, host, or embedding cluster strongly enough that a bandit over regions picks more informative probes per unit of compute than random hold-out sampling, and does the resulting influence model select better data? Test in progress: runs/infl (first-order influence, Qwen3-0.6B-Base, 16k docs across 800 domains, two reference sets). Paper shape if positive: "active probing for data influence models", competing against MATES with random probes at equal probe budget on a 125M to 410M run on hannah.

### D2. Secrets in code, by repository
The only sensitivity type where clustering was expected to be strong (concept log) and where an expensive verifier exists (live credential check against the provider). Test done at 2 shards of codeparrot/github-code-clean and rerunning at 10 shards with bootstrap CIs and a budgeted-audit simulator (runs/code). Caveat from lit/C: regex secret hits include test fixtures and documented example keys; a real version needs verification or a StarPII-style detector as the label.

### D3. The measurement paper
"Where does undesirable content live in FineWeb?" Neighbor-hit ratios, ICC, and top-k concentration by domain and host for quality (FineWeb-Edu score), loss under a reference model, regex personal information, and a contextual-sensitivity LLM judge, at 1M and 10M documents. Fills the gap lit/C identified. Every number needed is already produced or in flight (runs/s0-layouts, runs/density, runs/hyp, runs/judge). Small paper, honest, no method claim.

### D4. Contextual-sensitivity judge geometry
If the LLM-judge label clusters where the regex label does not (runs/judge, 20k docs), the Sep 9 gate reopens for personal information. Otherwise it closes for good on web text.

### Not worth pursuing
- Active search for generic personal information on web text: measured dead twice.
- Runtime reduction by dropping data: at a fixed token budget, dropping data changes the mixture, not the step count. Quality filtering is a quality-per-token story, already owned by distill-then-scan classifiers.
- Domain-level online mixing: crowded (ODM, ADO, Aioli, AC-ODM) and not a search problem.

## Results from this session
(filled in below as runs complete)

### R1. Judge label geometry (runs/judge, 20k random FineWeb docs, Qwen3-4B contextual-sensitivity rubric)
- Base rate 0.76% (152 positives: 84 contact, 28 medical, 10 financial, 6 legal). Regex strong_pii base rate on the same docs is 7.25%. Regex is a bad proxy for the judge: precision 2.4%, recall 23%. The Sep 9 negative result was measured on the wrong label.
- Neighbor ratios, judge vs regex on the same 20k docs: embedding cell (k=400) 2.97x vs 1.09x, bootstrap over cells [3.2, 4.6]; cell k=100 1.58x vs 1.00x; host 7.3x vs 2.8x but carried by 2 hosts with 2 hits each (4 of 152 hits), so not evidence; domain 1.19x, same sparsity problem at 20k docs.
- Concentration: top 10% of k=400 cells hold 70% of judge hits (regex by embedding cell on Sep 9: 28%). Top cells are topical: escort and adult listings (contact details), personal medical narratives and patient forums (medical), personal blogs.
- Cheap-feature AUC 0.67 (regex counts, length, language score). Adding leave-one-out domain regex rate from the 10-shard features does not help (0.66). Email count alone: 0.55.
- Reading: contextual sensitivity is a topic-level property, so it clusters in embedding space; regex PII is a footer-level property, so it does not. Judge cost at 7.6 docs/s in eager mode (compiled vLLM should be 5 to 10x faster) makes the budget real: judging all 10M docs of sample-10BT at 50 docs/s is 2.3 GPU-days on the 5090.
- Still thin: 152 positives. A judged run over the 73k domain-stratified sample is queued (runs/judge/labels_hyp.parquet) so judge, edu, loss and regex labels align on the same docs with real multi-doc domain groups.

### R2. Code secrets by repository (runs/code, 10 shards of codeparrot/github-code-clean, 1.27M files, 510k repos)
- Regex labels: high-confidence patterns (AWS, Google, OpenAI, GitHub, Slack keys, private key blocks) 0.04% of files, 564 hits; generic `password = "..."` style 0.23%, 2,928 hits.
- Neighbor ratio by repo: 31x (high), 7.8x (generic), 6.7x (any). Lift is real but coverage is tiny: only 7 to 10% of hits sit in a repo that has a second hit in-sample. Shards are random 1.1% file samples, so repos are fragmented (median 1 file per repo, 61% of files in repos with 3 or more files). A neighbor-expansion policy can therefore recover at most about a tenth of the hits at this density. The group-bootstrap CI printed in run10.log is biased for the leave-one-out ratio and should be ignored.
- Cheap features (language, extension, size, path depth) already give AUC 0.78 to 0.81, so condition 2 is weak here: extension-based triage is strong.
- Label quality: in the 2-shard sample, 68% of high-confidence hits were Google browser API keys embedded in front-end HTML (often intentionally public), and 25 of 104 sat in test, example or fixture paths. A real study needs a verified-secret label (live check) or StarPII, and whole-repo data rather than file samples.
- Verdict: direction D2 is plausible but this dataset cannot test it. Do not cite the 33x number without these caveats.

### R3. Quality, loss and regex labels by domain (runs/hyp, 73k docs: 44k stratified over 2,000 domains with 10+ docs, 29k uniform)
Scores: FineWeb-Edu classifier (0 to 5), mean token loss under Qwen3-4B (first 512 tokens), regex features.

| label | base rate | neighbor ratio domain | neighbor ratio host | top 10% domains hold | cheap-feature AUC |
|---|---|---|---|---|---|
| edu_high (score >= 3) | 5.7% | 1.86x | 1.97x | 54% | 0.65 |
| edu_low (score <= 1) | 74.5% | 1.01x | 1.03x | 14% | 0.65 |
| loss top decile (hard) | 10% | 1.64x | 1.81x | 48% | 0.75 |
| loss bottom decile (easy) | 10% | 1.37x | 1.47x | 41% | 0.67 |
| regex strong_pii | 6.8% | 1.80x | 1.99x | 47% | (leaks, regex is in the features) |

- Intraclass correlation by domain: edu score 0.47, loss 0.42. Roughly half the variance of quality and of loss is between domains, which is a strong continuous signal, but as a rare-hit search target neither clears 2x, let alone 3x. edu_low is a majority label (cap on its ratio is 1.34x), not a search target.
- Reading: for quality and learnability the useful structure is "domain means are informative", which is exactly what domain-level mixing methods (DoReMi, ODM, ADO) already exploit, and what a distilled classifier learns anyway. No active-search angle here on web text.
- Budgeted-audit simulation on code (runs/code/sim_code.py, 573k files in repos with 3+ files, 70k repos, 4 files per visit, 3 seeds). Hits found, label = any regex secret (930 hits):

| budget | uniform | static triage (ext, language, size, depth) | Thompson flat | Thompson + static prior | neighbor expansion | oracle |
|---|---|---|---|---|---|---|
| 1% | 9 | 74 (8.2x) | 12 (1.3x) | 44 (4.9x) | 42 (4.7x) | 930 |
| 3% | 25 | 198 (8.0x) | 43 (1.7x) | 114 (4.6x) | 128 (5.2x) | 930 |
| 10% | 89 | 411 (4.6x) | 130 (1.5x) | 276 (3.1x) | 315 (3.6x) | 930 |

  High-confidence label (146 hits): same ordering, static 3.8x to 9.3x, adaptive 3.1x to 9.7x, within noise of each other at 1% and below static at 3% and 10%. Static triage dominates because file extension is a strong cheap feature (AUC 0.8) and repos are fragmented. Same conclusion as web text, for the opposite reason: there condition 3 failed, here condition 2 fails.

### R4. All labels on the same 73k docs (runs/hyp/unified.md; judge = Qwen3-4B contextual-sensitivity rubric, 540 positives, 0.74%)

| label | base rate | neighbor ratio domain | host | embedding cell k=300 | k=1000 | top 10% domains hold | top 10% cells hold | AUC cheap | AUC cheap + 32 embedding PCs |
|---|---|---|---|---|---|---|---|---|---|
| judge sensitive | 0.8% | 6.45x | 8.09x | 1.49x | 2.68x | 100% | 55% | 0.65 | 0.87 |
| edu_high (>= 3) | 5.7% | 1.86x | 1.97x | 1.18x | 1.60x | 54% | 47% | 0.65 | 0.93 |
| loss top decile | 10% | 1.64x | 1.81x | 1.00x | 1.06x | 48% | 27% | 0.72 | 0.81 |
| regex strong_pii | 6.8% | 1.80x | 1.99x | 1.00x | 1.05x | 47% | 27% | (leaks) | (leaks) |

- The judge label is the only one that clears the 3x gate, and it does so by domain and host, not by embedding cell. 168 of 2,000 stratified domains carry a hit; 63% of hits sit in a domain with at least one other hit (host: 59%). So neighbor expansion can reach most hits, unlike regex on code (10%).
- Conditions check for the judge label: (1) budget is real (7.8 docs/s on the 5090, so 10M docs is 15 GPU-days; regex is free); (2) cheap features are weak (AUC 0.65) but cheap plus a sentence embedding is strong (0.87), which is the distill-then-scan competitor; (3) autocorrelation by domain 6.5x. Two of three hold; the third depends on whether the competitor is allowed an embedding.
- Budgeted-audit simulation, judge label, buckets = domains with 3+ docs (2,102 buckets, 286 hits), 4 docs per visit, 5 seeds. First version (adaptive policies inspect within a bucket in random order):

| static features | budget | uniform | static triage | Thompson + static prior | neighbor expansion | oracle |
|---|---|---|---|---|---|---|
| cheap | 1% | 2 | 11 (6.9x) | 6 (3.5x) | 6 (3.8x) | 286 |
| cheap | 3% | 8 | 29 (3.8x) | 12 (1.6x) | 15 (2.0x) | 286 |
| cheap | 10% | 27 | 63 (2.3x) | 32 (1.2x) | 27 (1.0x) | 286 |
| cheap + embedding | 1% | 2 | 35 (21.9x) | 20 (12.8x) | 19 (12.0x) | 286 |
| cheap + embedding | 3% | 8 | 65 (8.6x) | 39 (5.2x) | 59 (7.8x) | 286 |
| cheap + embedding | 10% | 27 | 135 (4.9x) | 78 (2.9x) | 102 (3.7x) | 286 |

  Same ordering by embedding cell (225 buckets). That version is unfair to the adaptive policies (they discard the doc-level score inside a bucket); the hybrid rerun (rank within bucket by static score, adapt across buckets) is in R5.

### R5. Fair budgeted-audit comparison on the judge label (runs/hyp/sims_hybrid.log, runs/sim_generic.py)
Policies at equal detector budget, 5 seeds, 4 docs per bucket visit:
- static (free labels): classifier fit on judge labels from a disjoint quarter of the buckets (about 13.7k free labels), then rank. This is distill-then-scan with a labelling budget about 10x the audit budget, an upper bound for static.
- static (budget-matched): spends half the budget on random labels, fits the same classifier, ranks the rest. Distill-then-scan at honest cost.
- TS flat: Thompson sampling over buckets, no features, no labels.
- TS online: Thompson over buckets, plus a classifier refit on the labels collected so far (every budget/8 inspections) used for within-bucket ranking and as the bucket prior. No free labels. This is the SOFisher analogue: cheap state features plus learning from what the audit finds.
- TS + free prior, neighbor: adaptive policies given the free-label static model (upper bounds for adaptive).

Buckets = 300 MiniLM k-means cells over the 73k docs (225 cells with 3+ docs in the sim split, median 232 docs, 398 hits), features = cheap + 32 embedding PCs. Hits found:

| budget | uniform | static, free labels | static, budget-matched | TS flat | TS online | oracle |
|---|---|---|---|---|---|---|
| 1% (548 calls) | 5 | 60 | 15 ± 8 | 4 | 12 ± 5 | 398 |
| 3% (1,644) | 11 | 111 | 42 ± 15 | 14 | 91 ± 17 | 398 |
| 10% (5,480) | 35 | 214 | 127 ± 23 | 84 | 231 ± 4 | 398 |

Same layout, cheap features only (no embedding): TS online 11 / 36 / 117 vs static budget-matched 8 / 19 / 67. Buckets = domains (2,102 buckets, median 14 docs): TS online 13 / 26 / 110 vs static budget-matched 7 / 21 / 68 with embeddings; with cheap features only they tie (33 vs 40 at 10%).

Reading:
- At equal label budget the self-training bandit beats distill-then-scan by about 2x at 3% and 1.8x at 10% over embedding cells, and at 10% it matches the static model that had 13.7k free labels. Below 1% budget nothing learns; the bandit needs a few hundred inspections before its classifier is useful.
- The gain comes from two things the static baseline cannot do: it spends its labels where hits are (so its classifier sees more positives per label), and it exploits cell-level hit rates directly. Domain buckets are too fine for the bandit at these budgets (2,102 arms, 4 docs per visit); embedding cells with hundreds of docs are the right granularity here. A hierarchical layout (cell, then domain within cell) is the obvious next step.
- Caveats: 5 seeds with large spread (± 17 at 3%); 398 positives total; the budget-matched static baseline uses random labelling, and an uncertainty-sampling active-learning baseline (which also learns from its own labels) is the stronger competitor and must be added before any claim; the sample is domain-stratified (domains with 10+ docs in the shard), and cells were built on the 73k docs, not the full corpus; single judge model and rubric, no human validation of the judge yet.

### R6. Ablation: does the layout matter, or just learning as you go? (8 seeds, runs/hyp/sims_hybrid.log)
Added greedy online: refit the classifier on collected labels every budget/8 inspections, inspect the top-scored unseen docs anywhere, no buckets. Judge label, cheap + embedding features:

| layout | budget | static, budget-matched | greedy online (no layout) | TS online (bandit over buckets) | static, free labels |
|---|---|---|---|---|---|
| 300 cells | 1% | 12 ± 8 | 40 ± 13 | 16 ± 13 | 60 |
| 300 cells | 3% | 40 ± 14 | 107 ± 19 | 91 ± 14 | 111 |
| 300 cells | 10% | 131 ± 22 | 220 ± 19 | 230 ± 5 | 214 |
| domains | 3% | 23 ± 7 | 50 ± 9 | 30 ± 14 | 65 |
| domains | 10% | 75 ± 16 | 130 ± 13 | 106 ± 11 | 135 |

With cheap features only (no embedding), 300 cells: greedy 20 / 38 / 89 vs bandit 9 / 31 / 113 vs static matched 7 / 20 / 73.

Reading:
- The 2x gain over budget-matched distill-then-scan is real and reproducible, but it belongs to sequential self-training (classical active search, Garnett 2012 line), not to the spatial bandit. With an embedding in the classifier the layout is redundant: the cells are a coarsening of the same embedding. The bandit only pulls ahead when features are weak (cheap-only, 10% budget), which is SOFisher's regime, and even there the margin is within noise.
- The bandit does have lower variance at 10% (± 5 vs ± 19), which is worth a footnote, not a paper.

## Verdict, web text (D3, D4)
- The Sep 9 negative was an artefact of the regex label. Against an LLM judge for contextual sensitivity, the target clusters by domain (6.5x) and host (8.1x), the budget is real (7.8 docs/s), and cheap features are weak (AUC 0.65).
- What beats distill-then-scan at equal label budget is a refit-as-you-go auditor, by about 2x at 3 to 10% budgets, matching a classifier trained on 10x more labels. The spatial navigation that defines SOFisher is not the source of the gain.
- Supportable paper: "budgeted sensitivity auditing of pretraining corpora: sequential self-training beats fixed distillation", positioned against the FineWeb-Edu recipe, with active search and active learning as the method family. Not a SOFisher paper. Before claiming it: uncertainty-sampling active learning as a baseline, 20+ seeds, a second judge model and a 300-doc human check of the judge, and the run on the full 10M docs rather than a domain-stratified 73k.
- The measurement paper (D3) stands on its own: first published domain, host and embedding-cell concentration of judge-labelled sensitivity, quality, and loss on FineWeb, with the regex-vs-judge disagreement as the headline.

### R7. Does data influence have structure an adaptive prober could exploit? (runs/infl, D1)
Qwen3-0.6B-Base, 13,034 docs across 800 domains (up to 20 each), first-order influence = grad_doc . grad_ref for two 256-doc reference sets (random held-out FineWeb; held-out docs with edu score >= 3), 256 tokens per doc.

| quantity | ICC by domain | ICC by host | top-decile neighbor ratio (domain) | AUC of top decile from cheap features | R2 from cheap + edu + loss |
|---|---|---|---|---|---|
| influence (general ref) | 0.14 | 0.24 | 1.19x | 0.68 | 0.01 |
| influence (edu ref) | 0.14 | 0.24 | 1.17x | 0.65 | 0.00 |
| cosine influence (general) | 0.15 | 0.25 | 1.15x | 0.53 | 0.00 |
| gradient norm | 0.23 | 0.33 | 1.33x | 0.93 | 0.35 |
| loss (0.6B) | 0.43 | 0.51 | 1.66x | 0.67 | 0.86 |

Reading: loss and gradient norm are predictable and moderately domain-clustered, but the influence direction, which is what a MATES-style data influence model has to learn, is neither. Its variance is almost entirely within-domain and unexplained by cheap features, quality, or loss. There is no structure for an adaptive prober to exploit, so MATES's random hold-out probing is the right choice at this granularity. Caveats: first-order single-step influence on a fully pretrained 0.6B model, 256-doc references; MATES's oracle is measured mid-training on a 410M to 1B model, where influence may be more structured early on. Not worth a paper as a negative; worth one paragraph if the measurement paper (D3) is written.

## Final ranking, 2026-09-18

1. **Budgeted sensitivity auditing by sequential self-training (D4, reframed).** Positive, reproducible, honest about the mechanism: a refit-as-you-go auditor finds about 2x the judge-flagged docs of budget-matched distill-then-scan at 3 to 10% budgets, and matches a classifier with 10x the labels. SOFisher's spatial navigation is not the ingredient; classical active search is. Next: uncertainty-sampling baseline, 20+ seeds, full 10M docs, second judge, human check. Then, if it holds, the pretraining arm from SCOPE.md at 125M on hannah becomes worth running: filtered by sequential auditor vs distilled classifier at equal label budget, measure verbatim extraction of judge-flagged content.
2. **The measurement paper (D3).** Every number is in hand. Headline: regex PII and LLM-judged sensitivity disagree (2.4% precision), and only the judge label clusters (6.5x by domain, 8.1x by host). Fills the gap WIMBD, FineWeb and Dolma leave.
3. **Code secrets (D2).** Plausible, untestable on file-sampled shards; static triage by extension wins at 8x. Needs whole repos and a verified-secret label.
4. **Active probe selection for MATES (D1).** Negative at first order: influence has no exploitable structure. Drop unless a mid-training measurement shows otherwise.
5. Drop: active search for regex PII, runtime reduction by dropping data, domain-level online mixing.

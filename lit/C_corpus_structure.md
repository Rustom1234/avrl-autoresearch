# Topic C: Clustering of undesirable content in LLM pretraining corpora

## Source-by-source findings

**Documenting C4 (Dodge et al. 2021)** — https://arxiv.org/abs/2104.08758 (full text: https://ar5iv.labs.arxiv.org/html/2104.08758)
Measured: TLD- and domain-level composition of C4, plus effects of the "bad words" blocklist filter. Finding: content is highly concentrated at the top. `patents.google.com` is the single largest domain by token count in C4. The paper shows the top 25 TLDs by token share; `.mil` alone contributes 33.87M tokens from 58,394 unique URLs, and `.mod.uk` contributes 1.22M tokens from 2,873 URLs, out of "more than 365 million domains" touched overall. No Gini coefficient is reported. Separately, the paper shows the blocklist filter (used to strip "bad words" content) disproportionately removes text in dialects associated with minority identities — a quality-proxy failure mode, not a concentration statistic. The authors did not use the domain concentration finding for upweighting; it is presented as a documentation/transparency finding.

**FineWeb / FineWeb-Edu (HuggingFace, 2024)** — paper: https://arxiv.org/abs/2406.17557, dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, classifier: https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier
Measured: document-level educational-quality score via a classifier trained on 450k Llama3-70B-Instruct annotations (LLM judge as the expensive detector, distilled into a cheap BERT-style regressor, F1 ~82% at a threshold of 3). PII handling is regex-based: emails and public IPs are anonymized. Toxicity/NSFW handling is done "at the URL level," i.e. via a domain/URL blocklist, not a learned classifier. No domain-level or per-TLD breakdown of quality scores, PII rate, or toxicity rate is published in the paper (checked via https://ar5iv.labs.arxiv.org/html/2406.17557). UNVERIFIED: whether HuggingFace has released a separate domain-level quality-distribution analysis outside the paper; none was found.

**DCLM / DataComp-LM (2024)** — https://arxiv.org/abs/2406.11794
Measured: document-level fastText quality classifier, keeping only the top ~10% of documents by score (trained against OpenHermes-2.5 + ELI5 positives). This is a document-level cutoff, not a domain-level one. No domain concentration statistics found in the paper.

**RefinedWeb / Falcon (2023)** — https://arxiv.org/abs/2306.01116
Measured: MinHash fuzzy dedup + exact substring dedup, removing "nearly 90%" of raw Common Crawl content; filtering is mostly heuristic (not ML-based, aside from language ID). No domain- or host-level concentration statistics for the removed content were found.

**Dolma (AI2, 2024)** — https://arxiv.org/abs/2402.00159, blog: https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64
Measured: regex-based PII detection (emails, phones, IPs) across CC/C4/Reddit/GitHub subsets. Docs with 1-5 PII spans (0.02% of docs, 25 CC snapshots) get spans masked; docs with 6+ spans (0.001%) are dropped entirely, on the reasoning that dense-PII pages likely leak other PII too — a per-document severity threshold, not a domain-level statistic. For the GitHub subset, Dolma reports using separate tooling to detect/remove code secrets; the accessible text did not name the tool or its cost. UNVERIFIED: tool name, and whether it does live credential verification.

**The Stack / StarCoder PII (BigCode)** — https://huggingface.co/bigcode/starpii, https://github.com/bigcode-project/pii-lib, SantaCoder paper: https://arxiv.org/abs/2301.03988
Measured: PII detection in code (names, emails, keys, passwords, IPs, usernames) via the StarPII NER model. BigCode tried regex tools and commercial software first and found them inadequate, so they built a human-annotated PII dataset (reported size varies by page: ~20,961 or ~22,950 secrets; exact figure UNVERIFIED beyond "tens of thousands, human-annotated"). Reported detector performance: ~90% F1, beating regex baselines especially on secret keys. This is document/file-level detection applied uniformly; no repo- or org-level concentration statistic was found.

**"How Bad Can It Git?" (Meli, McNiece, Reaves, NDSS 2019)** — https://www.ndss-symposium.org/ndss-paper/how-bad-can-it-git-characterizing-secret-leakage-in-public-github-repositories/, PDF: https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_04B-3_Meli_paper.pdf
Measured: real-time scan of public GitHub commits (~6 months) plus a snapshot covering 13% of open-source repos, targeting private keys and 11 high-impact API key formats. Via secondary summary (https://blog.acolyer.org/2019/04/08/how-bad-can-it-git-characterizing-secret-leakage-in-public-github-repositories/): leakage affects over 100,000 repos, median ~1,793 new unique secrets/day, and 89.1% of candidate secrets were judged truly sensitive (not test/dummy) rather than false positives. UNVERIFIED: repo- or org-level concentration, and whether/how secrets were live-verified via API calls — PDF text was not extractable and the summary does not cover this. Treat as a gap, not a negative result.

**What's In My Big Data? (Elazar et al., ICLR 2024)** — https://arxiv.org/abs/2310.20707
Measured: corpus-level counts of duplicates, PII, toxicity, and benchmark contamination across ten corpora (C4, The Pile, RedPajama, LAION-2B-en, etc). Findings: ~50% of documents in RedPajama and LAION-2B-en are duplicates; PII counts are given per-corpus (e.g. ~201M extrapolated email addresses in mC4-en) but not broken down by domain. One domain-concentration number does appear, though for raw document share rather than undesirable content: `cdn.shopify.com` alone contributes almost 6% of all documents in LAION-2B-en. The paper explicitly does not cross-tabulate domain identity against PII/toxicity/duplicate rate — this is a real gap in the literature, not something I failed to find.

**Towards Safer Pretraining (IJCAI 2025)** — https://arxiv.org/abs/2505.02009
Measured: harmful-content rates in 1M sampled pages each from Common Crawl, C4, and FineWeb, using their own classifier ("HarmFormer," trained on a custom Topical/Toxic Prompt dataset). Results: toxic content at 4.1% (CC), 2.1% (C4), 3.9% (FineWeb); "topical" (contextually harmful but not slur-level) content at 3.44% (CC) up to 10.63% (FineWeb). This is corpus-level, not domain-level; no per-domain breakdown found.

**Going Over FineWeb with a Fine-Tooth Comb (2025)** — https://arxiv.org/abs/2508.21788
Elasticsearch tooling to search FineWeb-2 for problematic content by keyword hit. Explicitly a proof-of-concept: no domain aggregation, no neighbor-hit-rate or spatial-autocorrelation analysis, no concentration statistics.

**Organize the Web (Wettig et al. 2025)** — https://arxiv.org/abs/2502.10341
Proposes constructing topic/format "domains" via clustering to improve data mixing. Premise implies quality varies across constructed clusters, but no concrete concentration numbers were extractable from accessible text. UNVERIFIED beyond the qualitative premise.

## Answers

**(1) Where is domain/repo clustering strong vs weak?**
Strong, by construction: benchmark contamination and code secrets are inherently host/repo-concentrated because they originate from a small set of known benchmark-hosting sites (GitHub repos mirroring eval sets, Hugging Face dataset pages) and a small set of repos that leak credentials (Meli et al. find leakage concentrated enough to be "pervasive" across 100k+ repos, but I could not verify a top-k percentage). Weak or unmeasured: for toxicity, PII, and low educational quality on the open web, no source reviewed here published a domain-level Gini or top-k-percent statistic. FineWeb and Dolma both operationalize an assumption that toxicity clusters by domain (they filter at the URL/domain-blocklist level) but do not publish the measurement that would justify it. WIMBD explicitly does not cross-tabulate domain against PII/toxicity. This is a genuine literature gap, not something obscured by search failures.

**(2) Any published neighbor-hit-rate or spatial-autocorrelation analysis on a corpus?**
None found. UNVERIFIED / likely does not exist in the papers checked. The closest analogues are duplicate-cluster-size analyses (FineWeb's MinHash dedup, WIMBD's ~50% duplicate rate in RedPajama/LAION) but these measure exact/near-duplicate clustering of text content, not spatial autocorrelation of undesirable-content labels across a corpus graph (e.g. by domain adjacency or crawl proximity).

**(3) Which content types have expensive detectors in practice?**
LLM judge: FineWeb-Edu's quality label comes from Llama3-70B-Instruct annotations, later distilled into a cheap classifier (https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier). Towards Safer Pretraining trains its own harmful-content classifier rather than using a live LLM judge at scale. Human review: BigCode's StarPII required human-annotated ground truth after regex and commercial tools underperformed (https://github.com/bigcode-project/pii-lib). Live secret verification: Meli et al. distinguish "truly sensitive" (89.1%) from likely-test secrets, but whether this involved live API verification is UNVERIFIED from the sources I could access; this is the most likely candidate for "expensive, live-checked" detection in the literature but I cannot confirm the method.

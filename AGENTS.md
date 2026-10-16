# Rules for experiment agents in this repo

You are one of many agents working on the same question in parallel. These rules exist because parallel agents clobber each other's files, re-implement the same loader 100 times, and report numbers nobody can reproduce. Follow them exactly.

## Question being attacked

Can a budgeted "where to inspect next" policy (SOFisher-style active search) find and exclude more sensitive content from a pretraining corpus per detector call than uniform sampling or static triage? Full scope: `SCOPE.md`. Concept: `~/claude.ideas/sensitive-data-active-audit.md`.

## Fixed inputs (read-only, never modify, never re-download)

- `data/fineweb/sample/10BT/000_00000.parquet`: about 1M FineWeb docs. Columns: `text, id, dump, url, date, file_path, language, language_score, token_count`.
- `data/stack-smol/data/<lang>/*.parquet`: The Stack smol, code files. Columns include `content, lang, repository_name, path, licenses, size`.
- Python env: `.venv` (activate with `source .venv/bin/activate`). Has pandas, pyarrow, numpy, scipy, scikit-learn, umap-learn, sentence-transformers, presidio-analyzer, detect-secrets, tldextract, regex. Do not install packages; if you need one, say so in your report and stop.
- Shared loader: `aa/data.py`. Use it. Do not write your own parquet reader or regex PII detector; extend the module only through your own copy in your run dir if you must, and say so.

## Your working directory

`runs/<your-label>/` where `<your-label>` is the label you were given. Create it. Write everything there: scripts, logs, `result.json`. Never write anywhere else in the repo. Never touch another agent's dir. Never write to `data/` or `aa/`.

## Reproducibility

- Fix seeds (`--seed 0` default) and state the sample size n for every number.
- Subsample deterministically with `aa.data.load_fineweb(n=..., seed=...)`.
- Every reported number needs: the script that produced it (path), n, seed, and wall-clock time.
- If a run takes more than 20 minutes, cut n and say so. This machine has 20 cores, one RTX 5090, 30 GB RAM shared by many agents. Use at most 2 CPU processes and at most 6 GB RAM per agent; GPU use only if your task says so.

## Reporting

Return `result.json` in your dir AND as your final structured output. Numbers only if you ran the code. If something failed, report the failure; do not estimate what the number would have been. Never describe a plan as a result.

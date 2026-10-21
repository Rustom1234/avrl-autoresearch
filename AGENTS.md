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

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

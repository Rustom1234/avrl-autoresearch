# Role-fitness calibration (DESIGN.md section 9, PREREG H5)

`calibration.py` sends every candidate model a frozen set of role tasks, scores each reply by code, and turns the pass
rates into the routing table that arm D reads (`loop/runs/calibration/routing_table.json`). Nothing here calls a model
at import time, and the unit tests (`tests/test_calibration.py`) use a fake call function.

| File | What |
|---|---|
| `calibration.py` | `run(...)`, `build_routing_table(...)`, scoring, CLI |
| `calib_specs.json` | the frozen tasks: 24 coder specs, 6 repair, 6 planner inputs, 6 analyst inputs |
| `tests/test_calibration.py` | 85 tests, no model calls |

## Run it

From `/homes/rdubash/active-audit`:

    .venv/bin/python -m loop.core.calibration \
        --models sonnet haiku qwen35-9b gemma4-12b r1-qwen3-8b qwen35-4b --out loop/runs/calibration

Before launching, check the open servers (`curl -s http://cherry:8000/v1/models`, same for pine, willow, walnut). A
model whose health check fails is skipped and listed under "skipped (health check failed)"; run again later with the
same `--out` and only the missing work is done.

Options: `--classes constant schedule structural repair planner analyst` (default all), `--samples 2` (per spec),
`--limit N` (first N specs per class, for plumbing checks), `--rebuild` (no model calls: rebuild the table from
`<out>/results.jsonl`).

Plumbing check that costs a few cents (haiku only, one spec per class, one sample):

    .venv/bin/python -m loop.core.calibration --models haiku --samples 1 --limit 1 --out <scratch dir>

Outputs in `--out`: `results.jsonl` (one line per trial), `routing_table.json`, `routing_table.md`, `summary.json`,
`calls.jsonl` (the call layer's per-attempt log), `replies/<model>/<spec_id>.<sample>.txt` (raw replies, for reading
why a model failed).

Resume: a trial is identified by (model, spec_id, sample). Finished trials are skipped; trials that failed for
infrastructure reasons are retried. The table is built from everything in `results.jsonl`, so use a fresh `--out`
(or delete the file) to exclude old data.

## What is sent and how it is scored

Prompts are the production ones from `prompts.py` (program card as the system prompt; `coder`, `repairer_lint`,
`planner`, `analyst` user prompts; the same JSON schemas). Call settings follow `services.ModelService` (600 s
timeout, 4096 max tokens) and `arms.py` (`open_extra` sent to every open model). Differences on purpose:
`attempts=1` (first-try reliability, no hidden retries) and temperature 0.7 for open models (so the two samples
differ; production leaves it at the server default).

| Class | Tasks | A trial passes when |
|---|---|---|
| constant, schedule, structural | 8 specs each, written against `substrate/base/train_ampere.py` | the `edits_text` applies (`patch.apply_text`), the result compiles, `linter.lint` has no hard violation, the edit is not a no-op, every `target_symbols` entry appears in the changed lines, no `must_not_touch` symbol changed, and every `expect` regex holds (a leading `!` means it must not match) |
| repair | 6 files: the base plus a legitimate change plus one deliberate violation (import subprocess, `prepare.EVAL_TOKENS =`, `"val"` loader, seed line 137, `open()`, try/except around `evaluate_bpb`) | same as above, applied to the broken file with lint run against the base: the violation is gone and the legitimate change survived |
| planner | 6 results tables with a memo, N = 4 to 6 items | exactly N items, non-empty title and spec, unique titles, schema valid, no result-like number (4 or more decimals, e.g. a val_bpb or delta) that is absent from the prompt |
| analyst | 6 results tables | schema valid, non-empty memo, at most 300 words, every number in the memo appears in the prompt |

Details of the code checks:

- `target_symbols` and `must_not_touch` entries that look like identifiers match as whole words; anything else is a
  regex. A `must_not_touch` entry naming a top-level def, class, assignment or `Class.method` is compared by AST
  dump with the base (comment and whitespace edits do not count); other entries are searched in the changed lines.
  For repair specs the comparison is against the base, so a repair must restore the violated statement.
- Numbers are standalone numerals (identifiers like `e003` or `bf16` are ignored, sign is dropped, list numbering such
  as `1.` at a line start is ignored). The reference set is every number in the user prompt, which includes counts such
  as "Propose 5 distinct ideas" and "150 words".
- A reply from a different model than requested (the call layer fell back, for example to haiku) is not scored: the
  trial is recorded as infra (`fallback:<served>`), excluded from n, and retried on resume. Same for connection errors,
  timeouts, HTTP 5xx and context overflow. Schema failures and unparseable JSON count as failures.
- Every spec was checked against the machinery in the tests: each coder spec has a reference reply that must pass, a
  no-op must fail, and each repair file must trip its expected lint rule.

## Qualification

Wilson 95 percent lower bound of the pass rate: at least 0.6, structural at least 0.75. With the default 2 samples:

| Class | Trials | Passes needed |
|---|---|---|
| constant, schedule | 16 | 14 of 16 (lower bound 0.64; 13 gives 0.57) |
| structural | 16 | 16 of 16 (lower bound 0.81; 15 gives 0.72) |
| repair, planner, analyst | 12 | 11 of 12 (lower bound 0.65; 10 gives 0.55) |

The structural bar is therefore a clean sweep. Infra trials shrink n, which lowers the bound, so a model that was
mostly unreachable cannot qualify by accident.

## Routing table

`build_routing_table(results, prices)` (also writes the files when given `out_dir`):

- `coder_by_class[constant|schedule|structural]`, `repairer`, `planner_cheap` (repeated as `planner`, the key
  `arms.py` routes planner roles through), `analyst`: the cheapest qualifying model by nominal price. Sonnet is
  always a candidate, so it is the fallback when nothing cheaper qualifies. Ties on nominal price go to the lower
  measured cost per trial, then the name.
- `voter`: the cheapest qualifying planner-class model whose family differs from the planner's (families: qwen, gemma,
  deepseek for the R1 distill, claude). If there is no such model, the cheapest other qualifier, else the planner, and a
  note says so.
- `critic`: `opus`. `fallback`: `sonnet`.
- Extra keys (`qualified`, `evidence`, `notes`, `thresholds`, `calibration` with total cost and latency) are dicts or
  lists. Every top-level string value is a model name because `arms.py` health-checks all of them.

Prices come from `models.json` for open models. Claude entries have none there, so `CLAUDE_PRICES` in
`calibration.py` holds placeholders (1/5, 3/15, 15/75 per M tokens). They only order the Claude tiers, which is all
the table needs.

Freezing: DESIGN section 9 freezes the table in `models.json` and `runs/calibration/routing_table.json` before block 1;
this module only writes the second. Any change after launch is a deviation (PREREG).

## Cost and time of a real run

Per model 84 trials (24 coder specs x 2, plus 12 repair, 12 planner, 12 analyst). Measured in the plumbing check with
haiku: about $0.003 per coder or repair trial (prompt caching of the shared train.py prefix), $0.0014 planner,
$0.0012 analyst, 3 to 9 s per call. Prompts are about 9K tokens for coder and repair, 1.5K for planner and analyst.

- haiku: about $0.2 with caching, at most about $0.7 without.
- sonnet: about 3 times haiku, so $0.6 to $2.
- open models: nominal about $0.001 per trial, $0.1 per model (plan cost is 0).
- Total Claude spend: about $1.5, range $0.8 to $3. The plan cost and the nominal cost are both recorded.
- Wall time: Claude models a few minutes each (2 workers each, at most 4 Claude calls in flight); open models run 2
  workers per server, about 15 to 40 s per call, so 20 to 30 minutes for the 4B and 9B models. The R1 distill is the
  long pole: it can spend the whole 4096-token budget reasoning, so allow up to about an hour.

State on 2026-10-21 when this was written: cherry (qwen35-9b), willow (r1-qwen3-8b) and walnut (qwen35-4b) answered
`/v1/models`; pine (gemma4-12b) refused connections. A run now would skip gemma4-12b.

## Assumptions and limits

1. The prompts come from `prompts.py` as it was on 2026-10-21. Changing the card or a builder changes what is being
   calibrated; rerun (PREREG treats prompt changes after launch as deviations).
2. Coder passes measure patch mechanics and spec fidelity, not training quality. Nothing is executed on a GPU, and a
   SwiGLU that forgets an initialisation still passes unless the spec's `expect` regexes catch it. Several specs carry
   the traps that matter (1-D parameters must not reach Muon, new parameters need `init_weights` entries, the parameter
   count assertion must still hold).
3. `struct-8` differs from "drop value embeddings on layer 0": at depth 8, `has_ve` is true only on layers 1, 3, 5, 7, so
   layer 0 has none. The spec drops them from the earliest layer that has one.
4. `sched-8` ("start the warmdown at 60 percent") expects `WARMDOWN_RATIO = 0.4`, which tests the inversion. A model
   that rewrites `get_lr_multiplier` instead fails the regex even if its schedule is right.
5. The planner number check covers result-like numbers only (4 or more decimals). Planner specs propose new values such as
   "set MATRIX_LR to 0.05", so checking every number would fail every competent planner. The analyst check covers all
   numbers, as its prompt demands.
6. The analyst contract is the production one (a memo of at most 150 words), not a 5-line memo; the cap here is 300
   words so only runaway output fails.
7. Open models run at temperature 0.7 with `chat_template_kwargs.enable_thinking=false` on all of them (production
   sends it to every open model; servers whose template ignores it are unaffected). The R1 distill gets the same
   4096-token cap as production, so a long think block can fail it; raise `MAX_TOKENS` here and in the controller together
   if that is not wanted.
8. The 24 coder specs, repair violations and table inputs were written once and are frozen. The tables follow the
   ledger layout, with decisions from the DESIGN section 8 z rules and a fixed seed; the numbers are synthetic.
9. `substrate/base/train_ampere.py` contains one em dash character in a comment (line 602). It is part of the base file
   and is sent to models as is; none of the files written for this module contain one.

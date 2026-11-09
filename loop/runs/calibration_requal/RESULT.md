# Haiku requalification result (structural coder, lint repairer)

**Verdict: haiku stays the structural coder and stays the lint repairer.** Both cells passed on all 100 trials of the 50 new tryout items, and both clear their preregistered bars with room to spare.

Recommendation for the P2 coder table: **structural coder = haiku, lint repairer = haiku** (no cell goes to sonnet).

## What was run

- Preregistration written before any call: `PREREG.md` in this folder (decision rule, time, spec sha256).
- Spec file `specs_requal.json`, sha256 `6e5ff4d10cc827bbdf770e590a2625d73462978c0025de7b853275fcecef297a`, equal to `specs.sha256`; `validate_specs.py` printed ALL CHECKS PASS before the run.
- Run: `calibration.run(["haiku"], ["structural", "repair"], 2, "loop/runs/calibration_requal", specs=specs)`, the harness's own production prompts, one try per sample, no retries, abort not offered. 2026-11-05 23:05:40 to 2026-11-05 23:08:01 local time on spruce, about 2.4 minutes wall clock.
- 100 Claude calls (30 items x 2 + 20 items x 2), against a cap of 130 for this track. All 100 were served by haiku; 0 fallback hops; 0 infra trials (limit was 10 percent).
- Raw data in this folder: `results.jsonl` (one line per trial), `calls.jsonl` (one line per call), `replies/haiku/<item>.<sample>.txt` (every reply), and `result.json` (all numbers below).

## Decision (preregistered primary: new items only, item level)

k is the sum over items of the fraction of that item's 2 samples that pass; lower bound is `calibration.wilson_lower(k, n)`.

| cell | k / n | Wilson lower bound | bar | haiku stays? | goes to |
|---|---|---|---|---|---|
| structural coder | 30 / 30 | 0.886 | 0.75 | yes | haiku |
| lint repairer | 20 / 20 | 0.839 | 0.6 | yes | haiku |

For scale: structural could have failed up to 5 of its 60 trials (spread over different items) and still met the bar; repair could have failed up to 7 of 40. It failed none.

## Secondary (reported only, not used for the decision)

Pooled with the October trials (haiku passed every old item: structural 8 items / 16 trials, repair 6 items / 12 trials). The old cell was picked after seeing results across about 36 cells, so it is an optimistic sample; the pooled numbers are a sensitivity check.

| cell | level | k / n | Wilson lower bound |
|---|---|---|---|
| structural | new items only, trial level | 60 / 60 | 0.940 |
| structural | pooled, item level | 38 / 38 | 0.908 |
| structural | pooled, trial level | 76 / 76 | 0.952 |
| repair | new items only, trial level | 40 / 40 | 0.912 |
| repair | pooled, item level | 26 / 26 | 0.871 |
| repair | pooled, trial level | 52 / 52 | 0.931 |

Trial-level bounds treat the 2 samples of an item as independent, which they are not, so they are higher than the item-level bounds. The item-level bound is the honest one and is the one the decision used.

## Failure reasons

None. No trial failed, so there is no reason breakdown (no apply errors, no parse errors, no lint hits, no missing targets, no touched symbols, no failed expectations, no aborts).

## Checks that the clean sweep is real

A 100/100 result on items built to be hard deserves a second look. These were done after the run and none of them feed the decision.

- Independent rescoring: every saved reply was rescored from the files with `calibration.score_edit`, and all 100 outcomes match `results.jsonl` (0 mismatches).
- No leak path: `call.py` starts the claude CLI with `--tools ""`, `--strict-mcp-config` and `--setting-sources ""`, so haiku could not read `references.json` or any repo file. Only 28 of 100 replies equal the stored reference text, and 27 of 50 items got the same text on both samples, so the replies are not copies and are not all alike.
- Runtime smoke test (extra, CPU only, no GPU): the 60 structural replies were run through the same checks `smoke_references.py` applies to the references (finite init, finite loss, a gradient for every parameter, every parameter in exactly one optimizer group, one optimizer step, plain cross entropy on the evaluation path, per-item probes). 60 of 60 pass. This covers the gap that `score_edit` only reads text and syntax. Repair replies have no runtime test beyond the linter and the `expect` checks.

## Cost and latency (haiku, nominal dollars from `calls.jsonl`)

| cell | trials | total $ | mean $ per trial | mean latency s | median s | p90 s | max s |
|---|---|---|---|---|---|---|---|
| structural | 60 | 0.1861 | 0.0031 | 6.7 | 5.9 | 10.9 | 15.8 |
| repair | 40 | 0.0985 | 0.0025 | 4.0 | 4.0 | 5.0 | 6.1 |

Total for the 100 calls: $0.2846. About 80 percent of it is prompt cache writes (a least-squares fit of cost on token counts gives the shares: cache writes 80 percent, output 20 percent, cache reads 1 percent). This run wrote 1,131,513 cache tokens and read 275,479, about 11k written and 3k read per call, while October's 28 haiku calls on the same two cells wrote about 5k and read about 9k per call (warmer cache). October's haiku figures were $0.00185 per structural trial and $0.00140 per repair trial with a warm cache, so these costs are higher by roughly 1.7x each; sonnet was $0.0315 and $0.0414 in October, so haiku is still about 10x to 17x cheaper.

Structural, single block (rstruct-01 to 15): 30/30 pass, $0.0029 and 4.1 s per trial. Structural, multi block (rstruct-16 to 30): 30/30 pass, $0.0033 and 9.3 s per trial. Repair tiers (simple, dependent, entangled) all passed in full.

## What this does and does not show

- It shows haiku, given one try and the production prompt, writes valid structural edits (including 2 to 7 block edits that must stay consistent) and clean lint repairs on this held-out set. With 30 and 20 items, an all-pass result supports a true item-level pass rate above 0.75 (structural) and above 0.6 (repair); it cannot tell 0.95 from 1.0.
- The items were written blind to haiku output but not taken from an independent source, so they may suit Claude models better than an independently sourced set.
- `score_edit` is text and AST only. The runtime smoke test covers the structural replies; nothing equivalent exists for the repair replies.
- Abort is not tested here (`allow_abort=False`). In R1 D, 9 of haiku's 47 coder rows were aborts of the "already implemented" kind, and that behavior is outside this tryout.
- The tryout structural class (by the spec's `edit_class`) is harder and different from production routing, where in R1 D about 31 of 38 non-abort "structural" edits were hyperparameter-like. A pass here is a stronger statement than the production mix needs.
- The `haiku` alias is not pinned: `calls.jsonl` records only the registry name, so whether it resolved to the same model as on Oct 21 cannot be checked from the logs.

## Per-item pass fractions

Structural coder (30 items, all pass):

| item | title | passed | mean cost $ | mean latency s |
|---|---|---|---|---|
| rstruct-01 | MLP hidden width 3x | 2/2 | 0.0031 | 3.6 |
| rstruct-02 | Normalise both residual branches | 2/2 | 0.0031 | 3.6 |
| rstruct-03 | Parallel attention and MLP | 2/2 | 0.0029 | 4.0 |
| rstruct-04 | Remove the x0 skip | 2/2 | 0.0028 | 3.7 |
| rstruct-05 | Partial rotary embeddings | 2/2 | 0.0033 | 6.1 |
| rstruct-06 | Grouped-query attention | 2/2 | 0.0029 | 4.0 |
| rstruct-07 | Z-loss on the logits | 2/2 | 0.0030 | 4.4 |
| rstruct-08 | Fixed attention scale 0.12 | 2/2 | 0.0029 | 4.2 |
| rstruct-09 | Normalise the values | 2/2 | 0.0029 | 4.2 |
| rstruct-10 | Gradient clipping in the optimizer step | 2/2 | 0.0030 | 5.4 |
| rstruct-11 | Muon without Nesterov | 2/2 | 0.0028 | 3.6 |
| rstruct-12 | Uncautious Muon weight decay | 2/2 | 0.0029 | 3.9 |
| rstruct-13 | No aspect-ratio scaling of the Muon lr | 2/2 | 0.0028 | 3.1 |
| rstruct-14 | Per-channel x0 skip weights | 2/2 | 0.0029 | 4.0 |
| rstruct-15 | Orthogonal init for q, k, v | 2/2 | 0.0029 | 3.7 |
| rstruct-16 | Learnable gate per block | 2/2 | 0.0038 | 8.9 |
| rstruct-17 | Learned unigram logit bias | 2/2 | 0.0037 | 8.6 |
| rstruct-18 | Larger Muon lr for the MLP matrices | 2/2 | 0.0035 | 8.4 |
| rstruct-19 | Value-embedding gates on AdamW | 2/2 | 0.0038 | 9.5 |
| rstruct-20 | Gated attention per head | 2/2 | 0.0023 | 8.5 |
| rstruct-21 | U-Net skip connections | 2/2 | 0.0032 | 13.6 |
| rstruct-22 | One shared value-embedding table | 2/2 | 0.0041 | 10.6 |
| rstruct-23 | Zero-initialised linear before the head | 2/2 | 0.0026 | 10.6 |
| rstruct-24 | Multi-token prediction head | 2/2 | 0.0036 | 8.4 |
| rstruct-25 | Rotary base as a config field | 2/2 | 0.0022 | 7.4 |
| rstruct-26 | Muon beta2 as a hyperparameter | 2/2 | 0.0020 | 6.0 |
| rstruct-27 | Causal depthwise conv on the values | 2/2 | 0.0044 | 13.5 |
| rstruct-28 | Bias on the MLP input projection | 2/2 | 0.0040 | 11.1 |
| rstruct-29 | No MLP in the last block | 2/2 | 0.0023 | 8.0 |
| rstruct-30 | Learnable gain after the final norm | 2/2 | 0.0033 | 6.9 |

Lint repairer (20 items, all pass):

| item | linter rule and violation | passed | mean cost $ | mean latency s |
|---|---|---|---|---|
| rrep-01 | seed-lines: deletes the torch.cuda.manual_seed(42) line | 2/2 | 0.0030 | 3.2 |
| rrep-02 | forbidden-import: imports sys and prints the vocab size to sys.stderr | 2/2 | 0.0017 | 4.3 |
| rrep-03 | seed-other: calls torch.manual_seed(1234) before model.init_weights() | 2/2 | 0.0028 | 4.4 |
| rrep-04 | os-usage: creates a checkpoint directory with os.makedirs | 2/2 | 0.0027 | 3.1 |
| rrep-05 | forward-signature: adds a return_hidden argument to GPT.forward and returns the hidden state with the logits | 2/2 | 0.0028 | 4.1 |
| rrep-06 | protected-name: stores the startup time in a variable named time | 2/2 | 0.0028 | 3.8 |
| rrep-07 | module-alias: stores F as self.act in MLP.__init__ and calls self.act.relu in forward | 2/2 | 0.0029 | 4.6 |
| rrep-08 | dunder-access: prints type(model).__qualname__ | 2/2 | 0.0027 | 4.2 |
| rrep-09 | builtin-getattr: calls the MLP activation through getattr(F, "relu") | 2/2 | 0.0015 | 3.2 |
| rrep-10 | forbidden-call: sets rotary_seq_len with setattr | 2/2 | 0.0027 | 3.1 |
| rrep-11 | dunder-escape: prints config.__dict__ instead of asdict(config) on that line | 2/2 | 0.0028 | 3.5 |
| rrep-12 | attr-denylist: reads tokenizer.get_token_bytes and prints the mean bytes per token | 2/2 | 0.0028 | 4.2 |
| rrep-13 | protected-attr: monkeypatches nn.Linear.forward | 2/2 | 0.0029 | 4.7 |
| rrep-14 | string-literal: prints a note that mentions token_bytes | 2/2 | 0.0017 | 5.0 |
| rrep-15 | env-read: reads the harness variable AR_T0 and uses it for total_seconds | 2/2 | 0.0028 | 3.8 |
| rrep-16 | string-concat-escape: prints a header built by concatenating the literals "final ", "val " and "metrics follow" | 2/2 | 0.0016 | 3.8 |
| rrep-17 | prepare-import: adds EVAL_TOKENS to the prepare import line and prints it | 2/2 | 0.0029 | 4.4 |
| rrep-18 | evaluate-call: moves the evaluate_bpb call into a helper function final_eval | 2/2 | 0.0018 | 5.1 |
| rrep-19 | val-split: builds a second loader on the val split and draws a batch from it | 2/2 | 0.0029 | 4.1 |
| rrep-20 | hide-failure: wraps the num_params_M summary print in try/except | 2/2 | 0.0016 | 3.6 |

## One-line recommendation

P2 coder table: structural coder = haiku, lint repairer = haiku.

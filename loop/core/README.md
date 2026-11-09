# loop/core: the loop engine

`loop/core` is the engine of the routed research loop: it turns a model's reply into a candidate `train.py`, lints
it, sends it to the GPU queue, applies the arm's acceptance rule and keeps the ledger. This file documents the three
candidate-handling modules (patching, linting and log sanitising). The rest of the directory is covered by
`ENGINE.md` (the per-loop engine, arms A to D), `CONTROLLER.md` (the run controller and the confirmation block),
`CALIBRATION.md` (role-fitness calibration and the routing table) and `ANALYSIS.md` (the pre-registered analysis).
The designs and preregistrations they cite (DESIGN.md, PREREG.md) are in `loop/design/`.

## Patching, linting and log sanitising

Three independent modules (no cross imports, stdlib only) used by arms B, C, D (arm A uses only
`patch.py` and `sanitize_log.py`; DESIGN.md sections 5, 6, 10).

| File | Entry points |
|---|---|
| `patch.py` | `parse_blocks(text)`, `apply_blocks(source, blocks)`, `extract_full_file(text)`, `apply_text(source, reply)`, `apply_with_retries(source, reply)` |
| `linter.py` | `lint(new_source, base_source) -> LintResult(ok, hard, warnings, diff_lines_changed, edit_class)` |
| `sanitize_log.py` | `sanitize_tail(log_text, result=None, max_lines=25)` |

Run the tests (from the repository root):

    .venv/bin/python -m pytest loop/core/tests -q

`tests/fixtures/train_base.py` is a copy of Karpathy's `train.py` (autoresearch @ 228791f) with its one
em dash replaced by a hyphen. Nothing else is changed.

## The linter is a gate, not the wall

The linter spends no GPU time on files the harness would reject and stops obvious gaming early. The
security boundary is the harness tail (`substrate/harness_tail.py` plus the worker): pinned sha256 of
`prepare.py`, bytecode hash of `evaluate_bpb`, constants check, recomputed official `val_bpb`, sealed
shard, wall time. Every linter rule is a speed bump that a determined adversary can route around (for
example by building a path string with `"".join` or `%` formatting); the tail catches the effect.

`LintResult.hard` and `.warnings` are lists of `{rule, line, message}` (line 0 means whole file).
`ok` is `not hard`. `diff_lines_changed` counts, per difflib non-equal block, the larger side
(a replaced line counts once).

## Hard rules

Rule ids are the strings in `rule`. "ext" marks rules or set members that extend the original rule list.

| id | Fires on |
|---|---|
| `parse` | file does not parse (also absurd nesting) |
| `file-size` | more than 64 KB (65536 bytes); files over 1 MB are rejected without parsing |
| `forbidden-import` | `import` or `from ... import` (including dotted and `as` forms) of subprocess, socket, requests, urllib, http, shutil, signal, ctypes, importlib, multiprocessing, sys, pickle, inspect; ext: builtins, pathlib, io, glob, runpy (they reach open/exec without calling `open`) |
| `forbidden-call` | call or any reference (`f = open`) to open, exec, eval, compile, setattr, delattr, globals, locals, `__import__`, breakpoint; call to vars or input. `torch.compile` and other attribute calls are fine. |
| `os-usage` | any `os.X` with X not in {environ, path}; `from os import X` likewise; a bare reference to `os` (aliasing, `getattr(os, ...)`). Reads and writes of `os.environ` and all of `os.path` are allowed (the base itself writes `os.environ[...]`). |
| `protected-name` | assignment, augmented or annotated assignment, walrus, `for`/`with`/`except`/`match` capture targets, `del`, `global`/`nonlocal`, or a `def`/`class` of: MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, prepare, time, os, math, gc, torch, nn, F. Also an import that rebinds one of them to something else, and aliasing a guarded module (`import os as o`, `import torch as th`). |
| `protected-attr` | attribute assignment or `del` whose root is one of the names above or `kernels`, e.g. `prepare.EVAL_TOKENS = 1`, `math.pi = 3`, `F.relu = f`, `torch.Tensor.x = f`. Allowed: `torch.backends.<anything>` and `torch._dynamo.config.<anything>`. |
| `val-split` | the literal `"val"` as 4th positional or `split=` argument of any `make_dataloader` call |
| `string-literal` | any string (also f-string pieces, docstrings, bytes) containing, case-insensitively: shard_, sealed, .parquet, DATA_DIR, VAL_FILENAME, token_bytes, .cache/autoresearch |
| `prepare-import` | not exactly one module-level `from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb` (any order, one physical line, no `as`); any other import of `prepare` anywhere |
| `evaluate-call` | no call `evaluate_bpb(...)` outside every def, lambda and class |
| `seed-lines` | the module-level statements `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` are not each present unchanged |
| `seed-other` | any other call named manual_seed, manual_seed_all or seed (so `torch.manual_seed(137)` trips both this and `seed-lines`); any use of `torch.random` |
| `hide-failure` | a `try` (any handlers) or a `with ...suppress(...)` whose subtree contains the `evaluate_bpb` call or a `print` of the final summary (`---` or one of the nine `key:` lines) |
| `dunder-escape` (ext) | attribute access to `__dict__`, `__globals__`, `__code__`, `__builtins__`, `__closure__`, `__defaults__`, `__kwdefaults__`, `__subclasses__`, `__loader__`, `__spec__`; closes `evaluate_bpb.__globals__["EVAL_TOKENS"] = 1` |
| `attr-denylist` (ext) | any attribute access, load or store, whose attr is DATA_DIR, VAL_FILENAME, CACHE_DIR, TOKENIZER_DIR, BASE_URL, list_parquet_files, _document_batches, get_token_bytes, text_iterator, train_tokenizer, download_data or download_single_shard (prepare.py internals); also `getattr`/`hasattr` with one of them as literal name |
| `dunder-access` (ext) | a dunder (`__x__`) attribute or string subscript key (`d["__x__"]`) other than a load of `__init__`, `__call__`, `__name__`, `__class__`, `__version__`, `__doc__`, `__len__`, `__iter__`, `__next__`; so `__globals__`, `__wrapped__`, `__dict__`, `__code__`, `__builtins__`, `__import__`, `__subclasses__`, `__bases__`, `__mro__`, `__closure__` always fail, and so does storing to or deleting any dunder (`x.__class__ = C`, `x.__doc__ = s`). A bare name use of the forbidden ones and of `__loader__`/`__spec__` fails (`__loader__.get_data(path)` reads files without `open`). Defining a dunder method other than `__init__`, `__call__`, `__repr__`, `__str__`, `__hash__`, `__eq__`, `__enter__`, `__exit__`, `__len__`, `__getitem__`, `__iter__`, `__post_init__` fails (no `__getattr__`, `__setattr__`, `__getattribute__`, `__next__`, `__setitem__`, ...). Superset of `dunder-escape`, which is kept for id stability; both fire on `__globals__`. |
| `builtin-getattr` (ext) | `getattr` or `hasattr` whose attribute name is not a string literal (a missing name counts); `getattr` whose first argument is rooted at prepare, torch, time, os, math, F, nn or kernels even with a literal name; any reference to `getattr`/`hasattr` that is not the callee of a call (`g = getattr`, `map(getattr, ...)`). Literal names on other objects pass (`getattr(self, "x", None)`). |
| `string-concat-escape` (ext) | adjacent literal pieces of a `+` chain or an f-string (constant-only f-string parts count, `f"sha{'rd'}_"`) that together, case-insensitively, contain a `string-literal` text as a substring, or `val` as a standalone word (regex `\bval\b`: `"va" + "l"` and `"the " + "val" + " set"` fire; `"eval" + "uate"`, `"interval " + "x"`, `"val" + "_loss"` and `"train_" + "val"` do not, because `_` is a word character). Two or more pieces are needed (a lone `"val"` is not a concatenation), and a non-literal operand breaks a run (`"va" + x + "l"` passes). |
| `env-read` (ext) | a read of a harness variable: `os.environ["AR_..."]` as a load, or `os.environ.get/pop/setdefault` with a key whose literal prefix starts with `AR_` (the worker sets `AR_SEALED_DIR` and `AR_T0`). Keys are folded through `+` chains and f-strings. Writes and `del` of any key, and reads of other keys, stay allowed (the base writes `os.environ[...]`); `os.getenv` is still `os-usage`. |
| `forward-signature` (ext) | the signature of `GPT.forward` (last `def forward` in the module-level class `GPT`) differs from the base's, currently `(self, idx, targets=None, reduction='mean')`, or `GPT` is gone. The harness scores with `model(x)` (targets=None), so a changed signature nulls the leak check. Compared on argument names, order and kind (positional-only, normal, `*args`, keyword-only, `**kwargs`) and default values only: annotations, quote style and number formatting (`1e-3` vs `0.001`) are ignored, a non-literal default is compared by its AST. |
| `module-alias` (ext) | a bare reference to time, math, gc, torch, nn, F, prepare or kernels not used as `X.attr` (`t = torch; t.foo = 1`) |

The `H100_BF16_PEAK_FLOPS` line may change freely: no rule looks at it and `edit_class` ignores it.

## Warnings (recorded, not blocking)

| id | Fires on |
|---|---|
| `diff-large` | more than 300 changed lines vs the base |
| `new-import` | an import (any depth) not present in the base, compared without `as` names |
| `batch-divisibility` | `DEVICE_BATCH_SIZE`, or a literal batch-size argument of `evaluate_bpb`, that is not a positive divisor of 10240 (= EVAL_TOKENS // MAX_SEQ_LEN, constants folded from small int expressions). Floor division then scores fewer tokens. |
| `training-branch` | more uses of the `.training` attribute than the base |
| `mode-switch` | a `.eval()` or `.train()` call whose (source text, enclosing defs, inside-a-loop) signature is not in the base |
| `seed-comment` | the two seed lines are present (the hard rule passes) but the physical line text differs from the base's, for example a trailing comment was added or changed. Harmless, recorded only. |
| `split-nonliteral` (ext) | `make_dataloader` split is not a literal, so `val-split` cannot check it |

## edit_class

Computed from top-level statements, ignoring comments and `H100_BF16_PEAK_FLOPS`. Statements are
diffed with difflib on `ast.dump`; paired changes are labelled by symbol. In this order:

1. `schedule`: every changed symbol is `get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay`
   or a constant whose name starts with WARMUP, WARMDOWN, FINAL_LR.
2. `constant`: every changed symbol is an UPPERCASE constant of the base's hyperparameter block
   (the run of top-level UPPERCASE assignments after `# Hyperparameters`) whose new value is
   side-effect free (constants, names, tuples, lists, unary and binary operators; no calls).
   A schedule constant together with another constant is therefore `constant`.
3. `structural`: any added, removed or changed class or def other than the three schedule functions.
4. `mixed`: everything else (training loop edits, new top-level code, a new constant, computed
   constant values, a schedule function plus a constant, a comment-only or no-op edit).

## What the harness tail covers that the linter cannot

Issue karpathy/autoresearch#599 lists four ways to move `val_bpb` without a better model.

| Path | Linter | Harness tail |
|---|---|---|
| path 1: skip the optimizer but finish the budget | NOT caught by the linter (legal Python touching no protected name; tested as documentation in `test_599_path1_*`) | The decision metric is the recomputed official `val_bpb` of the real final model, so an untrained model just scores badly. Paired with a faked printed number it trips the printed-minus-official check (1e-4), and the final parameter hash differs from a run that trained. Faking `total_training_time` to run long trips the 500 s wall-time check. |
| path 2: replace or wrap the metric | Caught: redefinition or assignment of `evaluate_bpb` (`protected-name`), `__code__`/`__globals__` tricks (`dunder-escape`), removed or def-only call (`evaluate-call`), `try`/`suppress` (`hide-failure`), changed import line (`prepare-import`). NOT caught: leaving the real call and overwriting `val_bpb = 0.05` afterwards. | Official value recomputed by the tail from the same model; mismatch flag; bytecode hash of `evaluate_bpb`. |
| path 3: shrink or swap the val set | Caught: attribute assignment on `prepare` (`protected-attr`, `prepare-import`), `sys.modules` (`forbidden-import`), path strings (`string-literal`), `make_dataloader(..., "val")` (`val-split`). Warned only: eval batch size that does not divide 10240. NOT caught: path strings assembled with `join`, `%`, `format` or computed pieces (adjacent literals joined with `+` or an f-string are caught by `string-concat-escape`). | `constants_ok` checks TIME_BUDGET, EVAL_TOKENS, MAX_SEQ_LEN; sealed shard gap exposes val contamination and swapped shards. |
| path 4: printed numbers never bound to code, data, model | NOT caught (no rule about what is printed; only `hide-failure` around the final print block). Tested as documentation in `test_599_path4_*`. | `harness_result.json` binds the official number to the candidate (sha256 of `train.py`, parameter hash and count, wall time). |

### Gaps neither side currently closes (found while writing the rules)

- The tail takes the eval batch size from the candidate's globals (`_g.get("DEVICE_BATCH_SIZE") or 32`),
  so a non-dividing `DEVICE_BATCH_SIZE` also shortens the official eval. The linter only warns. The tail
  should pin the batch size (128) or the worker should treat the warning as an integrity flag.
- The tail evaluates the candidate's own `model` object, and `evaluate_bpb` passes the targets into
  `model(x, y, reduction='none')`. A model class that peeks at `targets` inside `forward` (or branches on
  `self.training`) scores well on both official and sealed shards. This is a fifth path, not in issue
  599. The linter pins the `GPT.forward` signature (`forward-signature`) and warns on `.training` and
  new `.eval()/.train()`, but it cannot see what the body does with `targets`. A cheap harness check: on one
  val batch compare `model(x, y, reduction='none')` with the loss computed from `model(x)` logits.
- The tail runs in the candidate's process, so process-wide state it shares with the candidate
  (monkeypatched library classes imported by name, such as `from torch.nn import Linear` then
  `Linear.forward = ...`) is not covered by the attribute rule, which only knows the listed roots.
  Re-evaluating the saved weights in a fresh process would close this.
- Rules reason about literal syntax. `if False: val_bpb = evaluate_bpb(...)` satisfies `evaluate-call`
  (the printed-vs-official check catches the effect), and a string built with `join`, `%`, `format` or
  from non-literal pieces dodges `string-literal` and `string-concat-escape`.
- `env-read` only sees literal keys. `os.environ.items()`, `dict(os.environ)`, `e = os.environ; e[...]`
  and a key held in a variable (`k = "AR_T0"; os.environ[k]`) read harness variables unnoticed. The
  worker could drop the `AR_` variables from the candidate's environment once the tail has read them.
- `forward-signature` does not see `GPT.forward = other`, instance-level `model.forward = ...` or a
  subclass that overrides `forward` or `__call__` (defining `__call__` is allowed). `operator.attrgetter("...")`
  is an unchecked cousin of `getattr`.

## patch.py

- Format: aider style `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` (5 to 9 marker characters,
  leading whitespace and CRLF tolerated). Prose and code fences outside blocks are ignored.
  `parse_blocks` returns `[(search, replace)]`, each string with every line terminated by `\n`
  (so `""` is zero lines, `"\n"` one blank line). Malformed blocks raise `ValueError`.
- `apply_blocks` matches whole lines. Exact first; only if that finds nothing, a match ignoring
  trailing whitespace per line (leading whitespace is never relaxed). Each SEARCH must match exactly
  once; blocks are applied in order against the text produced by earlier blocks. An empty SEARCH is
  rejected. On any failure the original source comes back and the report names the block.
- Report keys: `mode`, `ok`, `n_blocks`, `failed_block` (0-based), `reason` (`no match`,
  `multiple matches (k)`, `empty SEARCH`, ...), `message` (1-based, ready to show a model),
  `fallback_blocks` (0-based indices that needed the whitespace fallback).
- `apply_text(source, reply)` is the base entry point for a raw model reply. If any SEARCH marker is
  present it is search/replace mode. Otherwise a reply holding exactly one python (or untagged) fenced
  block of more than 400 lines is accepted whole with `mode: full_file`; a truncated (unclosed) fence is
  never accepted. The full-file text is not parsed here; the linter does that.
- `apply_with_retries(source, reply)` is `apply_text` plus a syntax check (`ast.parse` of the result, full-file
  replies included) and a classified failure for a retry prompt. No model calls; the caller loops. It returns
  `(new_source, report)`, report being `apply_text`'s plus:
  - `failure_class`: `none`, `parse` (unusable reply: no blocks and no full file, malformed block, empty
    SEARCH), `no_match`, `multi_match`, or `syntax` (the edit applied but the result does not parse).
  - `retry_hint`: short text for the model (empty on success). `no_match`: block number, its first two
    SEARCH lines, and the nearest source line (`difflib.get_close_matches` on the first non-blank SEARCH
    line, cutoff 0.6) with its line number and indentation. `multi_match`: the first two SEARCH lines and
    up to five matching line numbers. For a block after the first, the source searched is the text after
    the earlier blocks, and the hint says so. `parse`: the parse message plus the expected format.
    `syntax`: the parser message and the offending line. Each quoted line is cut to 120 chars.
  - On any failure the original source is returned.

## sanitize_log.py

`sanitize_tail(log_text, result=None, max_lines=25)`. Rationale: harness numbers are authoritative, and
a fix attempt needs the traceback. `run.log` is produced by code the loop's coder model wrote, so it is a
prompt-injection channel into the model's next prompt (karpathy/autoresearch issue 64), and a printed
`val_bpb` is spoofable while the official one is recomputed by the harness tail.

- `result` given (the worker's `result.json` dict):
  - status `crash` or `timeout`: the last `max_lines` (25) raw log lines, `\r`-separated fragments split
    into lines, ANSI and control characters stripped, each line cut to 300 chars. Nothing else is
    filtered, so traceback frames and library messages arrive intact. The caller states the status.
  - any other status: the log text is not used. A summary block is rebuilt from the result:
    `status`, `official_val_bpb` (`none` when missing or tampered), `training_seconds`, `total_seconds`,
    `peak_vram_mb`, `num_steps`, `num_params_M` (from `printed`, only when numeric), and `flags`
    (`mismatch_flag`, `target_leak_flag`, `harness_missing`, `overbudget`, `load_flagged` when truthy,
    `canary_failed` when `canary_ok` is false, plus entries of a `flags` list that look like identifiers,
    `none` if empty). The printed `val_bpb` (in the log or in `printed`) is never shown.
- `result=None` (no worker): the old allowlist, minus the imperative-phrase filter. The nine summary
  lines and `harness_official_val_bpb`, each only as `key: <one number>` (last occurrence of each key);
  lines containing Traceback, error (any case), OOM, out of memory; and the 15 lines before the first
  Traceback. If more than `max_lines` qualify, summary lines win, then error lines, then context, newest
  first, and output keeps log order. Stack frame lines are not kept unless they contain "error".
- The imperative-phrase filter (ignore, you must, please, ...) is removed everywhere: it dropped
  legitimate error text and any paraphrase beat it. Free text is limited by construction instead
  (rebuilt block, or allowlist), and the crash tail is the one place raw text is shown because the fix
  attempt needs it.

The output is for the model's eyes only and the crash tail is spoofable. No accept/reject decision may
use it; decisions use `harness_result.json`.

## Assumptions

- Base is the current autoresearch `train.py`; schedule function names, the `# Hyperparameters` marker
  comment and `H100_BF16_PEAK_FLOPS` are taken from it. A different base needs `SCHEDULE_FUNCS` (and
  possibly the marker logic) updated in `linter.py`.
- `EVAL_TOKENS // MAX_SEQ_LEN = 10240` (pinned values from `prepare.py`, which the tail also checks).
- Python 3.9 or newer for `ast.unparse`, `end_lineno`; tested on 3.12.
- The harness line format is `harness_official_val_bpb: <number>` (matches `harness_tail.py`).

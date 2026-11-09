# Requalification items: haiku as structural coder, haiku as lint repairer

These are 50 new, held-out tryout items for the two routing-table cells that rest on only 8 structural and 6 repair items (all passed by haiku in the October calibration). The qualification procedure is described in `loop/core/CALIBRATION.md`. Nothing here calls a model.

| File | What it is |
|---|---|
| `specs_requal.json` | the items: a `coder` list (30 structural) and a `repair` list (20), same schema as `loop/core/calib_specs.json`, plus `version` and `base` |
| `references.json` | one passing reply per item (`coder`, `repair`), plus hand-written wrong edits that must fail (`negative_coder`, `negative_repair`) and valid alternative replies that must pass (`alternative_coder`, `alternative_repair`) |
| `validate_specs.py` | the acceptance checks (no torch needed) |
| `smoke_references.py` | optional extra: runs every structural reference on a tiny model on the CPU |
| `specs.sha256` | sha256 of `specs_requal.json`, taken before any model call (check with `cd loop/runs/calibration_requal && sha256sum -c specs.sha256`) |

A reference is the `edits_text` string of the production reply (SEARCH/REPLACE blocks). The full reply is the JSON `{"description": ..., "edits_text": <reference>}`.

## Written blind

The items were written without opening `loop/runs/calibration/replies/` or any results file, so they do not depend on what haiku did on the old items. The only haiku information in hand was the aggregate in `loop/runs/calibration/routing_table.md` (it passed every old structural and repair item). The items were authored within this project rather than taken from an independent source, so they may suit Claude models better than an independently sourced set would. The review in the last section was also done without opening any haiku output.

## Structural items (30, all `edit_class: structural`)

15 touch one function or one block and need one SEARCH/REPLACE block (`rstruct-01` to `rstruct-15`). 15 span several places and need 2 to 7 blocks (`rstruct-16` to `rstruct-30`): a new parameter, its `init_weights` line, an optimizer group and the parameter-count assertion where they apply. None repeats the old eight (SwiGLU, QK-norm removal, per-head q scale, GELU, value-embedding weight decay, softcap, smear gate, drop value embeddings). Families are `architecture` (22) and `other` (8), so `scheduler.FAMILY_CLASS` routes all of them to the structural coder. There is no `data` item, because the data loader lives in the read-only `prepare.py`.

Single block:

| id | title | what makes it tricky |
|---|---|---|
| rstruct-01 | MLP hidden width 3x | two coordinated lines in `MLP.__init__` (`c_fc` and `c_proj`) |
| rstruct-02 | Normalise both residual branches | two lines in `Block.forward` |
| rstruct-03 | Parallel attention and MLP | the MLP must stop seeing the attention output |
| rstruct-04 | Remove the x0 skip | one-line ablation, the parameter stays and gets no gradient |
| rstruct-05 | Partial rotary embeddings | rewrite `apply_rotary_emb`, cos and sin must be sliced to match |
| rstruct-06 | Grouped-query attention | one line in `build_model_config` |
| rstruct-07 | Z-loss on the logits | extra loss only on `reduction == 'mean'`, evaluation must stay plain cross entropy |
| rstruct-08 | Fixed attention scale 0.12 | keyword added to the flash attention call |
| rstruct-09 | Normalise the values | `v = norm(v)` at method level, outside the `if ve` block |
| rstruct-10 | Gradient clipping in the optimizer step | new code at the top of `MuonAdamW.step` |
| rstruct-11 | Muon without Nesterov | one line inside the fused Muon function |
| rstruct-12 | Uncautious Muon weight decay | delete the mask line and keep the decay term |
| rstruct-13 | No aspect-ratio scaling of the Muon lr | one line in `_step_muon` |
| rstruct-14 | Per-channel x0 skip weights | only the allocation changes, broadcasting does the rest |
| rstruct-15 | Orthogonal init for q, k, v | three init calls in `init_weights` |

Multi block (blocks in the reference):

| id | title | blocks | what makes it tricky |
|---|---|---|---|
| rstruct-16 | Learnable gate per block | 6 | new parameter, init, forward, list, assertion, AdamW group |
| rstruct-17 | Learned unigram logit bias | 6 | same six sites, group uses the unembedding lr |
| rstruct-18 | Larger Muon lr for the MLP matrices | 2 | split `matrix_params` and the Muon group loop, assertion must stay true |
| rstruct-19 | Value-embedding gates on AdamW | 3 | exclude exactly the `ve_gate` tensors from Muon, extend the assertion |
| rstruct-20 | Gated attention per head | 3 | gate applied between the flash call and the flatten, zero init so the gate starts at 1 |
| rstruct-21 | U-Net skip connections | 6 | stack of block outputs, index arithmetic, odd or even depth |
| rstruct-22 | One shared value-embedding table | 7 | every use of `self.value_embeds` (init, init_weights twice, estimate_flops, num_scaling_params, setup_optimizer, forward) must move to one table, and only `has_ve` layers may get it |
| rstruct-23 | Zero-initialised linear before the head | 4 | a 2-D matrix outside `transformer.h` must be routed to Muon by hand |
| rstruct-24 | Multi-token prediction head | 4 | auxiliary loss only on the `'mean'` path, shifted targets, extra head in the lm_head group |
| rstruct-25 | Rotary base as a config field | 5 | five sites; the rotary tables are computed twice and `init_weights` silently resets the base if missed |
| rstruct-26 | Muon beta2 as a hyperparameter | 4 | a new constant, a new `setup_optimizer` keyword, the Muon group line and the module-level call must all be wired |
| rstruct-27 | Causal depthwise conv on the values | 6 | 3-D weight inside `transformer.h`, identity init, left padding |
| rstruct-28 | Bias on the MLP input projection | 5 | uninitialised bias on the meta device, 1-D tensors would reach Muon |
| rstruct-29 | No MLP in the last block | 3 | three places must agree on `mlp is None` |
| rstruct-30 | Learnable gain after the final norm | 4 | extend an existing group so the assertion holds unchanged |

The multi-block specs name the new lists and parameters (for example `gate_params`), so each site has its own `expect` regex.

## Repair items (20)

Each broken file is the base plus a legitimate edit plus one deliberate violation. The repairer sees the same prompt as in production (`prompts.repairer_lint`), and a pass needs the violation gone, the legitimate edit kept, and the `expect` regexes true. Tiers: simple (remove or restore one thing, legitimate edit elsewhere), dependent (the violation spans statements that refer to each other, so the repair must clean up all of them), entangled (the legitimate edit and the violation sit in the same function or line, so restoring the base version would lose the legitimate edit).

| id | primary rule | also triggered | intended change (must survive) | violation | tier |
|---|---|---|---|---|---|
| rrep-01 | seed-lines | | SCALAR_LR 0.3 | deletes the `torch.cuda.manual_seed(42)` line | simple |
| rrep-02 | forbidden-import | | UNEMBEDDING_LR 0.006 | `import sys` and a print to `sys.stderr` | dependent |
| rrep-03 | seed-other | | ADAM_BETAS (0.9, 0.95) | `torch.manual_seed(1234)` before `init_weights` | simple |
| rrep-04 | os-usage | | TOTAL_BATCH_SIZE 2**18 | `os.makedirs("ckpt")` | simple |
| rrep-05 | forward-signature | | softcap 20 | extra `return_hidden` argument and return in `GPT.forward` | entangled |
| rrep-06 | protected-name | | ASPECT_RATIO 72 | a variable named `time` | simple |
| rrep-07 | module-alias | | `relu(x).pow(3)` in the MLP | `self.act = F` and `self.act.relu(...)` | entangled |
| rrep-08 | dunder-access | | ema_beta 0.95 | `type(model).__qualname__` | simple |
| rrep-09 | builtin-getattr | module-alias | momentum warmup 400 steps | `getattr(F, "relu")` | simple |
| rrep-10 | forbidden-call | | `rotary_seq_len = sequence_len * 5` | the same line written with `setattr` | entangled |
| rrep-11 | dunder-escape | dunder-access | depth added to the config log line | `config.__dict__` on that line | entangled |
| rrep-12 | attr-denylist | | x0_lambdas init 0.2 | `tokenizer.get_token_bytes(...)` and a print that uses it | dependent |
| rrep-13 | protected-attr | | DEVICE_BATCH_SIZE 32 | monkeypatches `nn.Linear.forward` | simple |
| rrep-14 | string-literal | | HEAD_DIM 64 | a print whose text mentions `token_bytes` | simple |
| rrep-15 | env-read | | fast-fail threshold 50 | reads `AR_T0` and uses it in `total_seconds` | dependent |
| rrep-16 | string-concat-escape | | WARMDOWN_RATIO 0.6 | `"final " + "val " + ...` | simple |
| rrep-17 | prepare-import | | lm_head init std 0.002 | `EVAL_TOKENS` added to the prepare import and printed | dependent |
| rrep-18 | evaluate-call | | gc interval 2000 steps | `evaluate_bpb` moved into a helper function | dependent |
| rrep-19 | val-split | | ns_steps 4 | second loader with `split="val"` and a `next()` on it | dependent |
| rrep-20 | hide-failure | | ve_gate_channels 24 | try/except around the `num_params_M` summary print | simple |

### Linter rule coverage

The 20 specs have 20 distinct primary rules. These are every hard rule in `loop/core/linter.py` that can reach the lint-repair step: `seed-lines`, `forbidden-import`, `seed-other`, `os-usage`, `forward-signature`, `protected-name`, `module-alias`, `dunder-access`, `builtin-getattr`, `forbidden-call`, `dunder-escape`, `attr-denylist`, `protected-attr`, `string-literal`, `env-read`, `string-concat-escape`, `prepare-import`, `evaluate-call`, `val-split`, `hide-failure`.

Fourteen of them were not in the old six repair items. The old six rules (`forbidden-import`, `protected-attr`, `val-split`, `seed-lines`, `forbidden-call`, `hide-failure`) appear again here with different surface forms.

Left out on purpose: `parse`, because `arms._apply` compiles every edit and bounces syntax errors back to the coder, so the linter never sees an unparseable file; and `file-size`, because a file over 64 KB needs about 40 KB of padding that a repairer could not delete with a SEARCH block.

Every legitimate edit is on a base line that no old repair spec touches.

## Difficulty split

| Cell | Split |
|---|---|
| structural | 15 single block, 15 multi block (2 to 7 blocks, 68 blocks in total) |
| repair | 10 simple, 6 dependent, 4 entangled |

Compared with production, the structural items are harder than the usual structural route: in R1 D, 31 of haiku's 38 non-abort "structural" edits were classed constant by the linter. Abort is not tested (`allow_abort=False`), and the tryout gives one try, with no free retries.

## How the items were validated

`validate_specs.py` (run: `.venv/bin/python -B loop/runs/calibration_requal/validate_specs.py`) passes all checks:

1. The specs load with `calibration.load_specs`, have 30 and 20 items, the same keys as `calib_specs.json`, valid regexes, and `calibration.tasks` builds all 50 production prompts.
2. Ids are unique, new, and each has a reference.
3. Coder: every reference passes `score_edit(BASE, BASE, ref, spec)` with no reasons and the linter classes it `structural`. A no-op fails. All 66 hand-written wrong edits fail (at least one per item) and all 21 hand-written valid alternatives pass. Dropping any one of the 68 blocks of the multi-block references makes the reference fail (68 of 68 rejected).
4. Repair: the broken file triggers `expect_rule`, the legitimate edit alone lints clean, the reference repair passes `score_edit(broken, BASE, ref, spec)`, an empty repair fails, throwing away the legitimate edit fails, dropping any block of the five two-block repairs fails, all 17 hand-written wrong repairs fail and all 13 hand-written valid alternative repairs pass.
5. No spec text, title, legitimate-edit line or violation equals an old one; the most similar coder pair is 0.35 by difflib ratio (limit 0.80). Text similarity is a weak test, so near-duplicates were also judged by hand (see the review section).
6. No em dash in the new files, and `specs.sha256` matches.

Extra checks done at authoring time:

- `smoke_references.py` applies each structural reference, stubs flash attention with torch SDPA, builds a tiny GPT on the meta device the way `train.py` does, and checks: all parameters finite after `init_weights` (they start NaN), finite loss and a gradient for every parameter, every parameter in exactly one optimizer group with Muon getting only same-shape 2-D tensors, one optimizer step, a second forward, and `reduction='none'` equal to plain cross entropy. Each item also has a probe (for example the rotary table really uses base 50000 after `init_weights`). It also calls `num_scaling_params` and `estimate_flops`, as `train.py` does right after `init_weights`. Result: 30 of 30 pass, and it catches 66 of the 68 dropped-block variants (the misses are the redundant `__init__` call in `rstruct-25`, which `init_weights` overwrites, and the module-level `setup_optimizer` call in `rstruct-26`, which the smoke test does not run; the scorer rejects both). Run it from the repo root with `.venv/bin/python -B loop/runs/calibration_requal/smoke_references.py`. It needs torch and hides the GPU.
- The valid alternatives stored in `references.json` (other variable names, keyword arguments, two-statement forms, `.data` writes, a helper function) are accepted by the scorer, and every coder alternative also passes the smoke test, so the regexes do not demand one exact spelling.

What the checks cannot see: `score_edit` is text and AST only, so a reply that satisfies every regex but is wrong at run time (for example a wrong index in the U-Net skip) still passes. Only the references are checked at run time. A few valid solutions with an unusual spelling could still be rejected by a regex.

## Running the requalification (the run itself is reported in `RESULT.md`)

`calibration.py` has no `--specs` flag, so use a short wrapper. Use the shared lock and breaker variables so the 4-call cap holds across concurrent processes, and never use `--out loop/runs/calibration`.

    AR_LOCK_DIR=<shared lock dir> AR_BREAKER_FILE=<shared breaker file> .venv/bin/python -B - <<'EOF'
    import sys
    sys.path[:0] = ["loop/core", "loop/models"]
    import calibration as C
    specs = C.load_specs("loop/runs/calibration_requal/specs_requal.json")
    C.run(["haiku"], ["structural", "repair"], 2, "loop/runs/calibration_requal/haiku_run", specs=specs)
    EOF

That is 60 structural and 40 repair trials, 100 haiku calls, about $0.15 to $0.25 as estimated before the run. Check the sha256 first (see the file table); it must be the version 2 hash in `specs.sha256`. The decision rule in `PREREG.md` (the qualification rule of `loop/core/CALIBRATION.md`) applies unchanged: item-level Wilson lower bound with `calibration.wilson_lower(k, n)`, where k is the sum of per-item pass fractions and n is the number of items, at least 0.75 for structural and at least 0.6 for repair.

## Review (2026-11-05)

A second review pass checked every item without opening any haiku output and without model calls. Plausible wrong edits and valid alternatives were written by hand and run through `calibration.score_edit` and, for coder items, through the smoke test. The items below were then fixed. All of those edits are now stored in `references.json` and checked by `validate_specs.py`. `version` went from 1 to 2 and `specs.sha256` was rewritten. No haiku run existed in this folder when the items changed.

Replaced as near-duplicates of old items (text similarity was low, but the idea and the edit sites were the same):

| id | was | why replaced | now |
|---|---|---|---|
| rstruct-22 | Previous-token embedding | same idea as old `struct-7` (smear gate): mix the previous token into the embedding stream at the same place (after `x = norm(x)`, before `x0 = x`) with the same `F.pad` shift, zero init and own AdamW group | One shared value-embedding table (7 blocks) |
| rstruct-26 | LayerScale on both branches | same recipe as old `struct-3` (per-head q scale): ones-initialised 1-D scale inside `transformer.h`, filled with 1.0, moved from Muon to an AdamW group with `lr=scalar_lr`, excluded from `matrix_params` | Muon beta2 as a hyperparameter (4 blocks) |

Wrong edits that the old regexes accepted, now rejected:

| id | wrong edit | change |
|---|---|---|
| rstruct-03 | sequential block that recomputes `h = norm(x)` after the attention update | negative regex: no renormalised x after the attention update in `Block.forward` |
| rstruct-05 | rotates the first half of the channels but never concatenates the untouched second half (crashes at run time) | the final `torch.cat` must carry a third part or the pass-through slice |
| rstruct-11 | also deletes the momentum-buffer update | expect that line to stay |
| rstruct-14 | makes `resid_lambdas` per-channel too | expect `resid_lambdas` unchanged |
| rstruct-15 | `orthogonal_(..., gain=s)`, or the MLP `c_fc` made orthogonal too | default gain only; MLP and `c_proj` init lines must stay |
| rstruct-16 | gate multiplies the block input; `fill_(1.5)` | gate must multiply a difference written inside parentheses; exact 1.0 fill |
| rstruct-17 | bias added after the softcap | bias line must come before the `tanh` softcap |
| rstruct-18 | the 1.5x lr on the attention groups; attention matrices dropped from the optimizer | 1.5 must sit with `mlp_params`; a loop or call must still cover attention or all matrices |
| rstruct-19 | `p.ndim == 2` filter, which keeps the 2-D gates in Muon (assertion fails at run time) | negative regex for that filter |
| rstruct-20, 21, 28 | fill values such as 0.1 or 0.5 matched `fill_(0` | exact zero fill (`fill_(0)`, `fill_(0.0)`) |
| rstruct-24 | default-std init, aux weight 0.5, `aux[:, 1:]` | std 0.001, weight 0.3, a `[:, :-1]` shift |
| rstruct-27 | other taps never zeroed (the base already contained `zeros_(`) | zeroing must touch `v_conv` |
| rstruct-29 | `self.mlp = None` in every block | `Block` must compare `layer_idx` with `n_layer` |
| rrep-15 | `os.environ.get("T0", ...)` (renamed variable) | `!environ.get(` |

Valid solutions that the old regexes rejected, now accepted:

| id | valid solution | change |
|---|---|---|
| rstruct-01 | `3*config.n_embd` without spaces | spacing-tolerant regexes |
| rstruct-09 | `v = norm(v)` right after `cos, sin = cos_sin` (still before the rotary call) | expect accepts both positions |
| rstruct-10 | params list built with a loop (more than four lines before the clip) | any lines may come first as long as no update has run |
| rstruct-16, 17 | `scalar_lr*0.01` without spaces | spacing-tolerant regexes |
| rstruct-16, 17, 20, 21, 23, 27, 28, 30 | init written through `.data` | optional `.data` |
| rstruct-18 | a nested helper `add_groups(mlp_params, matrix_lr * 1.5)` | accepted |
| rstruct-25 | `_precompute_rotary_embeddings` reads `self.config.rope_base` itself | accepted |
| rstruct-27 | `w[:, :, -1].fill_(1.0)` for the identity tap | accepted |
| rstruct-30 | `x = x * self.final_gain` on the line after the final norm | accepted |
| rrep-06 | any safe new name for the variable, or deleting the dead line | the repairer never sees the base name `startup_time`, so it is no longer required (`must_not_touch` now guards the neighbouring lines) |
| rrep-07 | `torch.relu(x).pow(3)`, or `self.act = F.relu` (aliasing the function is allowed) | accepted |
| rrep-09 | `torch.relu(x).square()` | `MLP` removed from `must_not_touch`, regex accepts `torch.relu` |
| rrep-11 | `{config}` (the dataclass repr) instead of `asdict(config)` | accepted |
| rrep-15 | `t_harness = t_start`, the smallest safe edit | accepted |

Known limits that remain:

- The multi-block half leans on one edit-site pattern: six items (`rstruct-16`, 17, 21, 23, 24, 30) touch exactly `GPT.__init__`, `init_weights`, `forward` and `setup_optimizer`, like old `struct-7`. The ideas differ, but the items are not independent draws, so the item-level Wilson bound with n = 30 is somewhat optimistic.
- `rstruct-18` still accepts a 1.5x lr that is defined but never used (for example an unused `mlp_lr = matrix_lr * 1.5`), and `rstruct-21` cannot see a wrong skip index. The smoke test catches these only for the references.
- `rstruct-26` (Muon beta2) is plumbing more than architecture; it is in the `other` family like the other optimizer items, which `FAMILY_CLASS` routes to the structural coder.

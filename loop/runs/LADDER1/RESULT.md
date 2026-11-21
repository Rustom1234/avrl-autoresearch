# LADDER1: offline planner ladder

Question: how often does each model put the batch-size cut (lower TOTAL_BATCH_SIZE below 2^19) in its FIRST idea list? No GPU runs, 12 first lists per model, 60 calls in total. Raw replies: `lists.jsonl`. Script: `ladder.py` (`validate`, `collect`, `score`). Numbers: `scores.json`, `validation.json`.

## Result

| Model | Lists with the batch cut (of 12) | Lists that name TOTAL_BATCH_SIZE at all, any direction (of 12) | Cut items per list | Actionable items | Format failures | Cost per list |
|---|---|---|---|---|---|---|
| qwen35-4b | 4 | 5 | 0.58 | 50% | 0 | $0.0004 nominal |
| qwen35-9b | 5 | 10 | 0.42 | 56% | 0 | $0.0007 nominal |
| gemma4-12b | 4 | 9 | 0.42 | 77% | 0 | $0.0013 nominal |
| haiku | 12 | 12 | 1.33 | 51% | 0 | $0.0029 plan |
| sonnet | 12 | 12 | 2.00 | 71% | 0 | $0.028 plan |

"Actionable" = the item names an ALL_CAPS setting from the train.py hyperparameter block. Open-model cost is tokens times the nominal price in models.json; Claude cost is what the CLI reported.

**Pre-stated rule** (a non-sonnet model with the cut in at least 10 of 12 first lists): **haiku qualifies** (12 of 12). It is the only qualifier, so it is trivially the cheapest. models.json gives haiku no list price (it is plan-billed); measured plan cost is about $0.003 per list against $0.028 for sonnet, roughly a tenth. No open model qualifies. qwen35-9b reaches 10 of 12 only if any mention of TOTAL_BATCH_SIZE counts, and most of those items raise the batch size (2^20 and up), which is the wrong direction.

## What was reproduced

The first planner call of a P2 loop at slot 0 is `arms._plan("planner_backlog", 30)`: `prompts.planner(base train.py, 30, empty results table, no memo, no family stats, empty backlog)`, system prompt `program_card(20)`, schema `BACKLOG_SCHEMA`, temperature left at the server default, max_tokens 4096, timeout 600 s, 3 attempts (the `call()` default), and for open models `extra={"chat_template_kwargs": {"enable_thinking": False}}`. The base file is `loop/substrate/base/train_ampere.py`; the results table shows the anchor bar from `loop/runs/P2F/anchor.json` (1.1146 over 16 runs). Check: the prompt built here is 3818 characters, exactly the `prompt_chars` the P2F Qwen loop QS_01 logged for its first planner call.

## Scorer and validation

No reusable rule exists in `analysis_p2.py` or in P2F (the 36 of 36 and 10 of 24 are only quoted in text), so the scorer is new. An item has the cut if its title or spec lowers the optimizer batch: it names TOTAL_BATCH_SIZE with a target below 2^19, or a lowering word (reduce, lower, halve, ...) is aimed at TOTAL_BATCH_SIZE or "batch size". DEVICE_BATCH_SIZE-only changes, increases, splits into chunks and "reduce gradient noise" do not count. A list counts if any of its first 30 items does.

Validated on the stored first lists of all 60 P2F loops (first list = the items kept before the first voter calls of slot 0):

| Group | Scorer | Quoted in P2F |
|---|---|---|
| Sonnet-planned, 36 loops | **36 of 36** | 36 of 36 |
| Qwen-planned, 24 loops | **5 of 24** (loose "names TOTAL_BATCH_SIZE": 12 of 24) | 10 of 24 |

The Sonnet number reproduces. The Qwen number does not: I could not find any rule on the stored lists that gives 10 of 24 that is also sensible. Only 12 of the 24 stored Qwen lists even name TOTAL_BATCH_SIZE, and many of those raise it (2^20) or "split" it. My strict rule gives 5, the loose rule 12, and 10 lies between them, so P2F probably used a looser or hand-judged rule. The consequence is that my counts are conservative for weak planners. The conclusion for this question does not hinge on it: with the loose rule the ranking is the same (haiku and sonnet 12, 9b 10, gemma 9, 4b 5) and haiku is still the only model whose items mostly lower the batch size.

Actionable share on the stored data: Sonnet 71% (P2F: 68%), Qwen 39% (P2F: 28%). The P2F rule is not in the repo; mine counts any hyperparameter-block name, so it is a little more generous.

## Assumptions and caveats

- Offline first lists only, no GPU runs. A list that has the cut is not a loop that finds it: P2F showed the voter priors, the coder and abort rates also matter (a strong coder rescued 11 of 14 Qwen cuts).
- The stored P2F lists are post-dedup; the new lists are raw replies cut to 30 items (the engine keeps `kept[:30]`, and removes near-duplicates, which is not applied here).
- 12 lists per model: 12 of 12 against 4 or 5 of 12 is a clear gap, but 9 against 10 of 12 is not.
- One prompt (the slot-0 prompt), default temperature; each rep is an independent sample. All 60 calls were served by the requested model (no fallbacks), no format failures.
- qwen35-9b and gemma4-12b were served on cherry and pine as found; I did not start or touch any server.
- Haiku's actionable share (51%) is low despite a perfect cut rate: its lists reach the cut but many other items name no real setting.

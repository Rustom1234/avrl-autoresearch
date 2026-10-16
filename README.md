# active-audit: an autoresearch loop and what it found

Rustom Dubash, Haverford College, Sept to Oct 2026. One person, a lab pool of 16 GB cards, and a 4x A100 node, running a research project through an autonomous experiment loop with a pre-registered ledger.

The project asked one question: during RL post-training, when a cheap grader scores every rollout and an expensive grader (a large judge, a full test suite) is only affordable for a fraction of them, which rollout groups should get the expensive call? The loop ran about 60 training runs and 50-odd analyses across three task families and killed two of the three pre-registered claims. The surviving result replicated at 7B scale.

## The loop

`avrl/autoresearch.workflow.js` is a 46-line orchestrator. Each iteration runs four agents in sequence, each with a JSON output schema:

| role | job |
|---|---|
| planner | reads the backlog, the ledger and the newest result files; picks the experiment with the highest information value per GPU hour that fits a 45-minute cap; prefers experiments that can kill a claim |
| runner | implements it by copying and modifying existing scripts, launches it detached with a done flag, polls, kills it at the cap |
| analyst | reads the raw result files (not the runner's report), appends a ledger entry: hypothesis, run, a table of numbers, verdict against the pre-registered pass and kill thresholds, what it changes, follow-ups |
| critic | re-derives every number from the raw files, checks leakage, budget parity and seed adequacy, appends a correction note if anything is wrong |

Guardrails: one GPU job at a time, all writes under one directory, no git pushes, kill patterns that cannot hit the agent's own shell.

What happened when it ran, honestly:

- Nine iterations plus direct passes produced about 40 pre-registered ledger entries in the first four days (`avrl/RESEARCH_LOG.md`, now 115 entries).
- The critic corrected numbers or verdicts in several entries without overturning a conclusion, and once caught a verdict label that contradicted the entry's own table (M01).
- An evaluation bug (the judge's 6-token output cap passed as the policy's generation cap, so every held-out "quality" score on the free-text task measured truncated stubs) was found not by the critic but by a common re-evaluation pass. Every affected number was voided and re-measured.
- Failure mode: the planner drifted toward cheap variants of the same replay analysis. A priority note in the backlog redirected it; the one experiment it kept skipping (the code task) was run under an explicit spec.
- The loop is single-model (Sonnet in every role) with no routing. `avrl/autoresearch-routing-context.md` compares this with AlphaEvolve, ShinkaEvolve, OpenEvolve, AI Scientist v2 and AI co-scientist and argues from the evidence for two cheap upgrades: the strongest model on the critic role only, and OpenEvolve-style "inspirations" (sampled killed experiments) in the planner's context.

## Research on the loop itself (AR1, Oct 16)

`avrl/exp/ar1_loop.py` replays the loop's own trace as an offline evaluation of planner context. The 52 critic-checked experiments of the small-scale campaign are the oracle: a planner picks from the menu of not-yet-run experiments, the runner is a lookup that returns the archived numbers with the verdict stripped, an analyst writes verdicts and the state of the three claims. Three arms differ only in what the planner sees about past work (full logs, compressed ledger, compressed ledger plus inspirations). The metric is experiments-to-correct-verdict, with model cost as the secondary. Pre-registration and result are in the ledger (entries AR1); raw traces in `avrl/runs/AR1/`.

Result (3 arms x 5 seeds, Sonnet): full logs 5.2 experiments to verdict on average, compressed ledger 7.2, compressed plus inspirations 10.2. Both pre-registered reads failed. The full-log planner was also the cheapest per run because it stopped earliest; the longer runs in the other arms spent picks on experiments that could not move a claim. The write-up's "cheapest upgrade" is withdrawn; the recommendation becomes: give the planner the whole ledger while it fits.

AR2 (`avrl/exp/ar2_critic_eval.py`) evaluates the critic role the same way: one defect planted per ledger entry (a narrative number that disagrees with the table, or a flipped verdict word), untouched entries as controls, Haiku against Sonnet reading the entry alone. Pre-registration and result in the ledger (entries AR2).

Result: both models detect the planted defects at similar rates (number defects located 0.72 Haiku, 0.81 Sonnet; verdict flips 0.83 and 0.74). The difference is specificity: Haiku flags half of the untouched entries as inconsistent, Sonnet one in five, at twelve times the cost. The case for a stronger model on the critic role is precision, not recall. A by-product: both models found real contradictions left in the prose of untouched entries, which the file-based critic had not removed.

## What the research found

Full statement: `avrl/PROJECT.md`, section "Where the claims stand". Figures: `lit/avrl/fig/`.

- A threshold on the cheap grader's own confidence (vote agreement on exact answers, agreement among visible-test passers on code, response consistency on free text) is never beaten at equal budget by a learned allocator. Claim 2 (a learned allocator admits fewer wrong rewards than any fixed rule) killed where the signal is strong, tie elsewhere. Claim 3 (prompt-space structure helps) killed everywhere.
- Claim 1 (equal quality at a quarter of the expensive calls) passes on GSM8K, code and free text at small scale, and at 7B scale on two seeds (Qwen2.5-7B policy, 0.6B reward model as cheap grader, 14B judge as verifier, 4000 steps per arm). The fair bound with a rank-shaped judge reward is not yet run.
- The cheap grader's error rate measured before training predicts the outcome; no cheap grader drifted or was gamed over 100 to 4000 steps. A burn-in diagnostic (`avrl/avrl_router/diagnose.py`) turns that measurement into a budget.
- Reward-model overoptimization appeared only under high optimization pressure at 7B, and the decline is judge-dependent: two judge families agree, a third reverses it. Reported as mixed, not claimed.

## Layout

| path | what |
|---|---|
| `avrl/autoresearch.workflow.js` | the loop |
| `avrl/RESEARCH_LOG.md` | the ledger, append-only, 115 entries with pre-registered verdicts and critic notes |
| `avrl/PROJECT.md` | claims, kill tests, where the claims stand |
| `avrl/STATUS.md` | running narrative, Oct 1 to Oct 5 |
| `avrl/exp/` | one script per experiment, written by the runner agents |
| `avrl/runs/` | result files per experiment (weights and logs are not tracked) |
| `avrl/p0_*.py`, `avrl/p1_*.py`, `avrl/allocators.py`, `avrl/judge_rewards.py` | the harnesses: rollout sampling, structure statistics, GRPO with a pluggable verification allocator |
| `avrl/avrl_router/` | the regime diagnostic CLI |
| `avrl/autoresearch-routing-context.md`, `avrl/llm-credit-measurement.md` | side write-ups |
| `lit/avrl/` | literature sweeps (14 angles, 34 agents), synthesis, proposal, backlog, gap checks, figures |
| `SCOPE.md`, `LOOP.md`, `AGENTS.md`, `aa/`, `runs/` | the predecessor project (Sept 9 to 18): a budgeted active audit for sensitive pretraining data, same SOFisher framing, stopped at a negative result |

Models (`models/`) and datasets (`data/`) are not tracked; the scripts name what they load.

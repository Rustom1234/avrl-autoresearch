# The 100-agent loop: scope (not yet run)

Date: Sept 9, 2026. Status: designed, awaiting Rustom's go. Two informing agents (`s0-layouts`, `sim-policies`) are running first; their results set the loop's parameters.

## What the evidence says about running 100+ agents

Sources: Anthropic's multi-agent research system writeup, Google Research "Towards a science of scaling agent systems", the token-budget incident catalog, context-rot and worktree-collision papers, plus our own July 2026 incidents (memory: subagent idle-wait, shared-scratchpad collisions, the unsupervised 20-worker burn).

1. Orchestrator-worker, never flat. A central coordinator caps error amplification at ~4.4x versus ~17x for uncoordinated peers. Here the coordinator is the workflow script plus me reading every phase result.
2. Fan out only decomposable work. Sequential-dependency tasks got 39 to 70% worse with more agents. Experiment variants are decomposable; "design the method" is not, so design is a small judge panel, not a fleet.
3. Budget 15x chat-level tokens for a full multi-agent run. Estimate below.
4. Every task spec has an objective, an output schema, tool guidance, and hard boundaries. AGENTS.md plus per-agent JSON schemas.
5. Isolated workspaces and read-only shared data. `runs/<label>/` per agent, `data/` and `aa/` read-only, no generic filenames.
6. Cheap-first funnel: many cheap agents to generate and screen, few expensive ones to verify. Screening wants fewer, stronger agents; verification justifies multi-pass.
7. Verify by execution and adversarial re-derivation, not by trust. Every number that matters gets re-run by a different agent from the artifact, and a refuter tries to break it.
8. Harness-tracked waits only. No agent polls or sleeps for another; the script sequences them.
9. No silent caps: log what was dropped at every filter.
10. Stay far below the ~500-agent coordination cliff. 100 to 130 is the design point; the harness runs ~16 concurrently on goldberg's 20 cores, so the loop is 7 to 9 waves.

## Phases

Phase 0, parameters (already running, 2 agents): autocorrelation per layout; whether static triage already beats adaptive on regex labels. Outputs fix the layouts and the policy classes worth testing.

Phase 1, theory generation (judge panel, ~12 agents): 8 proposers each write one candidate method for the auditor under a distinct constraint (pure bandit, active search nonmyopic, SOFisher DQN port, embedding-neighborhood expansion, hierarchical domain-tree search, online model-signal coupling, masking versus dropping, canary design). 4 judges score all 8 on feasibility on this data, novelty versus the prior-art list, and expected effect size. Barrier here is correct: the next phase needs the ranked set.

Phase 2, experiment fleet (pipeline, ~60 agents): each item is one (method, layout, budget, seed-set) cell from the top-ranked methods. Each agent implements or adapts the simulator in its own run dir, runs it, and returns a schema-checked result with the script path. Cheap model for mechanical cells, stronger model for the cells that need a new component (DQN port, embedding neighborhoods).

Phase 3, verification (pipeline, ~30 agents): for every cell whose result beats the static-triage baseline, one replicator re-runs the script from the artifact with a fresh seed, and one refuter reads the code looking for leakage of the label into features, budget accounting errors, or seed cherry-picking. A cell survives only if the replication is within the stated CI and the refuter finds nothing fatal.

Phase 4, synthesis and completeness (3 agents): one writes the ranked findings table with survivors only, one lists what was not tested, one drafts the next-loop plan. I read all three and decide with Rustom.

Total: about 105 agents.

## What each agent gets and returns

- Gets: AGENTS.md, `aa/data.py`, the cell spec (method, layout, budget, seeds), and the read-only data paths. Nothing else.
- Returns: `result.json` matching a fixed schema: numbers with n and seed, script path, wall-clock, assumptions, failures. Text outside the schema is ignored.

## Cost and time

- Phase 0 and 1: ~1M tokens. Phase 2: ~60 agents x 60 to 120k tokens = 4 to 7M. Phase 3: ~3M. Phase 4: small. Total roughly 8 to 12M tokens, in the range of the July 2026 session-limit burn, so this needs Rustom's explicit go and a `budget` ceiling passed to the workflow.
- Wall clock: 6 to 10 hours with 16 concurrent agents, dominated by the experiment fleet's CPU time on goldberg (each cell is a 5 to 15 minute CPU job; 16 at a time on 20 cores is the real cap, so the fleet must cap n per cell).

## What the loop cannot do

- It cannot run the expensive LLM-judge detector for every cell; only a few designated cells get GPU time on the 5090. Regex labels stand in for the detector everywhere else, which biases toward "static triage wins" because regex is exactly the cheap feature. This is stated in the synthesis.
- It cannot do pretraining runs; those are hannah jobs after the loop, on the surviving method only.

## Gate to launch

Run only if phase 0 shows a layout ratio at or above 3x and adaptive policies are not already dominated by static triage on the regex proxy. If both fail, the 100-agent loop would be an expensive way to confirm a negative; do the cheaper thing instead.

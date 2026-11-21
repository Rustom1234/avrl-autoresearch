# P2B build notes (built, NOT launched)

Built (new files only): loop/core/plan_p2b.py, loop/core/tests/test_plan_p2b.py, loop/runs/P2B/launch_p2b.sh, this file.
Builder reuses plan_p2 (`import plan_p2 as P`, same module object the tests use): during one make_plan call it swaps
P.FACTORIAL and P.routing_table, then restores them. H is in every block via harm_blocks = n_blocks. Seed 2 (P2's) is refused.
Cells per block: SS, ST, QT, H, F, R (+ M with --mid-planner). R tables are written per loop as routing/R_NN.json;
the 9 coins (5 planner roles, 3 coder classes, repairer) are in the loop config ("r_draws") and plan.json.

F split honored by the engine: loop/core/arms.py `_route` (lines 242-255) looks up the role's own key first, and
`_pick_item` (lines 430-435) calls `_plan("planner_backlog")` for the initial backlog, "planner_widen" for widening and
"planner_refresh" for refreshes. So planner_backlog=sonnet with the other planner keys on qwen35-4b takes effect, no engine
change. (The stale comment at arms.py:251 saying backlog/widen stay on sonnet is out of date; the table wins.)
Verified in test_plan_p2b via the real LoopEngine._route. Keys "planner" and "planner_cheap" are inert fallbacks
(planner_refresh is set in every table), so their R coins change nothing; they still count as draws.

Launch (from spruce, after P2B_PREREG.md is final and loop/design/P2B_PREREG.sha256 exists, which the launcher requires):
  .venv/bin/python -m loop.core.plan_p2b --root loop/runs/P2B --seed N --n-blocks 12 [--mid-planner qwen35-9b]
  cd /homes/rdubash/active-audit && nohup loop/runs/P2B/launch_p2b.sh > loop/runs/P2B/launch.out 2>&1 &

Cost estimate (P2F loops.csv: 18.7 jobs/loop, 0.112 h/job overall; per cell SS 19.9 jobs 0.114 h, ST 18.3 0.113,
QT 15.8 0.098, H 19.9 0.115; F and M taken as QT, R as mean of SS/ST/QT):
  72 loops (SS ST QT H F R x12): ~1290 loop jobs, ~141 A100-h. With 144 confirmation jobs and ~48 sentinels: ~1480 jobs, ~162 h.
  84 loops (+M x12): ~1480 loop jobs, ~160 A100-h. With 168 confirmation and ~53 sentinels: ~1700 jobs, ~184 h.
  Wall clock on 4 A100s: roughly 40 h (72) or 46 h (84), inside the 66 h window.

Hannah (read-only check): the queue is one shared root /home/rustom/arloop, set by QueueClient's default root, not by
any loop root; start_workers.sbatch and worker.py take only ROOT=/home/rustom/arloop. Nothing ties them to P2.
Nothing needs to change for a new loop root. Observed: queue empty, no worker job of ours running, a STOP file
(Nov 8) in /home/rustom/arloop that the sbatch removes at start (FIRST_JOB=1). n7 contention (see memory notes) still applies.

Open risks:
- P2B_PREREG.sha256 does not exist yet; launcher aborts until it does.
- R with all-SS or all-QT coins duplicates SS or QT (P(all 9 equal) ~ 0.4 percent per loop).
- cherry is shared with qwen7b-awq and qwen35-9b has max_context_tokens 8192 in models.json (an assumption) vs
  --max-model-len 16384 on the server; M planner prompts could be truncated or rejected. Check before using M.
- launcher health step for cherry only runs when plan.json has mid_planner == qwen35-9b; cherry server untested here.
- F makes sonnet planner calls only for backlog, so served-fraction analyses keyed on planner roles need care.
- Controller, anchor and confirmation have no cell-specific logic (grepped), so new cell letters are safe there.

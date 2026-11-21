# S1: the loop engine on ShinkaEvolve's circle-packing task

Written Nov 16 2026, before any loop of this run started (a 2-loop, 4-slot plumbing pilot ran first, see Pilot).
Frozen by `S1_PREREG.sha256`.

## Why

ShinkaEvolve (arXiv 2509.19349) reports a sum of radii of 2.635983 for 26 circles in the unit square in about 150
generations, starting from a hand-placed layout that scores 0.96. Gideoni, Risi and Gal (arXiv 2602.16805) report
that under a 20 USD budget a simple sampling baseline reaches 2.63590 and plain independent sampling 2.632. One of
our Sonnet loops reached the same top packing in 10 experiments (BG1_R, p26_SS_3), but from our own starting script,
an optimiser that already scores 2.628, so that was not a fair comparison. S1 is the fair one.

## Task (loop/shinka/)

- Starting program: `initial_shinka.py`, a verbatim copy of ShinkaEvolve's `examples/circle_packing/initial.py`
  (sum of radii 0.9598).
- Evaluation: one call of `run_packing()` per job, their validity check with zero tolerance (their current
  `evaluate.py`; the paper's 2.635983 allowed an overlap of 1e-6, and its value under an exact check is 2.635978),
  5 minutes of wall time per evaluation as in their `run_evo.py`, one CPU thread. An exact rational check is
  recorded next to it.
- Domain hints: the eight "key directions" of their task message are in the program card word for word.
- Extra restriction on our side: numpy and scipy only, and no pasted coordinates.
- Budget: 150 charged slots per loop (every job counts: candidates, replicates, crash fixes). Engine unchanged;
  sigma is set to 1e-6 because the evaluator is exact, so the accept rule is in effect "keep if better".

## Cells

Planner then coder: SS (Sonnet, Sonnet), HH (Haiku, Haiku), SH (Sonnet plans, Haiku codes). Analyst Sonnet, critic
Opus, voter Haiku everywhere. 3 loops per cell, 9 loops, two per lab machine (`make_plan.py --seed 41`).

## Outcome

Per loop: the best valid sum of radii over all its jobs; the number of evaluations and the model dollars (list
price) spent when it first reached 2.6359 (the simple baseline's level) and 2.635977 (the top packing).

| Claim | Holds when |
|---|---|
| M match | in at least one cell, at least 2 of 3 loops reach 2.635977 within 150 evaluations |
| C cheaper | M holds, and the median model cost at the moment of reaching it, in that cell, is below 6 USD (half of the 12 USD cited for a 150-generation ShinkaEvolve run) |
| R routing | the best sum of SH is at least that of SS in the median loop, at lower median total cost |

Described without a rule: evaluations to reach each level, per loop; what the winning scripts do; how many
evaluations came after the last improvement.

Predictions (probability the claim holds): M 0.4, C 0.35, R 0.45. Probability that at least one loop of nine reaches
2.635977: 0.7.

## Limits known in advance

- Different language models from theirs, and their numbers are taken from the papers, not rerun here.
- 3 loops per cell.
- Our engine keeps one incumbent and edits it; ShinkaEvolve keeps an archive. Their ablation reports that hill
  climbing plateaus early, so a miss here would be evidence for their design, and is reported as such.
- Model dollars only; evaluation compute (up to 5 CPU minutes each) is reported separately.
- The models may know this problem and its literature. That holds for ShinkaEvolve's models too.

## Shared machines

CPU only, two loops per machine on willow, walnut, pine, cherry, spruce, one single-thread job per loop. Measured
peak memory of the harness on the starting program: 0.03 GB; 16 GB address-space limit and a quiet watchdog at 6 GB
per job; no automatic retry of a job stopped for memory. A watcher stops the run on any authentication failure.

## Pilot

Two loops of 4 slots on pine (`loop/runs/S1_PILOT`, one HH, one SH) tested the plumbing. Not part of S1; nothing was
changed after it. For the record, since I saw it before freezing: no model call failed; the HH loop went 0.96 to
2.254, 2.415, 2.578, 2.613 in four evaluations and the SH loop to 2.578, 2.607, 2.631 in three, then one reject.
The predictions above were written before the pilot finished and were not changed.

## Deviations

None yet.

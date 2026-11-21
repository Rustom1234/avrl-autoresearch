# B2: does a stronger planner pay where a cheap one fails, or where it succeeds?

Written 2026-11-18, about 21:55, before any B2 test result exists. Drafted by the coding agent under Rustom's
"do what you think is best" instruction for this evening; he has not reviewed it yet. It is his to audit or void.

## Why

B1 (six stalled Haiku loops, slot 40) gave: Haiku, Sonnet, Opus and Fable as planner found 1, 2, 3 and 3 promising
ideas in about 100 ideas each on the five fast tasks. Every promising idea was in one loop (clearance). In the other
four loops no model found any (0 of 318). The coder made no difference (same six ideas under a Haiku and a Sonnet
coder). So the stall looked like a property of the loop state, not of the model.

The usual routing rule is a cascade: use the cheap model, move to the strong one when the cheap one fails. B1 suggests
the opposite for research loops: the strong model finds something only where the cheap one also does. One loop with
room is not evidence. B2 tests it on more states.

## Design

- States: the slot-40 state of every non-lipo loop of BG1_R: 24 states (bbb, caco, clearance, pack26, pack32; Haiku
  loops and Sonnet loops). Five of them are already done in B1 and B1F and are reused as they are. 19 are new.
  Lipo is left out: its runs sit at 7.5 to 9.5 of the 10 allowed minutes, so its results depend on machine load.
- Planners: haiku, sonnet, opus, fable. Context: continued only (the loop's table, notes, backlog titles). B1 showed
  that a fresh context mostly re-proposes tried ideas. 10 ideas per planner per state, one call, one try.
- Test: every idea against the frozen slot-40 incumbent with the unchanged engine pipeline, first run on seed 42,
  rerun rule unchanged. Coder and repairer: haiku in every state, including the Sonnet loops (B1: the coder did not
  change any outcome on 298 ideas; this keeps a test at about 0.005 USD). This is a deviation from "the loop's own
  coder" for the 11 Sonnet states and is a limit of the result.
- An idea proposed by two planners (MiniLM cosine 0.90 or more) is tested once and counts for both.

## Outcomes

- Primary: promising = first-run z1 of 0.75 or more. Secondary: accepted; mean z1 over valid runs; best z1.
- Unit of analysis: the state (24). Intervals by bootstrap over states.

## Questions and rules, fixed now

Q1. Escalate on failure. Among states where the Haiku planner's 10 ideas contain no promising idea, in what share
    does the strong planner (opus, fable, taken separately and together) find at least one?
Q2. Escalate on success. The same share among states where Haiku found at least one.
Q3. Reading: "room is a property of the state" is supported if the Q2 share is at least 2 times the Q1 share and the
    bootstrap interval of the difference (Q2 minus Q1) excludes zero. It is rejected if the Q1 share is 0.25 or more
    (the cascade does rescue stalled states often). Anything else: undecided, say so.
Q4. Policy replay (every arm was tested in every state, so each policy's result is read off, not estimated):
      A  always haiku          B  always sonnet        C  always opus          D  always fable
      E  cascade: haiku first, then fable only if haiku found nothing promising
      F  reverse: haiku first, then fable only if haiku found something promising
      G  haiku first, then sonnet under rule F
    For each: promising ideas found, accepted ideas found, dollars (ask plus test, nominal), dollars per promising
    idea. Same table at test prices 0.05 and 1 USD (test count times price, ask prices as measured).
Q5. Does any strong planner beat haiku on mean z1 inside states (paired over states, sign-flip test)? Reported with
    the size, whatever the answer.

## Known limits, stated before the data

- 24 states, 10 ideas per arm: only large effects can show. "Promising" is noisy in some pack states (in E1 two Sonnet
  pack loops had 13 promising first runs and no accept).
- "Haiku found something" is measured on the same 10 Haiku ideas that policy A is paid for, so E and F use
  information a live loop would have at that point. No information from the strong arms leaks into the rule.
- One slot (40) per loop; one seed for first runs; Haiku coder on Sonnet-loop incumbents.
- All parts of any rule that comes out are known decision theory (cascades, value of information). What is new, if
  anything, is the measured direction of the effect in research loops.

## Cost

About 76 planner calls (about 14 USD nominal, mostly fable and opus) and 760 tests (about 4 USD nominal).
No retries of any kind. Jobs stay under the harness limits (8 threads, 16 GB address space, 6 GB memory stop).

## Deviations

1. 2026-11-18, 22:50. State bbb_SS_2 is left out and its two shards were stopped after 5 of 39 ideas. Reason: the
   same rule that leaves lipo out. Its incumbent runs close to the 600 s limit (median 484 s, longest 597 s over 41
   valid runs in the E1 continuation), so its results depend on machine load; 6 of its first 10 runs here timed out.
   Decided from run times and statuses only, before reading any score of that state. B2 has 23 states, 18 of them new.
2. 2026-11-18, 22:50. Machine load. The first six drug-property shards (bbb states) and two clearance shards ran on
   spruce together and saturated it (load 33 on 32 threads): runs of Sonnet-loop incumbents take 2 to 6 minutes, not
   the 45 s measured on Haiku-loop incumbents. Recorded up to then: 1 timeout in bbb_HH_2 and 1 in bbb_SS_1, both
   kept as they are (not rerun). All shards were stopped between slots and resumed with at most 3 drug-property
   shards per machine (5 on walnut, 2 of them nearly finished). Timeouts are reported per state in the result.
3. 2026-11-19, 01:49. The resume named in deviation 2 did not start on Nov 18: the coding agent's wait loop matched
   its own command line and waited from 22:56 to 01:45. No job ran in that time and nothing was lost. The placement
   of deviation 2 (3 per machine, 5 on walnut) was never used. Resumed at 01:49 under Rustom's rule of Nov 19 (never
   overload the machines): 14 shards in three waves, at most 2 drug-property shards per machine (16 of 32 threads in
   the worst case), one at a time on spruce, with a load guard (`loop/ideas/load_guard.py`) that writes the STOP
   file if any machine stays above load 24. No score was read between the stop and the resume.

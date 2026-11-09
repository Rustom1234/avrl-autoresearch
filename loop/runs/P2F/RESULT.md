# P2 result: which loop role earns the expensive model

Run 2026-11-06 19:02 UTC to 2026-11-08 10:56 UTC on hannah n7 (Slurm job 82627, four A100s). It used 60 loops of 20 slots on Karpathy's autoresearch nanochat benchmark, under the frozen preregistration loop/design/P2_PREREG.md, with one deviation (below). The analysis code is loop/core/analysis_p2.py, unchanged since freezing, and its output is in analysis/.

Three independent checks verified every number in this file:
- **Recompute:** from the raw logs without the analysis code, with zero discrepancies.
- **Confound search:** reruns, GPU, time, budget, served models and confirmation waves.
- **Mechanism analysis:** why the cells differ.

Their scripts (p2verify/) are not included in this folder.

## Deviation

**A model-provider outage forced a partial rerun.** From 14:19 to about 16:10 UTC on Nov 7, every Claude call failed with HTTP 403, an account-level refusal by the model provider.
- 28 loops were damaged.
- They were rerun from scratch with identical configs, and the call layer now pauses on such errors.
- The final set (this folder) is the 32 untouched loops plus the 28 reruns.
- In these 60 loops there is no failed Claude call, no 403 and no fallback, and every planner and coder call went to its designated model.
- The rerun choice was mechanical: exactly launch indices 32 to 59.
- Details are in P2_PREREG, Deviations.

## Preregistered verdict: KILL

The harm control was not detected on the preregistered endpoint. H (Sonnet everywhere, but half the coder's new edits withheld) did not reach a total batch of 2^17 later than SS: one-sided log-rank p = 0.22. By the preregistered precedence, the factorial is reported as a cost breakdown, with per-role effects as description.

The PIVOT condition (cheap planner raises the stall rate by more than 0.25) was also met: +0.29, Fisher p 0.060. It clears the margin by a single loop.

## Results

| Cell | Planner | Coder | Stall (final batch above 2^17) | Held-out gain (bpb) | Accepted changes per loop | Model $ per loop | GPU hours per loop |
|---|---|---|---|---|---|---|---|
| SS | Sonnet | Sonnet | 7 of 12 | 0.0545 | 2.17 | 1.43 | 2.27 |
| ST | Sonnet | cheap table | 6 of 12 | 0.0560 | 2.08 | 0.25 | 2.08 |
| QS | Qwen3.5-4B | Sonnet | 9 of 12 | 0.0257 | 1.25 | 1.66 | 2.22 |
| QT | Qwen3.5-4B | cheap table | 11 of 12 | 0.0124 | 0.67 | 0.15 | 1.56 |
| H | Sonnet, half the edits withheld | Sonnet | 9 of 12 | 0.0328 | 1.17 | 1.45 | 2.29 |

Held-out gain is the mean of the confirmation anchors minus the loop's mean on the sealed shard, from two reruns per loop. Model dollars are Claude at list price plus open models at hosted prices.

| Main effect (24 vs 24 loops) | Held-out gain | Stall | Settling (slots to last change) | Model dollars |
|---|---|---|---|---|
| Cheap planner (Q minus S) | -0.036 bpb, 90% interval -0.044 to -0.028, p below 0.001 | +0.29, p 0.060 | 14.2 vs 9.5 slots, log-rank p 0.001 | ratio 0.83, p 0.55 |
| Cheap coder table (T minus S) | -0.006 bpb, -0.017 to +0.006, p 0.40 | +0.04, p 1.0 | 11.5 vs 12.2, p 0.67 | ratio 0.13, p below 0.001 |

- **The planner effect is robust.** It shows in 11 of 12 blocks (sign-flip p 0.0015). It is the same in the kept loops (-0.0365) and the reruns (-0.0355). It is unchanged by controls for block, GPU, launch order and GPU jobs used. It also holds between QS and SS, which used their budgets almost equally (-0.029, p 0.002). Held-out gain was a preregistered secondary outcome.
- **No coder main effect was detected, but the cheap coder may not be harmless with a weak planner.** With the Sonnet planner, the cheap coder table is neutral (ST minus SS +0.0015). With the Qwen planner it may hurt (QT minus QS -0.013, p 0.12). The supported claim is "no detectable coder main effect", not "the cheap coder is harmless".
- **Cost.** The cheap coder table cuts model dollars about 8 times. With GPU time at $2 per A100-hour it cuts total loop cost by about 38 percent. QT is 2.5 times cheaper than SS per backed accepted change, short of the 5 times PASS required. That ratio rests on only 4 backed changes in QT.

## Why: the mechanism

- **One setting decides most of the outcome.** Final TOTAL_BATCH_SIZE alone explains 78 percent of the variance in held-out gain across the 60 loops. Every batch cut from 2^19 that reached a GPU was accepted (42 of 42).
- **Sonnet proposes that cut; Qwen mostly doesn't.** Sonnet planners put the batch cut into every initial idea list (36 of 36). Qwen put it into 10 of 24, and 14 of the 24 Qwen-planned loops never ran a batch cut. All 14 stalled, against 1 of 24 Sonnet-planned loops.
- **Qwen's ideas were less actionable.** 28 percent named a real setting of the script, against 68 percent for Sonnet. Coders aborted 27 to 35 percent of attempts in Qwen-planned cells, against 8 percent in SS.
- **A strong coder partly rescues a weak planner.** Of the 14 batch cuts that ran in Qwen loops, 11 came from the coder swapping the known lever in for a vague idea. QS ran a cut in 7 of 12 loops, QT in 3 of 12.
- **The planner tryout could not see any of this.** All five models scored 12 of 12 on it, but it checked only the reply format.

## Why the harm control was not detected

The harm did happen, but the preregistered endpoint could barely register it:
- **The endpoint had too few events.** SS itself reached 2^17 in only 5 of 12 loops, against 18 of 22 in R1, whose times set the power estimate. That left 8 events in total.
- **The no-ops cut both ways.** They withheld helpful batch cuts (13) and also depth cuts that often lead into the stall trap (12), which partly cancel on stall and slots.
- **The damage is visible elsewhere.** On outcomes not preregistered for the harm comparison, H lost a full accepted change per loop (p 0.003) and 0.022 bpb of held-out gain (p 0.005).

The KILL therefore reflects a weak endpoint, not a loop that cannot tell good work from waste.

## Corrections to the generated report (analysis/report.md)

The frozen code's report has three text errors. Its numbers are all correct.
1. The KILL caveat quotes power "roughly 0.2 to 0.7 for 6 H loops". This run had 12 H loops, and the preregistered estimate for 12 is 0.47 to 0.89. Given how few SS loops reached 2^17, the real power was likely below that.
2. "GPU time is the same for every cell" is wrong. QT used 1.56 GPU hours per loop against 2.08 to 2.29, because 50 of its 80 no-GPU charged slots were patch failures and aborts.
3. The preregistered comparison of kept and rerun loops is missing. Here it is: the planner effect is -0.0365 on kept loops and -0.0355 on reruns. The coder effect is consistent in both halves. The harm stall difference is 0.00 among kept loops and +0.33 among reruns, on only 6 loops each.

## What can be claimed

- **Measured:** in this loop, the planner is the role that earns the expensive model. A cheap planner that passed its tryout 12 of 12 lost about two thirds of the held-out gain (0.036 of 0.055 bpb). A calibrated cheap coder table cut model spend about 8 times, with no detectable loss when Sonnet planned.
- **Measured:** a per-role tryout that checks only reply format cannot qualify a planner. Whether a planner proposes the few levers that matter is a property of the content, and only the loop revealed it.
- **Not claimed:** that the cheap coder is harmless in general (possible interaction with a weak planner), or that the harm control proves the substrate sensitive. The preregistered test of that failed. The supporting evidence is post hoc.

## Limits

- **One benchmark,** where one lever decides most of the outcome. A substrate with many small levers could weight the roles differently.
- **Stall separated outcomes less cleanly than in R1.** Stalled loops range from 0.008 (still at 2^19) to 0.058 (2^18 at depth 6).
- **The reruns ran about a day after the kept loops.** In-run sentinels drifted slightly during the reruns, but sealed gain comes from one common confirmation block with flat anchors, and the effects agree across the halves.
- **The voter (gemma4-12b) is held constant but is part of the mechanism.** It gave Qwen's ideas lower priors (0.7 of 3 yes votes against 2.0).

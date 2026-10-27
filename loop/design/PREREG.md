# Pre-registration: routed autoresearch loop on nanochat (hannah n7)

Frozen before any loop run starts. Deviations are logged in the Deviations section with timestamps; nothing above that section changes after launch.

## Fixed inputs

- Substrate: karpathy/autoresearch 228791f; pinned prepare.py sha256 4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150; Ampere baseline train.py = base with DEVICE_BATCH_SIZE 64 (loop/substrate/base/train_ampere.py).
- Noise job (hannah 79132, loop/runs/noise/): n = 16, pooled sd 0.006297, residual sd after per-GPU means 0.005857, per-GPU offsets GPU0 -0.00306, GPU1 +0.00474, GPU2 +0.00166, GPU3 -0.00334 (n = 4 each; re-estimated at launch with the 8 live sentinels, n = 6 each, then frozen in anchor.json).
- sigma_used = 1.15 x 0.005857 = 0.006736.
- Anchor value: mean of the 24 baseline runs (16 noise + 8 launch sentinels), frozen in anchor.json.
- N_SLOTS = 20. Loop runs: A 8, D 8, B 3, C 3. Confirmations: 4 per incumbent, seeds 1001 to 1004, randomized four-wide waves mixing arms, one anchor run per wave.
- Window: 24 hours from controller start. Loop runs stop submitting at hour 19.5; confirmation block hours 19.5 to 23.5. Spend ceiling: 60 million Claude tokens (all types) from calls.jsonl, which is roughly 40 percent of a week of plan usage at recent rates; symmetric stop at the current slot index of every in-flight loop.

## Decision rules (frozen)

Arm A: keep when raw official val_bpb of the single run is strictly lower than the incumbent bar (initial bar = anchor value).
Arms B, C, D: values adjusted by GPU offset; z1 = (incumbent_mean minus candidate) / (sigma_used x sqrt(1/1 + 1/n_i)); reject if z1 < 0.75; accept if z1 >= 2.0; else replicate with a fresh seed and accept if z2 >= 1.5 with n_c = 2; incumbent n_i = runs behind the incumbent (24 for the anchor, then its own runs).
All arms: integrity flags (section 4 of DESIGN.md) reject and charge the slot; tripwires (gain over 5 x sigma_used x sqrt(2) = 0.0476 in one step, or over 0.03 cumulative) force an audit run; invalid_seed rejects.

## Hypotheses and predictions

H1 (confirmatory). Sealed improvement over the same-wave anchor, D versus A. Test: two-sided exact permutation on the difference of means over loop runs (8 vs 8, 12870 permutations). Decision: "D better" if p < 0.05 and the estimate exceeds 0.003; "A better" symmetric; otherwise "inconclusive", reported with the 90 percent permutation interval. Prediction stated in advance: inconclusive is the most likely outcome unless D finds the batch-size class of change more often than A; power at a true 0.008 difference is roughly 0.5.

H2 (rule check, descriptive, expected sign stated). Selection gap gap_sel (claimed official minus confirmed official, anchor adjusted) is larger for A than for B, C and D. Prediction: A about 1.5 to 2 sigma_used (0.010 to 0.013); verified arms under 0.006.

H3 (descriptive). Bad-accept proxy: number of accepted changes whose single-step adjusted delta was under 1 sigma_used. Prediction: A more than 3 per run; verified arms at most 1.

H4 (descriptive, routing). D reaches a sealed improvement within 0.01 of C (noninferiority margin) at lower model cost per accepted change; served fraction of designated open-model calls at least 0.8 in clean D runs.

H5 (descriptive). Open-model coders qualify on constant and schedule edits (Wilson lower bound at least 0.6) and not on structural edits.

Secondaries (at most 6, Benjamini-Hochberg q under 0.10, exploratory): gap_sel; gap_shard; accepts per run; integrity plus tripwire counts; model dollars per run; D cost per accepted change.

## Validity rules

- A loop run is valid if it completed at least 12 slots; shorter runs are reported but excluded from H1 (intention-to-treat sensitivity analysis includes them).
- A D run is "degraded" when under 80 percent of designated open-model calls were served by open models; degraded runs are excluded from H4 and included in H1 with a flag.
- Live sigma: if the sd of sentinel runs over the window differs from the noise-job residual sd by more than 25 percent, all B, C, D decisions are replayed offline with the sentinel sigma and both versions are reported.
- Any change to prompts, models.json, routing_table.json, acceptance thresholds, N_SLOTS or arm counts after launch is a deviation and is logged below.

## Analysis plan

analysis.py reads every state.json, ledger.jsonl, confirmation result and sentinel, produces: table of loop runs (arm, slots, accepts, flags, claimed, confirmed official, confirmed sealed, improvement, model dollars, served fraction), H1 permutation test and interval, H2 to H5 estimates with 90 percent bootstrap intervals, figures (improvement by arm, claimed versus confirmed per run, cost versus improvement, routing shares in D), and a deviations table.

## Clarifications fixed before launch

- gap_sel is on the gain scale: claimed_gain (anchor mean minus claimed best official) minus confirmed_gain_official (same-wave anchor official minus confirmed official); positive means an optimistic claim. H2 predictions refer to this sign.
- Secondaries 1 to 5 contrast D versus A; secondary 6 (dollars per accepted change) contrasts non-degraded D versus C. Secondaries use nominal dollars; H4 reports plan and nominal separately.
- H5 is read from the calibration results (runs/calibration), with the ledger proxy as a cross-check.
- Degraded D runs stay in H1 (flagged) and in secondaries 1 to 5; they are excluded from H4 and secondary 6.
- Confirmation values are compared within wave and are not GPU-offset adjusted; decisions inside loops are offset adjusted.
- If the live-sigma check fires, the offline replay of decisions is done by rerunning acceptance.decide_verified over each ledger with the sentinel sigma on the recorded values; candidates that were never replicated under the original rule are replayed as rejects when the new z1 band would have required a replicate.

## Pre-launch amendments (Oct 21, 02:50, before any R1 loop run)

1. Anchor and noise source. The 16-run sbatch noise job is not comparable with runs under the production worker (worker runs score about 0.03 lower at identical data, code and step counts; see loop/runs/T1/STRESS.md). The anchor is therefore the mean of 16 live launch sentinels run through the worker four-wide with fresh seeds: anchor_mean 1.114885, n 16, sd 0.004125 (loop/runs/R1/anchor.json). The noise job stays a reference measurement only.
2. sigma_used = 1.15 x 0.004125 = 0.004743 (was 0.006736 from the noise job). The tripwire single-step threshold becomes 5 x sigma_used x sqrt(2) = 0.0335; the cumulative threshold stays 0.03. The live-sigma check compares the sd of running sentinels with 0.004125 (not 0.005857), same 25 percent band.
3. GPU offsets from the live sentinels (one-way ANOVA across GPUs F(3,12) = 8.58, above the 0.05 critical value 3.49): GPU0 +0.002617, GPU1 -0.004187, GPU2 -0.002207, GPU3 +0.003776, frozen in anchor.json and subtracted from every value used in a B, C, D decision. Caveat recorded: the offsets do not track step counts and plausibly come from per-GPU compile caches, so they may not transfer to modified code; the analysis reports decisions replayed with zero offsets as a sensitivity.
4. Anchor n for the initial incumbent is 16 (was 24).
5. Routing table frozen from calibration (loop/runs/calibration/routing_table.json, copy in loop/runs/R1/routing_table_frozen.json): coder constant qwen35-4b, coder schedule gemma4-12b, coder structural haiku, repairer haiku, planner_cheap qwen35-4b, analyst sonnet (no model met the analyst bar), voter gemma4-12b, critic opus; r1-qwen3-8b qualified for nothing. Calibration cost $2.89 plan, 504 trials.
6. Integration test T1 found and fixed an engine bug (every clean result flagged harness_missing because the harness always writes an error key with value None); regression test added; T2 rerun with the fix.

## Deviations

Launched: controller started 2026-10-21 02:51:20 EDT (06:51:20 UTC) (plan loop/runs/R1/plan.json, 22 loop runs, 20 slots each, window 24 h). The last integration loops (T2) were still finishing their in-flight slots on the shared queue during the first minutes; this delays all arms equally.

Deviation 1 (Oct 21): all 22 loop runs finished at 16:46 EDT (20:46:53 UTC), 13.9 h after launch, with 440 slots and 36 sentinels; the confirmation block started at 18:28 EDT instead of immediately, because the launcher watching for the controller's exit matched its own process name and never fired. The GPUs were idle for about 100 minutes. Confirmation waves still run four-wide with a same-wave anchor, so arm contrasts are unaffected; the gap only shifts the block later in the window.

(no other deviations)

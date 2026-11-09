# analysis.py: analysis for the pre-registered routed-autoresearch experiment

`analysis.py` is the analysis step for run R1 of the routed research loop. It implements section 12 of `loop/design/DESIGN.md` and the analysis plan of `loop/design/PREREG.md`. Needs only numpy, pandas and matplotlib. Reads a run directory, recomputes everything from raw records (never trusts confirmation/summary.json), writes a table, a JSON of all statistics, six figures and a markdown report. Nothing in the run directory is modified.

## Run

From the repository root:

    .venv/bin/python -m loop.core.analysis --run loop/runs/<TAG> --out loop/runs/<TAG>/analysis

Options: `--seed N` (default 1), `--prereg PATH` (default loop/design/PREREG.md, source of the copied Deviations section), `--calibration PATH` (optional JSON {"<model>": {"<edit_class>": [passes, n]}} for the H5 table).

Tests: `.venv/bin/python -m pytest loop/core/tests/test_analysis.py -q` (38 tests; a seeded synthetic run with D better than A by 0.008, including a frozen routing table and a few health-check reroutes in D, and a no-effect run).

Inputs: `<loop_id>/state.json`, `<loop_id>/ledger.jsonl`, `confirmation/results.jsonl`, `sentinels.jsonl`, `anchor.json`, `plan.json` (sigma_used, default 0.006736, and the D loop configs that locate the routing table), and a frozen routing table (see Served fraction). Outputs: report.md, loop_runs.csv, results.json, fig_improvement_by_arm.png, fig_claimed_vs_confirmed.png, fig_cost_vs_improvement.png, fig_routing_shares_D.png, fig_sentinels.png, fig_ledger_timeline.png.

## Table columns

arm, loop_id, status; slots_completed (state.slot_index, else ledger count); valid (>= 12 slots); accepts, replicates, audits; audit_rejects (kind audit, decision starting with reject), audit_accepts (kind audit, decision starting with accept), would_be_keeps (accepts + audit_rejects: what a keep-if-lower rule would have kept had audits not run; for arm A the pure Karpathy keep count), audit_reject_rate (audit_rejects / audits, NaN with no audits); integrity_flags, tripwires; lint_failures; bad_accepts (accepted records whose incumbent mean moved by less than sigma_used, the H3 proxy); claimed_best_official (anchor mean when null); n_confirm; confirmed_official_mean, confirmed_sealed_mean; improvement (primary: mean over confirmation runs of same-wave anchor sealed minus run sealed, higher is better); claimed_gain (anchor mean minus claimed); confirmed_gain_official; gap_sel = claimed_gain minus confirmed_gain_official (positive = optimistic claim); gap_shard = mean(official minus sealed) minus the same-wave anchor's (official minus sealed); model_usd_plan, model_usd_nominal, model_calls (from ledger model_calls); served_fraction (state counters), served_fraction_ledger (ledger and routing table, arm D only), degraded (arm D with served_fraction_ledger under 0.8); usd_per_accept.

Confirmation pairing: each run pairs with the mean of usable anchors in its own wave, falling back to all anchors. No GPU-offset adjustment on confirmations.

## Statistics

- H1: exact permutation on mean(D) minus mean(A) over valid runs when C(n, k) <= 20000 (8 vs 8 = 12870 splits), else 20000 seeded Monte Carlo splits. Decision: D better if p < 0.05 and estimate > 0.003; A better symmetric; else inconclusive. 90 percent interval by test inversion under a constant-shift model. Sensitivities: intention to treat, ITT with an OLS slots adjustment, and excluding degraded D runs.
- Rung contrasts B minus A, C minus B, D minus C: difference of means with a 90 percent percentile bootstrap (10000 draws), no p-values.
- H2 (gap_sel) and H3 (bad_accepts): per-arm means with 90 percent bootstrap intervals and checks of the stated predictions.
- H4: non-degraded D versus C; noninferior if the lower interval end is above -0.01; dollars per accepted change in plan and nominal terms.
- H5: Wilson 95 percent lower bound per edit class against 0.6 (structural 0.75), from a ledger proxy and the optional calibration table.
- Secondaries (exactly six, Benjamini-Hochberg q < 0.10): gap_sel, gap_shard, accepts, integrity plus tripwires, nominal dollars per run (all D minus A); nominal dollars per accepted change (non-degraded D minus C). The same permutation test as H1 on valid runs.
- Live sigma: sentinel sd after removing the frozen GPU offsets against 0.005857, flagged outside the 0.75 to 1.25 ratio. The offline replay PREREG requires when flagged is not implemented here; the report flags it.
- Power note: minimum detectable difference at 80 percent power, 2.80 x pooled within-arm sd x sqrt(1/nA + 1/nD).

## Assumptions

slot_index is the count of charged slots; claimed values are on the anchor-mean scale; ledger kind strings contain "replic" or "audit"; flags containing "tripwire" are tripwires and every other flag is an integrity flag; model dollars come only from ledger model_calls (controller-level calls are not included); an open model is any served name without sonnet, haiku, opus or claude in it; valid runs with no confirmation data stay in the table but drop out of the tests; degraded D runs stay in secondaries 1 to 5 and are excluded from the sixth.

## Audit columns

The live run showed the cumulative tripwire (gain over the anchor above 0.03) is crossed early by every arm, after which every would-be accept is audited in all arms, including the control. The four audit columns separate what audits removed from what survived. Audits in arm A are replication the control would not otherwise have had, so A's `accepts` are already filtered; `audit_rejects` in A estimate how many noise keeps the plain rule would have accepted, and `would_be_keeps` is the keep count of the unfiltered rule. The per-arm summary in report.md carries the five audit columns (audits, the four above) as means over valid runs, with a short paragraph repeating this reading; `audit_reject_rate` is averaged over the valid runs that had audits.

## Served fraction

The engine increments `designated_open_calls` only for calls it actually sent to an open model. When the pre-slot health check finds an open model down, it routes the role straight to the Claude fallback and logs `{slot, model, fallback}` in `state.fallbacks` without touching that counter, so the state-based `served_fraction` (open_served_calls / designated_open_calls) overstates. `served_fraction_ledger` is recomputed for arm D from the ledger and the frozen routing table:

- A ledger `model_calls` entry counts only if its role is in coder, repairer, planner_refresh, analyst, voter (planner_backlog, critic and any other role are ignored).
- Its designated model is the table's entry for the role, mirroring the engine's `_route`: coder goes through `coder_by_class` using the record's `edit_class`, planner_refresh goes to `planner_cheap` (a `planner_refresh` entry first, then `planner`), the rest use the table key of the same name. A role with no entry designates nothing.
- The call is designated-open when that model is open (name without sonnet, haiku, opus or claude) and open-served when `served_model` is open. `served_fraction_ledger` = open-served / designated, empty when nothing is designated.
- `degraded` uses `served_fraction_ledger` (under 0.8). The state-based `served_fraction` stays as a second column, and the H4 table reports the mean of both.

The table is the `routing` of the first D loop config listed in plan.json that gives a readable table (the entry's `config` file; `routing` is a path, tried as given and then relative to the repo root, or an inline dict), or plan.json's own `routing`; else `<RUN>/routing_table_frozen.json`; else `<RUN>/../calibration/routing_table.json`. If none is found, `served_fraction_ledger` falls back to the state counters (equal to `served_fraction`), `meta.routing_table` in results.json is null and the report says so. The report names the table it used. Example: loop/runs/T2/D_T2 has one reroute (qwen35-4b to haiku in slot 1) and shows state 85/85 = 1.0 but ledger 85/87 = 0.977, because the two constant-edit coder calls in slot 0 were designated qwen35-4b yet served by haiku.

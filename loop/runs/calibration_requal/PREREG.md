# Preregistration: haiku requalification (structural coder, lint repairer)

Written before any model call of this tryout. The decision rule below was fixed in advance and is copied here unchanged.

- Recorded at: 2026-11-06T04:05:28Z (host clock; local 2026-11-05 23:05:28 EST, host spruce)
- Spec file: loop/runs/calibration_requal/specs_requal.json
- Spec sha256 (equals specs.sha256, checked just before this entry): 6e5ff4d10cc827bbdf770e590a2625d73462978c0025de7b853275fcecef297a
- Spec validator (validate_specs.py) passed just before this entry: ALL CHECKS PASS
- Items: 30 structural coder items (rstruct-01 to rstruct-30) and 20 repair items (rrep-01 to rrep-20), all new and held out from the October calibration.
- Run: model haiku only, 2 samples per item, the harness's own production prompts (calibration.run), out dir loop/runs/calibration_requal. 100 planned calls, hard cap 130 Claude calls for this track. Any served model other than haiku is an infra trial, not a pass or a fail.

## Decision rule (primary)

New items only, item level.

- For each cell (structural coder, lint repairer): k = sum over the items of the fraction of that item's 2 samples that pass, n = number of items (30 structural, 20 repair), lower bound lb = calibration.wilson_lower(k, n) (z = 1.96, fractional k allowed, computed in floating point with no rounding before the comparison).
- Haiku stays the structural coder if lb >= 0.75.
- Haiku stays the lint repairer if lb >= 0.6.
- Otherwise that cell goes to sonnet.

## Secondary (reported only, never used for the decision)

- Pooled with the old October trials (old items all passed: structural 8 items, 16 trials; repair 6 items, 12 trials), at item level (structural n = 38, repair n = 26) and at trial level (structural 76 trials, repair 52 trials with the new 2 samples per item), using the same Wilson lower bound.
- Per-item pass fractions, failure reasons, cost and latency.

## Bookkeeping fixed in advance

- Infra trials (call failures, context errors, or a served model other than haiku) are excluded from n and are retried by the harness on resume. If more than 10 percent of trials are infra, the track stops and reports instead of deciding.
- An item counts only with both of its samples resolved as real (pass or fail). An item with an unresolved sample after retries blocks the decision and is reported.
- The routing table that calibration.run may write into this directory is not used for the decision.

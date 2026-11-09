# P3 result: thinking harder versus asking a stronger model, for a cheap critic

Run Nov 6 2026 under the frozen preregistration in PREREG.md (sha256 3390824c..., verified before analysis; analysis code sha256 a4b139f1... unchanged). Haiku reviewed each of the 299 scorable AR3 items once at four reasoning-effort levels: default, low, medium and high. That is 1,196 calls, all valid, none served by a fallback model, for about $1.70 at list price. Escalation used the stored Sonnet pass from AR3. Analysis output: analysis.json.

## Preregistered tests (test split, 224 items in 48 entries)

| Test | Weighted (primary) | Unweighted (sensitivity) | Robust? |
|---|---|---|---|
| H1: AUROC of stated confidence, high minus low | -0.028, 90% interval -0.086 to 0.024: inconclusive | +0.091, 0.023 to 0.153: high better | No, the views disagree |
| H2: accuracy at equal dollars, escalate-to-Sonnet from low minus Haiku high alone | +0.168, 0.073 to 0.267: escalation better | +0.013, -0.008 to 0.036: no difference | No, the views disagree |

By the preregistered rule, neither hypothesis has a robust verdict. The weighted view gives clean entries most of the weight (AR3's design weights, prevalence 0.25). Sonnet is far better than Haiku on clean entries and no better on planted errors, so escalation helps only where clean entries count.

## Descriptive results (not preregistered tests)

| Level | Accuracy | Weighted accuracy | Clean entries right (of 34) | Planted errors caught (of 190) | Mean $ per call | Mean output tokens |
|---|---|---|---|---|---|---|
| low | 0.82 | 0.39 | 8 | 176 | 0.00106 | 1,762 |
| default | 0.80 | 0.28 | 3 | 177 | 0.00154 | 2,328 |
| medium | 0.82 | 0.26 | 2 | 181 | 0.00121 | 2,332 |
| high | 0.81 | 0.26 | 2 | 180 | 0.00202 | 3,810 |
| Sonnet, one pass (AR3) | 0.83 | 0.68 | | | 0.01571 | |

- **More thinking did not buy accuracy.** Overall accuracy is flat from low to high (0.82 to 0.81), while cost rises by 90 percent and output tokens more than double.
- **More thinking made false alarms worse.** On correct write-ups Haiku was right 8 times of 34 at low effort and twice at high. Paired by entry, low was right and high wrong on 7 entries, and the reverse on 1 (two-sided sign test p of about 0.07). Detection of planted errors moved from 93 to 95 percent. This pattern was not a preregistered test.
- **The burn split agrees on cost and tokens.** Its tests are inconclusive or show no difference (75 items).

## What this means for the project question

For this cheap critic, buying more thinking was the wrong lever. It cost more, caught slightly more planted errors, and raised false alarms on correct work. Where correct work matters (the weighted view), sending the least-confident cases to a stronger model did better at the same price. Neither statement is a robust preregistered result, because the two weightings disagree, and the false-alarm pattern is post hoc on 34 clean entries.

## Limits

- **One cheap model, one task.** The test split has 34 clean entries and 190 plants.
- **Escalation reuses stored Sonnet passes** from October 17, made under an older CLI version.
- **Haiku still thinks at low effort,** with a median of 1,172 thinking tokens on Day 0. So the comparison is roughly half as much thinking against twice as much, not thinking against none.

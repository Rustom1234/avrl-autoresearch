# E10p0 result

## base

| metric | value |
|---|---|
| n problems | 464 |
| mean pass rate | 0.0121 (95% CI 0.0078-0.0170) |
| fraction any_correct (solvable at pass@8) | 0.0668 (95% CI 0.0453-0.0905) |
| signature-vote proxy error rate | 0.9871 (95% CI 0.9763-0.9957) |
| visible-test (first assert) per-completion disagreement rate | 0.0081 (95% CI 0.0054-0.0110) |
| AUC: agreement predicts proxy_error | 0.1337 (95% CI 0.0461-0.2370) |
| AUC: visible_majority predicts proxy_error | 0.9887 (95% CI 0.9637-1.0000) |

## instruct

| metric | value |
|---|---|
| n problems | 464 |
| mean pass rate | 0.0932 (95% CI 0.0752-0.1123) |
| fraction any_correct (solvable at pass@8) | 0.2328 (95% CI 0.1940-0.2737) |
| signature-vote proxy error rate | 0.8728 (95% CI 0.8426-0.9009) |
| visible-test (first assert) per-completion disagreement rate | 0.0234 (95% CI 0.0159-0.0321) |
| AUC: agreement predicts proxy_error | 0.6795 (95% CI 0.6056-0.7465) |
| AUC: visible_majority predicts proxy_error | 0.9815 (95% CI 0.9701-0.9908) |

Caveats: error-rate and AUC bootstraps resample GROUPS (one problem = one group, n=1000), not individual rollouts. `agreement`/`visible_majority`/`proxy_error` are defined in E10p0_rollouts.py; the regime diagnostic CLI output for the same two columns is at `<model>/diag/diagnosis.json`.

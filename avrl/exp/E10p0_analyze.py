"""E10p0 analysis: for the base and Instruct OLMo-2-0425-1B rollouts on the E10 MBPP training
problems (avrl/runs/E10p0/{base,instruct}/{rollouts,groups}.parquet, from E10p0_rollouts.py),
report per-problem solvability, the signature-vote proxy's error rate, and the visible-test
(first assert) proxy's per-completion disagreement rate with the real asserts, each with a
1000-resample group bootstrap 95% CI. AUC of agreement and of visible_majority for predicting
proxy_error reuses avrl_router.diagnose's own _bootstrap_ci/diagnose() (not reimplemented here)
so the point estimate and CI match exactly what the separately-run diagnose CLI reports for the
same columns.

Writes avrl/runs/E10p0/result.json, avrl/runs/E10p0/RESULT.md,
avrl/runs/E10p0/base_solvable_task_ids.json (task_ids with any_correct=1 for the base model --
the curriculum for the next experiment), and, via the diagnose CLI (subprocess, same
interpreter), avrl/runs/E10p0/<model>/diag/diagnosis.json for each model.

Usage (CPU, .venv/bin/python, cwd=active-audit):
  avrl/exp/E10p0_analyze.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # active-audit/
AVRL = os.path.join(ROOT, "avrl")
RUN = os.path.join(AVRL, "runs", "E10p0")
sys.path.insert(0, AVRL)
from avrl_router.diagnose import _bootstrap_ci, diagnose  # noqa: E402

N_BOOT = 1000
CONFIDENCE_COLS = ["agreement", "visible_majority"]
BUDGETS = [0.1, 0.25, 0.5]

MODELS = ["base", "instruct"]
result = {}

for model in MODELS:
    mdir = os.path.join(RUN, model)
    G = pd.read_parquet(os.path.join(mdir, "groups.parquet"))
    n = len(G)

    def rate(col):
        vals = G[col].to_numpy().astype(float)
        point = float(vals.mean())
        lo, hi = _bootstrap_ci(lambda idx: vals[idx].mean(), n, N_BOOT, seed=0)
        return {"point": point, "ci95_lo": lo, "ci95_hi": hi}

    diag = diagnose(G, CONFIDENCE_COLS, "proxy_error", BUDGETS, target_corruption=0.02, n_boot=N_BOOT, seed=0)

    result[model] = {
        "n_problems": n,
        "mean_pass_rate": rate("pass_rate"),
        "frac_any_correct": rate("any_correct"),
        "signature_proxy_error_rate": rate("proxy_error"),
        "visible_proxy_error_rate": rate("visible_proxy_error"),
        "auc_agreement": diag["confidence_columns"]["agreement"]["auc"],
        "auc_visible_majority": diag["confidence_columns"]["visible_majority"]["auc"],
    }

    # Separately, run the regime diagnostic CLI (same command/args as the task spec) so
    # avrl/runs/E10p0/<model>/diag/diagnosis.json exists as its own reproducible artifact,
    # independent of the in-process diagnose() call above.
    diag_out = os.path.join(RUN, model, "diag")
    subprocess.run(
        [sys.executable, "-m", "avrl_router.diagnose",
         "--groups", os.path.join("runs", "E10p0", model, "groups.parquet"),
         "--confidence", ",".join(CONFIDENCE_COLS),
         "--error", "proxy_error",
         "--budgets", ",".join(str(b) for b in BUDGETS),
         "--out", os.path.join("runs", "E10p0", model, "diag")],
        cwd=AVRL, check=True,
    )

    if model == "base":
        solvable = sorted(int(t) for t in G.loc[G.any_correct == 1, "task_id"])
        json.dump(solvable, open(os.path.join(RUN, "base_solvable_task_ids.json"), "w"), indent=1)

json.dump(result, open(os.path.join(RUN, "result.json"), "w"), indent=1)


def fmt(d):
    return f"{d['point']:.4f} (95% CI {d['ci95_lo']:.4f}-{d['ci95_hi']:.4f})"


lines = ["# E10p0 result\n"]
for model in MODELS:
    r = result[model]
    lines.append(f"## {model}\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| n problems | {r['n_problems']} |")
    lines.append(f"| mean pass rate | {fmt(r['mean_pass_rate'])} |")
    lines.append(f"| fraction any_correct (solvable at pass@8) | {fmt(r['frac_any_correct'])} |")
    lines.append(f"| signature-vote proxy error rate | {fmt(r['signature_proxy_error_rate'])} |")
    lines.append(f"| visible-test (first assert) per-completion disagreement rate | {fmt(r['visible_proxy_error_rate'])} |")
    lines.append(f"| AUC: agreement predicts proxy_error | {fmt(r['auc_agreement'])} |")
    lines.append(f"| AUC: visible_majority predicts proxy_error | {fmt(r['auc_visible_majority'])} |")
    lines.append("")

lines.append("Caveats: error-rate and AUC bootstraps resample GROUPS (one problem = one group, "
             "n=1000), not individual rollouts. `agreement`/`visible_majority`/`proxy_error` are "
             "defined in E10p0_rollouts.py; the regime diagnostic CLI output for the same two "
             "columns is at `<model>/diag/diagnosis.json`.")
open(os.path.join(RUN, "RESULT.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

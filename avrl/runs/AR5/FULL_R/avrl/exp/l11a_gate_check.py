"""L11a regression gate: with target n_sim 0, split seed 0, cheap_r features, cell100, every policy that runs/L1a/sim_<tag>_cheap_cell100_s0.json holds must reproduce its per-seed hits exactly
(n_sim, n_buckets and hits too); recal_charged_25/50 are new and are not compared. Usage: l11a_gate_check.py (exit 3 if any difference)"""
import json, sys
tot = bad = 0
for tag in ("3B_q_corrupted", "7B_q_corrupted", "3B_q_judgewrong", "7B_q_judgewrong"):
    A, B = json.load(open(f"runs/L1a/sim_{tag}_cheap_cell100_s0.json")), json.load(open(f"runs/L11a/gate/g_{tag}.json"))
    for k in ("n_fit", "n_sim", "n_buckets", "hits"):
        tot += 1
        if A[k] != B[k]: bad += 1; print("DIFF", tag, k, A[k], B[k])
    for bud in ("0.1", "0.25", "0.5"):
        for p, v in A[f"budget_{bud}"].items():
            tot += 1
            if p not in B[f"budget_{bud}"] or v["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", tag, bud, p)
out = {"policy_budget_cells_compared": tot, "differences": bad, "passed": bad == 0}
json.dump(out, open("runs/L11a/gate/gate_check.json", "w")); print(out)
sys.exit(0 if bad == 0 else 3)

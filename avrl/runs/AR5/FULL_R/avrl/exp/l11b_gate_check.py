"""L11b regression gate: the no-oracle replay (cheap_r only) on the augmented tables must reproduce the runs/L1a sim_<tag>_cheap_<layout>_s<split>.json files with 0 per-seed differences
(cell100 splits 0-3 and cell30 split 0, both rungs, every policy x budget). Usage: l11b_gate_check.py (exit 3 on any difference)"""
import json, sys
tot = bad = files = 0
for tag in ("3B_q_corrupted", "7B_q_corrupted"):
    for lay, sps in (("cell100", (0, 1, 2, 3)), ("cell30", (0,))):
        for sp in sps:
            n = f"sim_{tag}_cheap_{lay}_s{sp}.json"; A, B = json.load(open(f"runs/L1a/{n}")), json.load(open(f"runs/L11b/{n}")); files += 1
            for k in ("n_fit", "n_fit_pos", "n_sim", "n_buckets", "hits"):
                if A[k] != B[k]: bad += 1; print("DIFF header", n, k, A[k], B[k])
            for bud in ("0.1", "0.25", "0.5"):
                assert set(A[f"budget_{bud}"]) == set(B[f"budget_{bud}"]), n
                for p, v in A[f"budget_{bud}"].items():
                    tot += 1
                    if v["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", n, bud, p)
out = {"files_compared": files, "policy_budget_cells_compared": tot, "differences": bad, "passed": bad == 0}
json.dump(out, open("runs/L11b/gate/gate_check.json", "w")); print(out)
sys.exit(0 if bad == 0 else 3)

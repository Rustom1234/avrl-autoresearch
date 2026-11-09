"""L1b regression gate: at split seed 0 the existing policies must reproduce runs/E04/sim_<name>_cell100_cheap.json exactly (per-seed hits). Usage: l1b_gate_check.py"""
import json
pairs = {"E01": ("runs/E04/sim_E01_cell100_cheap.json", "runs/L1b/gate/g_E01.json"), "3B": ("runs/E04/sim_Qwen2.5-3B-Instruct_cell100_cheap.json", "runs/L1b/gate/g_3B.json"),
         "7B-AWQ": ("runs/E04/sim_Qwen2.5-7B-Instruct-AWQ_cell100_cheap.json", "runs/L1b/gate/g_7B.json")}
tot = bad = 0
for k, (a, b) in pairs.items():
    A, Bj = json.load(open(a)), json.load(open(b)); assert (A["n_sim"], A["n_buckets"], A["hits"]) == (Bj["n_sim"], Bj["n_buckets"], Bj["hits"])
    for bud in ("0.1", "0.25", "0.5"):
        for p, v in A[f"budget_{bud}"].items():
            tot += 1; w = Bj[f"budget_{bud}"][p]
            if v["per_seed"] != w["per_seed"]: bad += 1; print("DIFF", k, bud, p)
out = {"policy_budget_cells_compared": tot, "differences": bad, "passed": bad == 0}
json.dump(out, open("runs/L1b/gate/gate_check.json", "w")); print(out)

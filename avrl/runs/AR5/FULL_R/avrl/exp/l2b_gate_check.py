"""L2b regression gate: with m = 0 at split seed 0 the reference policies must reproduce runs/L2a/sim_<name>_split0.json per-seed hits exactly.
Compared: uniform, static, oracle, thr_agreement, greedy_online, greedy_n25_R25, greedy_seeded_n25_R25, and recal/seeded/split_charged_n for n shared with the L2b grid (25, 50, 100, 200).
Usage: l2b_gate_check.py <gate_dir> <l2a_dir>   (gate_dir holds g_3B.json and g_7B.json)"""
import json, sys, os
gd, ld = sys.argv[1], sys.argv[2]
pairs = {"3B": ("g_3B.json", "sim_Qwen2.5-3B-Instruct_split0.json"), "7B-AWQ": ("g_7B.json", "sim_Qwen2.5-7B-Instruct-AWQ_split0.json")}
NAMED = ("uniform", "static", "oracle", "thr_agreement", "greedy_online", "greedy_n25_R25", "greedy_seeded_n25_R25"); NSHARED = {25, 50, 100, 200}
def REF(p):
    if p in NAMED: return True
    f, _, n = p.rpartition("_charged_")
    return f in ("recal", "seeded", "split") and n.isdigit() and int(n) in NSHARED
tot = bad = 0; diffs = []
for k, (new, old) in pairs.items():
    A, Bj = json.load(open(os.path.join(ld, old))), json.load(open(os.path.join(gd, new)))
    for f in ("n_sim", "n_buckets", "hits", "split_seed"):
        tot += 1
        if A[f] != Bj[f]: bad += 1; diffs.append(f"{k} {f}: L2a {A[f]} vs L2b {Bj[f]}")
    for bud in ("0.1", "0.25", "0.5"):
        for p, v in A[f"budget_{bud}"].items():
            if not REF(p): continue
            tot += 1; w = Bj[f"budget_{bud}"].get(p)
            if w is None or v["per_seed"] != w["per_seed"]: bad += 1; diffs.append(f"{k} {bud} {p}: L2a {v['per_seed']} vs L2b {None if w is None else w['per_seed']}")
        for p in NAMED:  # every named reference policy must exist on both sides
            if p not in A[f"budget_{bud}"]: tot += 1; bad += 1; diffs.append(f"{k} {bud} {p}: missing in L2a")
for d in diffs: print("DIFF", d)
out = {"checks_compared": tot, "differences": bad, "passed": bad == 0 and tot > 0, "diffs": diffs}
json.dump(out, open(os.path.join(gd, "gate_check.json"), "w"), indent=1); print({k: v for k, v in out.items() if k != "diffs"})

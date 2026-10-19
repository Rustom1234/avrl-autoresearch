"""L4a regression gate: at m = 500, subset seed 0, both judge labels, every reference policy must reproduce runs/L2b/sim_<name>_m500_s0.json per-seed hits exactly.
Required (the gate): uniform, oracle, thr_agreement, greedy_online, greedy_n10_R5, greedy_n25_R25, and recal/seeded/split_charged_n for every n present in the L2b file (all n shared with L4a),
plus the header fields n_sim, n_buckets, hits, subset_seed, m. Extra checks (reported, not part of the gate): every other policy in the L2b file (static, greedy_seeded_n25_R25, the other grid members).
Every required policy must exist on both sides. Policies that exist only in L4a (new grid, offset arms, n = budget // 2 extras) are not compared.
Usage: l4a_gate_check.py <gate_dir> <l2b_dir>   (gate_dir holds g_3B.json and g_7B.json)"""
import json, sys, os
gd, ld = sys.argv[1], sys.argv[2]
pairs = {"3B": ("g_3B.json", "sim_Qwen2.5-3B-Instruct_m500_s0.json"), "7B-AWQ": ("g_7B.json", "sim_Qwen2.5-7B-Instruct-AWQ_m500_s0.json")}
NAMED = ("uniform", "oracle", "thr_agreement", "greedy_online", "greedy_n10_R5", "greedy_n25_R25")
def required(p):
    if p in NAMED: return True
    f, _, n = p.rpartition("_charged_")
    return f in ("recal", "seeded", "split") and n.isdigit()
tot = bad = xtot = xbad = 0; diffs = []; xdiffs = []
for k, (new, old) in pairs.items():
    A, Bj = json.load(open(os.path.join(ld, old))), json.load(open(os.path.join(gd, new)))
    for f in ("n_sim", "n_buckets", "hits", "subset_seed", "m"):
        tot += 1
        if A[f] != Bj[f]: bad += 1; diffs.append(f"{k} {f}: L2b {A[f]} vs L4a {Bj[f]}")
    for bud in ("0.1", "0.25", "0.5"):
        for p in NAMED:  # every named reference policy must exist on both sides
            for side, D in (("L2b", A), ("L4a", Bj)):
                if p not in D[f"budget_{bud}"]: tot += 1; bad += 1; diffs.append(f"{k} {bud} {p}: missing in {side}")
        for p, v in A[f"budget_{bud}"].items():
            w = Bj[f"budget_{bud}"].get(p); differ = w is None or v["per_seed"] != w["per_seed"]
            msg = f"{k} {bud} {p}: L2b {v['per_seed']} vs L4a {None if w is None else w['per_seed']}"
            if required(p):
                tot += 1
                if differ: bad += 1; diffs.append(msg)
            else:
                xtot += 1
                if differ: xbad += 1; xdiffs.append(msg)
for d in diffs: print("DIFF", d)
for d in xdiffs: print("EXTRA DIFF", d)
out = {"checks_compared": tot, "differences": bad, "extra_checks_compared": xtot, "extra_differences": xbad, "passed": bad == 0 and tot > 0, "diffs": diffs, "extra_diffs": xdiffs}
json.dump(out, open(os.path.join(gd, "gate_check.json"), "w"), indent=1); print({k: v for k, v in out.items() if k not in ("diffs", "extra_diffs")})

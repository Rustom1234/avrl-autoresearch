"""L5a regression gate: the L4a prompt set for (m = 250, subset seed 0), both judge labels, both per-step budgets (62 = budget_0.25, 125 = budget_0.5).
thr_agreement, uniform, greedy_n3_R1 and offset_n0_R5_lam10 computed through the copied functions must reproduce runs/L4a/sim_<judge>_m250_s0.json per-seed hits exactly (4 policies x 8 seeds x 2 judges x 2 budgets = 128 comparisons).
Also reported in the same file and also required to be 0 (implementation self-check of the new carry code): carry_plain on a single batch vs run_online, carry_offset / carry_window125 on a single batch vs run_offset,
carry_offset with lam = inf vs thr_agreement (single batch, and over the real 12-batch layout of order seed 0).
Usage: l5a_gate_check.py <gate_dir> <L4a_dir>   (gate_dir holds g_<judge>_B<budget>.json)"""
import json, sys, os
gd, ld = sys.argv[1], sys.argv[2]
JUDGES = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct-AWQ"); BKEY = {62: "budget_0.25", 125: "budget_0.5"}
tot = bad = stot = sbad = 0; diffs = []; sdiffs = []
for j in JUDGES:
    L4 = json.load(open(os.path.join(ld, f"sim_{j}_m250_s0.json")))
    for B, bk in BKEY.items():
        G = json.load(open(os.path.join(gd, f"g_{j}_B{B}.json")))
        assert G["budget"] == B and G["m"] == 250 and G["subset_seed"] == 0
        for p, mine in G["policies"].items():
            ref = L4[bk][p]["per_seed"]; tot += len(ref)
            if mine != ref: bad += sum(a != b for a, b in zip(mine, ref)) + abs(len(mine) - len(ref)); diffs.append(f"{j} B{B} {p}: L4a {ref} vs L5a {mine}")
        for k, v in G["selfcheck"].items():
            stot += len(v["carry"])
            if v["max_abs_diff"] != 0: sbad += sum(a != b for a, b in zip(v["carry"], v["ref"])); sdiffs.append(f"{j} B{B} {k}: ref {v['ref']} vs carry {v['carry']}")
for d in diffs: print("DIFF", d)
for d in sdiffs: print("SELFCHECK DIFF", d)
out = {"checks_compared": tot, "differences": bad, "selfcheck_compared": stot, "selfcheck_differences": sbad, "passed": bad == 0 and sbad == 0 and tot == 128 and stot > 0, "diffs": diffs, "selfcheck_diffs": sdiffs}
json.dump(out, open(os.path.join(gd, "gate_check.json"), "w"), indent=1); print({k: v for k, v in out.items() if k not in ("diffs", "selfcheck_diffs")})

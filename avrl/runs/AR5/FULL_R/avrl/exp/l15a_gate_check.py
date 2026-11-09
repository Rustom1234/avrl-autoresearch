"""L15a gates (run from avrl/). Usage: l15a_gate_check.py pre | ident | post
pre  : gate 1 (regression). The gate jobs (l15a_sim_alloc.py, STALE=none, refits on, n_sim 2250, split 0, subsample 0, both rungs) must reproduce runs/L13a/sim_<tag>_n2250_sp0_sub0.json with 0 per-seed
       differences in uniform, static, greedy_online, recal_free_200 and warm_online_200 at all three budgets (n_fit, n_sim, n_buckets, hits too). Writes runs/L15a/gate/gate_check.json. exit 3 = VOID.
ident: gate 2 (identity). The identity jobs (norefit flag, n_sim 2250, split 0, subsample 0, both rungs, all four STALE variants) must give warm_online_200 == recal_free_200 per policy seed at all three
       budgets under every STALE variant (also the w25 twin == warm_online_200, since without refits the weight is never used). Also checks that the corruption is really applied: for flip25, flip50 and
       lowlen recal_free_200 must differ from the clean L13a recal_free_200 in at least one budget cell of each rung. Reports one-class free draws. Writes gate/gate_check_ident.json. exit 3 = VOID.
post : after the grid, the 128 files must exist with the six arms at all three budgets and refits on; the 32 STALE=none files must equal the matching runs/L13a files per seed in uniform, static,
       greedy_online, recal_free_200 and warm_online_200; in every STALE variant uniform, static and greedy_online (which do not read the free labels) must equal L13a per seed, and n_sim / n_fit / hits /
       n_buckets must match L13a (same population). Writes gate/gate_check_post.json. exit 3 = VOID."""
import json, sys, os
mode = sys.argv[1]
TAGS = ("3B_q_corrupted", "7B_q_corrupted"); STALES = ("none", "flip25", "flip50", "lowlen"); BUDS = ("0.1", "0.25", "0.5")
ARMS6 = ("uniform", "static", "greedy_online", "recal_free_200", "warm_online_200", "warm_online_200_w25")
ARMS_NOFREE = ("uniform", "static", "greedy_online"); ARMS_L13A = ARMS_NOFREE + ("recal_free_200", "warm_online_200")
jl = lambda p: json.load(open(p))


def diffs(A, B, name, arms):
    """A = L13a file, B = L15a file; every arm in `arms` must have identical per-seed hits at all budgets; population metadata must match."""
    tot = bad = 0
    for k in ("n_fit", "n_sim", "n_buckets", "hits"):
        tot += 1
        if A[k] != B[k]: bad += 1; print("DIFF", name, k, A[k], B[k])
    for bud in BUDS:
        for p in arms:
            tot += 1
            if p not in B[f"budget_{bud}"] or A[f"budget_{bud}"][p]["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", name, bud, p)
    return tot, bad


os.makedirs("runs/L15a/gate", exist_ok=True)
if mode == "pre":
    tot = bad = 0
    for tag in TAGS:
        A, B = jl(f"runs/L13a/sim_{tag}_n2250_sp0_sub0.json"), jl(f"runs/L15a/gate/g_{tag}_none.json")
        assert B["stale"] == "none" and B["warm_refit"] is True, "gate-1 files must be STALE=none with refits on"
        t, b = diffs(A, B, tag, ARMS_L13A); tot += t; bad += b
    out = {"cells_compared": tot, "differences": bad, "passed": bad == 0}
    json.dump(out, open("runs/L15a/gate/gate_check.json", "w")); print(out); sys.exit(0 if bad == 0 else 3)

if mode == "ident":
    tot = bad = 0; one = {}; active = {}
    for tag in TAGS:
        A = jl(f"runs/L13a/sim_{tag}_n2250_sp0_sub0.json")
        for st in STALES:
            B = jl(f"runs/L15a/gate/i_{tag}_{st}.json"); assert B["warm_refit"] is False and B["stale"] == st, "identity files must be norefit and carry their STALE variant"
            one[f"{tag}|{st}"] = B["warm_oneclass_draws"]
            for bud in BUDS:
                tot += 2; w, f, w25 = (B[f"budget_{bud}"][k]["per_seed"] for k in ("warm_online_200", "recal_free_200", "warm_online_200_w25"))
                if w != f: bad += 1; print("IDENT DIFF warm vs free", tag, st, bud, w, f)
                if w25 != w: bad += 1; print("IDENT DIFF w25 vs warm", tag, st, bud, w25, w)
            if st != "none":
                n_diff = sum(A[f"budget_{bud}"]["recal_free_200"]["per_seed"] != B[f"budget_{bud}"]["recal_free_200"]["per_seed"] for bud in BUDS)
                active[f"{tag}|{st}"] = n_diff
                tot += 1
                if n_diff == 0: bad += 1; print("STALE NOT APPLIED", tag, st)
    out = {"cells_compared": tot, "differences": bad, "oneclass_free_draws": one, "stale_applied_budget_cells_changed_of_3": active, "passed": bad == 0}
    json.dump(out, open("runs/L15a/gate/gate_check_ident.json", "w"), indent=1); print(out); sys.exit(0 if bad == 0 else 3)

tot = bad = 0; n_files = 0; missing = 0
for tag in TAGS:
    for st in STALES:
        for sp in range(4):
            for sub in range(4):
                f = f"sim_{tag}_n2250_sp{sp}_sub{sub}.json"; A = jl(f"runs/L13a/{f}"); g = f"runs/L15a/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json"
                if not os.path.exists(g): missing += 1; print("MISSING FILE", g); continue
                B = jl(g); n_files += 1
                if B["stale"] != st or B["warm_refit"] is not True: missing += 1; print("BAD HEADER", g)
                for bud in BUDS:
                    for p in ARMS6:
                        if p not in B[f"budget_{bud}"]: missing += 1; print("MISSING ARM", g, bud, p)
                t, b = diffs(A, B, f"{tag}|{st}|sp{sp}|sub{sub}", ARMS_L13A if st == "none" else ARMS_NOFREE); tot += t; bad += b
out = {"files_found": n_files, "files_expected": 128, "cells_compared": tot, "differences": bad, "missing": missing, "passed": bool(bad == 0 and missing == 0 and n_files == 128)}
json.dump(out, open("runs/L15a/gate/gate_check_post.json", "w"), indent=1); print(out)
sys.exit(0 if out["passed"] else 3)

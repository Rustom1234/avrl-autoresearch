"""L13a gates (run from avrl/). Usage: l13a_gate_check.py pre | ident | post
pre  : gate 1. The gate jobs (l13a_sim_alloc.py with the nowarm flag, corrupted label, 3B and 7B-AWQ, target n_sim 2250 and target 0, split 0, subsample 0, cell30, cheap_r) must reproduce
       runs/L12c/sim_<tag>_n<target>_sp0_sub0.json with 0 per-seed differences in every arm (n_fit, n_sim, n_buckets, hits too). Writes runs/L13a/gate/gate_check.json. exit 3 = VOID.
ident: gate 2 (identity). The identity jobs (norefit flag: warm_online_<n> with the refits disabled, n_sim 2250, split 0, subsample 0, both rungs) must give warm_online_n == recal_free_n per policy seed
       for n in 100, 200, 400 at all three budgets. Also reports the one-class free draws (fall back to the cold random score, which recal_free does not match). Writes gate/gate_check_ident.json. exit 3 = VOID.
post : after the grid, every pre-existing arm of every grid file (all 104 jobs) must equal the matching runs/L12c file per seed (the warm arms draw from their own rng, so nothing else may move), and
       every file must carry the three warm arms with refits on at all three budgets. Writes gate/gate_check_post.json. exit 3 = VOID."""
import json, sys, os
mode = sys.argv[1]
TAGS = ("3B_q_corrupted", "7B_q_corrupted")
WARM = (100, 200, 400)


def diffs(A, B, name):
    """A = L12c file, B = L13a file; every policy of A must be in B with identical per-seed hits."""
    tot = bad = 0
    for k in ("n_fit", "n_sim", "n_buckets", "hits", "free_ns"):
        tot += 1
        if A[k] != B[k]: bad += 1; print("DIFF", name, k, A[k], B[k])
    for bud in ("0.1", "0.25", "0.5"):
        for p, v in A[f"budget_{bud}"].items():
            tot += 1
            if p not in B[f"budget_{bud}"] or v["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", name, bud, p)
    return tot, bad


os.makedirs("runs/L13a/gate", exist_ok=True)
if mode == "pre":
    tot = bad = 0
    for tag in TAGS:
        for tgt in (2250, 0):
            A, B = json.load(open(f"runs/L12c/sim_{tag}_n{tgt}_sp0_sub0.json")), json.load(open(f"runs/L13a/gate/g_{tag}_n{tgt}.json"))
            assert not any(p.startswith("warm_online_") for p in B["budget_0.25"]), "gate files must be produced with the nowarm flag"
            t, b = diffs(A, B, f"{tag} n{tgt}"); tot += t; bad += b
    out = {"policy_budget_cells_compared": tot, "differences": bad, "passed": bad == 0}
    json.dump(out, open("runs/L13a/gate/gate_check.json", "w")); print(out); sys.exit(0 if bad == 0 else 3)

if mode == "ident":
    tot = bad = 0; one = {}
    for tag in TAGS:
        B = json.load(open(f"runs/L13a/gate/i_{tag}_n2250.json")); assert B["warm_refit"] is False, "identity files must be produced with the norefit flag"
        one[tag] = B["warm_oneclass_draws"]
        for bud in ("0.1", "0.25", "0.5"):
            for n in WARM:
                tot += 1; a, f = B[f"budget_{bud}"][f"warm_online_{n}"]["per_seed"], B[f"budget_{bud}"][f"recal_free_{n}"]["per_seed"]
                if a != f: bad += 1; print("IDENT DIFF", tag, bud, n, a, f)
    out = {"warm_vs_free_cells_compared": tot, "differences": bad, "oneclass_free_draws": one, "passed": bad == 0}
    json.dump(out, open("runs/L13a/gate/gate_check_ident.json", "w")); print(out); sys.exit(0 if bad == 0 else 3)

tot = bad = 0; n_files = 0; missing = 0
for tag in TAGS:
    for tgt in (1125, 2250, 4500, 0):
        for sp in range(4):
            for sub in (range(4) if tgt else [0]):
                f = f"sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json"; A, B = json.load(open(f"runs/L12c/{f}")), json.load(open(f"runs/L13a/{f}")); n_files += 1
                t, b = diffs(A, B, f); tot += t; bad += b
                for bud in ("0.1", "0.25", "0.5"):
                    for n in WARM:
                        if f"warm_online_{n}" not in B[f"budget_{bud}"]: missing += 1; print("MISSING", f, bud, n)
                if B["warm_refit"] is not True: missing += 1; print("REFIT OFF", f)
out = {"files_compared": n_files, "policy_budget_cells_compared": tot, "differences": bad, "arms_unchanged": bad == 0, "missing_warm_arms": missing, "passed": bool(bad == 0 and missing == 0)}
json.dump(out, open("runs/L13a/gate/gate_check_post.json", "w"), indent=1); print(out)
sys.exit(0 if out["passed"] else 3)

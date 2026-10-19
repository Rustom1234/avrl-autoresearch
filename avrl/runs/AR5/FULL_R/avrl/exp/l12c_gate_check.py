"""L12c gates (run from avrl/). Usage: l12c_gate_check.py pre | post
pre : the gate jobs (l12c_sim_alloc.py with the recal_free policies skipped, corrupted label, 3B and 7B-AWQ, target n_sim 2250 and target 0, split 0, subsample 0, cell30, cheap_r) must reproduce
      runs/L11a/sim_<tag>_n<target>_sp0_sub0.json with 0 per-seed differences (n_fit, n_sim, n_buckets, hits too). Writes runs/L12c/gate/gate_check.json. exit 3 = VOID.
post: after the grid, (a) every existing arm of every grid file (all 104 jobs) must equal the matching runs/L11a file per seed (the new policies draw from their own rng, so nothing else may move), and
      (b) sanity: recal_free_400 must not exceed the free-label static ceiling by more than 2 SE, per rung x scale x budget cell: d = cluster mean of recal_free_400 minus static, SE = sd(d, ddof 1) / sqrt(clusters),
      check mean(d) <= 2 * SE. Writes runs/L12c/gate/gate_check_post.json. exit 3 = VOID."""
import json, sys, os
import numpy as np
mode = sys.argv[1]
TAGS = ("3B_q_corrupted", "7B_q_corrupted")


def diffs(A, B, name):
    """A = L11a file, B = L12c file; every policy of A must be in B with identical per-seed hits."""
    tot = bad = 0
    for k in ("n_fit", "n_sim", "n_buckets", "hits"):
        tot += 1
        if A[k] != B[k]: bad += 1; print("DIFF", name, k, A[k], B[k])
    for bud in ("0.1", "0.25", "0.5"):
        for p, v in A[f"budget_{bud}"].items():
            tot += 1
            if p not in B[f"budget_{bud}"] or v["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", name, bud, p)
    return tot, bad


if mode == "pre":
    tot = bad = 0
    for tag in TAGS:
        for tgt in (2250, 0):
            A, B = json.load(open(f"runs/L11a/sim_{tag}_n{tgt}_sp0_sub0.json")), json.load(open(f"runs/L12c/gate/g_{tag}_n{tgt}.json"))
            assert not any(p.startswith("recal_free_") for p in B["budget_0.25"]), "gate files must be produced with the nofree flag"
            t, b = diffs(A, B, f"{tag} n{tgt}"); tot += t; bad += b
    out = {"policy_budget_cells_compared": tot, "differences": bad, "passed": bad == 0}
    json.dump(out, open("runs/L12c/gate/gate_check.json", "w")); print(out); sys.exit(0 if bad == 0 else 3)

tot = bad = 0; n_files = 0; san = []; sanity_ok = True
for tag in TAGS:
    for tgt in (1125, 2250, 4500, 0):
        d = [(sp, sub) for sp in range(4) for sub in (range(4) if tgt else [0])]
        Bs = []
        for sp, sub in d:
            f = f"sim_{tag}_n{tgt}_sp{sp}_sub{sub}.json"; A, B = json.load(open(f"runs/L11a/{f}")), json.load(open(f"runs/L12c/{f}")); n_files += 1
            t, b = diffs(A, B, f); tot += t; bad += b; Bs.append(B)
        for bud in ("0.1", "0.25", "0.5"):
            dd = np.array([np.mean(B[f"budget_{bud}"]["recal_free_400"]["per_seed"]) - B[f"budget_{bud}"]["static"]["mean"] for B in Bs]); C = len(dd)
            se = dd.std(ddof=1) / np.sqrt(C); ok = bool(dd.mean() <= 2 * se)
            sanity_ok &= ok; san.append({"tag": tag, "scale": tgt or "full", "budget": bud, "mean_free400_minus_static": float(dd.mean()), "se": float(se), "clusters": C, "ok": ok})
out = {"files_compared": n_files, "policy_budget_cells_compared": tot, "differences": bad, "arms_unchanged": bad == 0, "sanity_free400_le_static_plus_2se": sanity_ok, "sanity_cells": san, "passed": bool(bad == 0 and sanity_ok)}
json.dump(out, open("runs/L12c/gate/gate_check_post.json", "w"), indent=1); print({k: v for k, v in out.items() if k != "sanity_cells"})
for s in san:
    if not s["ok"]: print("SANITY FAIL", s)
sys.exit(0 if out["passed"] else 3)

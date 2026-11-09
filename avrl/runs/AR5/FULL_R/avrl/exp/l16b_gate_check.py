"""L16b gates (run from avrl/). Usage: l16b_gate_check.py pre | ident | post
pre  : gate 1 (regression), early. The gate jobs (l16b_sim_alloc.py, STALE=flip25, refits on, n_sim 2250, split 0, subsample 0, both rungs) must reproduce runs/L15a/sim_<tag>_flip25_n2250_sp0_sub0.json with
       0 per-seed differences in uniform, static, greedy_online, recal_free_200, warm_online_200 and warm_online_200_w25 at all three budgets (n_fit, n_sim, n_buckets, hits too). exit 3 = VOID.
ident: gate 2 (identity) and an early gate 3. The identity jobs (norefit flag, n_sim 2250, split 0, subsample 0, both rungs, all six STALE variants) must give warm_online_200 == recal_free_200 per policy seed at
       all three budgets under every variant, and the w25 and w0 twins == warm_online_200 (without refits the weights are never used). Also checks the corruption is applied: recal_free_200 must differ from the
       clean L13a recal_free_200 in at least one budget cell of each rung and variant. Gate 3 (corruption is active) on these files: see gate3(). exit 3 = VOID.
post : after the grid, the 192 files must exist with the seven arms at all three budgets and refits on; flip25 files must equal runs/L15a on the six L15a arms per seed (gate 1 on all 32 files); in every variant
       uniform, static and greedy_online (which do not read the free labels) must equal L13a per seed and n_sim / n_fit / hits / n_buckets must match L13a; gate 3 pooled over the whole grid. exit 3 = VOID.
gate3: from out[stale_stats] (the realised flips of every policy seed's 200 free labels, recorded inside free_draw). For every variant: no changed label on a non-eligible row. jl variants: lo variants change
       nothing at or above the fit-half median, hi variants nothing at or below it; jlrev100 changes every eligible label; jlflip50 pooled realised fraction of changed free labels per rung within 0.20 to 0.30.
       The realised fraction per variant is reported (pooled changed / n and changed / eligible)."""
import json, sys, os
mode = sys.argv[1]
TAGS = ("3B_q_corrupted", "7B_q_corrupted"); STALES = ("jlflip50", "jlflip50_hi", "jlrev100", "fn80", "fp35", "flip25"); BUDS = ("0.1", "0.25", "0.5")
ARMS_L15A = ("uniform", "static", "greedy_online", "recal_free_200", "warm_online_200", "warm_online_200_w25"); ARMS7 = ARMS_L15A + ("warm_online_200_w0",)
ARMS_NOFREE = ("uniform", "static", "greedy_online")
jl = lambda p: json.load(open(p))
os.makedirs("runs/L16b/gate", exist_ok=True)


def diffs(A, B, name, arms):
    """A = reference file (L13a or L15a), B = L16b file; every arm in `arms` must have identical per-seed hits at all budgets; population metadata must match."""
    tot = bad = 0
    for k in ("n_fit", "n_sim", "n_buckets", "hits"):
        tot += 1
        if A[k] != B[k]: bad += 1; print("DIFF", name, k, A[k], B[k])
    for bud in BUDS:
        for p in arms:
            tot += 1
            if p not in B[f"budget_{bud}"] or A[f"budget_{bud}"][p]["per_seed"] != B[f"budget_{bud}"][p]["per_seed"]: bad += 1; print("DIFF", name, bud, p)
    return tot, bad


def gate3(files):
    """files: {(tag, stale): [json, ...]}. Returns (report, n_failures)."""
    rep = {}; bad = 0
    for (tag, st), ds in files.items():
        T = dict(n=0, changed=0, elig=0, ch_not_elig=0, below=0, above=0, ch_below=0, ch_above=0, ch_at=0, pos=0, seeds_ok=True); fr = []
        for d in ds:
            ss = d["stale_stats"]
            if sorted(map(int, ss)) != list(range(d["n_policy_seeds"])): T["seeds_ok"] = False
            n = sum(v["n"] for v in ss.values()); c = sum(v["n_changed"] for v in ss.values()); fr.append(c / n)
            for key, f in (("n", "n"), ("changed", "n_changed"), ("elig", "n_eligible"), ("ch_not_elig", "n_changed_not_eligible"), ("below", "n_below_median"), ("above", "n_above_median"),
                           ("ch_below", "n_changed_below_median"), ("ch_above", "n_changed_above_median"), ("ch_at", "n_changed_at_median"), ("pos", "n_pos_before")):
                T[key] += sum(v[f] for v in ss.values())
        r = {"files": len(ds), "n_labels": T["n"], "n_changed": T["changed"], "frac_changed": T["changed"] / T["n"], "frac_changed_min_file": min(fr), "frac_changed_max_file": max(fr),
             "frac_changed_of_eligible": T["changed"] / max(T["elig"], 1), "frac_eligible": T["elig"] / T["n"], "frac_pos_before": T["pos"] / T["n"], "changed_not_eligible": T["ch_not_elig"],
             "changed_below_median": T["ch_below"], "changed_above_median": T["ch_above"], "changed_at_median": T["ch_at"], "jl_median_mean": sum(d["jl_median"] for d in ds) / len(ds)}
        errs = []
        if not T["seeds_ok"]: errs.append("stale_stats does not cover seeds 0..7 in every file")
        if T["ch_not_elig"] != 0: errs.append(f"{T['ch_not_elig']} changed labels on non-eligible rows")
        if st in ("jlflip50", "jlrev100") and (T["ch_above"] or T["ch_at"]): errs.append(f"flips on the keep side (above {T['ch_above']}, at {T['ch_at']})")
        if st == "jlflip50_hi" and (T["ch_below"] or T["ch_at"]): errs.append(f"flips on the keep side (below {T['ch_below']}, at {T['ch_at']})")
        if st == "jlrev100" and T["changed"] != T["elig"]: errs.append("jlrev100 did not change every eligible label")
        if st == "jlflip50" and not (0.20 <= r["frac_changed"] <= 0.30): errs.append(f"jlflip50 realised fraction {r['frac_changed']:.4f} outside 0.20 to 0.30")
        if st == "flip25" and not (0.20 <= r["frac_changed"] <= 0.30): errs.append(f"flip25 realised fraction {r['frac_changed']:.4f} outside 0.20 to 0.30")
        r["errors"] = errs; bad += len(errs); rep[f"{tag}|{st}"] = r
        print(f"gate3 {tag} {st}: changed {r['frac_changed']:.4f} of labels ({r['frac_changed_of_eligible']:.4f} of eligible, per-file {r['frac_changed_min_file']:.3f} to {r['frac_changed_max_file']:.3f}), changed below/above/at median {T['ch_below']}/{T['ch_above']}/{T['ch_at']}, non-eligible changed {T['ch_not_elig']}", "OK" if not errs else errs)
    return rep, bad


if mode == "pre":
    tot = bad = 0
    for tag in TAGS:
        A, B = jl(f"runs/L15a/sim_{tag}_flip25_n2250_sp0_sub0.json"), jl(f"runs/L16b/gate/g_{tag}_flip25.json")
        assert B["stale"] == "flip25" and B["warm_refit"] is True, "gate-1 files must be STALE=flip25 with refits on"
        t, b = diffs(A, B, tag, ARMS_L15A); tot += t; bad += b
    out = {"cells_compared": tot, "differences": bad, "passed": bad == 0}
    json.dump(out, open("runs/L16b/gate/gate_check.json", "w")); print(out); sys.exit(0 if bad == 0 else 3)

if mode == "ident":
    tot = bad = 0; one = {}; active = {}; files = {}
    for tag in TAGS:
        A = jl(f"runs/L13a/sim_{tag}_n2250_sp0_sub0.json")
        for st in STALES:
            B = jl(f"runs/L16b/gate/i_{tag}_{st}.json"); assert B["warm_refit"] is False and B["stale"] == st, "identity files must be norefit and carry their STALE variant"
            files[(tag, st)] = [B]; one[f"{tag}|{st}"] = B["warm_oneclass_draws"]
            for bud in BUDS:
                tot += 3; w, f, w25, w0 = (B[f"budget_{bud}"][k]["per_seed"] for k in ("warm_online_200", "recal_free_200", "warm_online_200_w25", "warm_online_200_w0"))
                if w != f: bad += 1; print("IDENT DIFF warm vs free", tag, st, bud, w, f)
                if w25 != w: bad += 1; print("IDENT DIFF w25 vs warm", tag, st, bud, w25, w)
                if w0 != w: bad += 1; print("IDENT DIFF w0 vs warm", tag, st, bud, w0, w)
            n_diff = sum(A[f"budget_{bud}"]["recal_free_200"]["per_seed"] != B[f"budget_{bud}"]["recal_free_200"]["per_seed"] for bud in BUDS)
            active[f"{tag}|{st}"] = n_diff; tot += 1
            if n_diff == 0: bad += 1; print("STALE NOT APPLIED", tag, st)
    g3, b3 = gate3(files)
    out = {"cells_compared": tot, "differences": bad, "oneclass_free_draws": one, "stale_applied_budget_cells_changed_of_3": active, "gate3": g3, "gate3_failures": b3, "passed": bool(bad == 0 and b3 == 0)}
    json.dump(out, open("runs/L16b/gate/gate_check_ident.json", "w"), indent=1); print({k: v for k, v in out.items() if k != "gate3"}); sys.exit(0 if out["passed"] else 3)

tot = bad = 0; n_files = 0; missing = 0; files = {}
for tag in TAGS:
    for st in STALES:
        files[(tag, st)] = []
        for sp in range(4):
            for sub in range(4):
                f = f"sim_{tag}_n2250_sp{sp}_sub{sub}.json"; A = jl(f"runs/L13a/{f}"); g = f"runs/L16b/sim_{tag}_{st}_n2250_sp{sp}_sub{sub}.json"
                if not os.path.exists(g): missing += 1; print("MISSING FILE", g); continue
                B = jl(g); n_files += 1; files[(tag, st)].append(B)
                if B["stale"] != st or B["warm_refit"] is not True: missing += 1; print("BAD HEADER", g)
                for bud in BUDS:
                    for p in ARMS7:
                        if p not in B[f"budget_{bud}"]: missing += 1; print("MISSING ARM", g, bud, p)
                t, b = diffs(A, B, f"{tag}|{st}|sp{sp}|sub{sub}|vsL13a", ARMS_NOFREE); tot += t; bad += b
                if st == "flip25":
                    t, b = diffs(jl(f"runs/L15a/sim_{tag}_flip25_n2250_sp{sp}_sub{sub}.json"), B, f"{tag}|flip25|sp{sp}|sub{sub}|vsL15a", ARMS_L15A); tot += t; bad += b
g3, b3 = gate3({k: v for k, v in files.items() if v})
out = {"files_found": n_files, "files_expected": 192, "cells_compared": tot, "differences": bad, "missing": missing, "gate3": g3, "gate3_failures": b3,
       "passed": bool(bad == 0 and missing == 0 and n_files == 192 and b3 == 0)}
json.dump(out, open("runs/L16b/gate/gate_check_post.json", "w"), indent=1); print({k: v for k, v in out.items() if k != "gate3"})
sys.exit(0 if out["passed"] else 3)

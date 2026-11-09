"""L7a-tiebreak critic check (CPU). Independent recompute of the cross-fit table and main-mode ratios straight from the raw boot JSONs, no use of l7a_finalize.py."""
import json, numpy as np, os
R = "/homes/rdubash/avrl_live/FULL/avrl/runs/L7a-tiebreak"
for pool in ["rm", "judge3b", "L3b-judge-proxy", "rm_sens"]:
    for frac in ["0.25", "0.1"]:
        x = json.load(open(f"{R}/{pool}/boot_xfit_f{frac}.json")); H = x["halves"]
        full = lambda a: sum(np.mean(h["full_half"][a]) for h in H)
        per = lambda a: sum(np.array([np.mean(r[a]) for r in h["resamples"]]) for h in H)
        print(f"== {pool} f{frac} n_boot {x['n_boot']} seeds {x['seeds_per_resample']} halves n={[h['full_half']['n'] for h in H]} budgets={[h['full_half']['budget'] for h in H]} nerr={[h['full_half']['n_err'] for h in H]} nres={[len(h['resamples']) for h in H]}")
        print(" choices", [h["tb_xfit_is"] for h in H], "sel", [(h['selection']['tier_agreement'], h['selection']['n_tier'], h['selection']['n_err_tier'], h['selection']['auc_soth'], h['selection']['auc_sdiff'], h['selection']['fallback']) for h in H])
        print(" best fixed per half", [h["full_half"]["best_fixed"] for h in H])
        for a in ["agree_acc", "agree_acc_soth", "agree_acc_sdiff", "tb_xfit", "greedy_warm", "greedy_blend", "oracle", "uniform"]: print(f"  {a}: {full(a):.3f}", end=";")
        print()
        for num, den in [("tb_xfit", "greedy_blend"), ("greedy_blend", "tb_xfit"), ("tb_xfit", "agree_acc"), ("greedy_blend", "agree_acc"), ("agree_acc", "greedy_blend")]:
            r = per(num) / per(den); q = np.percentile(r, [2.5, 97.5]); pl = full(num) / full(den)
            print(f"  {num}/{den}: plug {pl:.4f} pct [{q[0]:.3f},{q[1]:.3f}] basic [{2*pl-q[1]:.3f},{2*pl-q[0]:.3f}]")
        # budget equality: every arm in every resample uses same budget (single value)
        print(" resample budgets", {r["budget"] for h in H for r in h["resamples"]}, "n_err range", min(r["n_err"] for h in H for r in h["resamples"]), max(r["n_err"] for h in H for r in h["resamples"]))
    for frac in ["0.25"]:
        m = json.load(open(f"{R}/{pool}/boot_tb_f{frac}.json")); fp = m["full_pool"]; rs = m["resamples"]
        print(f"-- main {pool} f{frac} n_pool {m['n_pool']} nerr {fp['n_err']} best5 {fp['best_fixed']} best7 {fp['best_fixed7']} nres {len(rs)} seeds {m['seeds_per_resample']} budgets {{r['budget'] for r in rs}}={ {r['budget'] for r in rs} }")
        print(" fixed_mean_at_25_7", {k: round(v, 2) for k, v in fp["fixed_mean_at_25_7"].items()})
        mean = lambda a: np.array([np.mean(r[a]) for r in rs]); fm = lambda a: np.mean(fp[a])
        for a in ["agree_acc", "agree_acc_soth", "agree_acc_sdiff", "greedy_warm", "greedy_blend", "greedy_blend7", "oracle"]: print(f"  {a}: {fm(a):.3f}", end=";")
        print()
        b7 = fp["best_fixed7"]
        for num, den in [("greedy_blend7", b7), ("greedy_warm7", b7), ("greedy_blend", b7), ("greedy_blend", "agree_acc")]:
            r = mean(num) / mean(den); q = np.percentile(r, [2.5, 97.5]); print(f"  {num}/{den}: plug {fm(num)/fm(den):.4f} pct [{q[0]:.3f},{q[1]:.3f}]")
        print(" sanity", m["sanity"])

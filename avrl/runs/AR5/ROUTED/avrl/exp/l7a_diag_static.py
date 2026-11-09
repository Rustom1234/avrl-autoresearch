"""L7a diagnostic, not a registered gate (CPU only): is the learned arm's loss to the frozen table caused by its scorer or by its refits?

static_p0 is the greedy_prior initial scorer on its own: the StandardScaler plus LogisticRegression fitted on the same 200 p0
labels as each subset's frozen table, never refit, no eval labels read, top-budget docs taken by score with the per-seed random
tie-break. It is compared with the frozen table on the same subsets, splits and tie seeds (recomputed here with the L7a
run_pool). greedy_prior minus static_p0 is then the effect of the eval-label refits, taken from runs/L7a/<POOL>/result.json.

Usage: python -I l7a_diag_static.py --out OUTDIR      (writes OUTDIR/result.json; prints the summary)
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import l7a_learned_equal_labels as L7

FIT = f"{L7.RUNS}/p0/feats.parquet"
POOLS = ["L2b", "L4b"]


def static_hits(feats, sub, split, frac, seeds):
    """Hits of the p0-fitted scorer alone on one split, one value per tie-break seed."""
    sim, _ = L7.L1B.build_sim(pd.read_parquet(feats), L7.GROUP, split)
    y = sim[L7.LABEL].to_numpy(int); X = sim[L7.CHEAP].fillna(0).to_numpy(float)
    s0 = L7.scorer().fit(sub[L7.CHEAP].fillna(0).to_numpy(float), sub[L7.LABEL].to_numpy(int)).predict_proba(X)[:, 1]
    b = int(frac * len(y)); hits = []
    for s in seeds:
        rnd = np.random.default_rng(s).random(len(y))
        hits.append(int(y[np.lexsort((rnd, -s0))[:b]].sum()))
    return b, hits


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fit_df = pd.read_parquet(FIT); splits, seeds, frac, k0 = list(range(8)), list(range(8)), 0.10, 50.0
    out = {"experiment": "L7a diagnostic", "policy": "static_p0 (p0 200-label scorer, no refit, no eval labels)",
           "subsets": 20, "splits": splits, "tie_seeds": seeds, "budget_frac": frac, "pools": {}}
    for pool in POOLS:
        feats = f"{L7.RUNS}/{pool}/feats.parquet"
        greedy_gap = json.load(open(f"{L7.RUNS}/L7a/{pool}/result.json"))["pools"][pool]["gap_pp_per_subset"]
        st_fr, gr_st = [], []
        for i in range(20):
            idx = L7.draw_subset(fit_df, i, 200); sub = fit_df.iloc[idx]
            fr = L7.run_pool(feats, L7.fit_table(sub, k0), frac, splits, seeds)
            per_split = []
            for s in splits:
                b, hits = static_hits(feats, sub, s, frac, seeds)
                assert b == fr[s]["budget"]
                per_split.append(float(np.mean([100.0 * (h - z) / b for h, z in zip(hits, fr[s]["frozen_hits"])])))
            st_fr.append(float(np.mean(per_split)))
            gr_st.append(float(np.mean(np.asarray(greedy_gap[i]) - np.asarray(per_split))))
        m1, m2 = np.asarray(st_fr), np.asarray(gr_st)
        out["pools"][pool] = {"static_minus_frozen_subset_pp": st_fr, "static_minus_frozen_mean_pp": float(m1.mean()),
                              "static_minus_frozen_sd_pp": float(m1.std(ddof=1)), "static_minus_frozen_above_zero": int((m1 > 0).sum()),
                              "greedy_minus_static_subset_pp": gr_st, "greedy_minus_static_mean_pp": float(m2.mean()),
                              "greedy_minus_static_sd_pp": float(m2.std(ddof=1)), "greedy_minus_static_below_zero": int((m2 < 0).sum())}
        p = out["pools"][pool]
        print(f"{pool}: static_p0 minus frozen {p['static_minus_frozen_mean_pp']:+.2f} (sd {p['static_minus_frozen_sd_pp']:.2f}, "
              f"{p['static_minus_frozen_above_zero']} of 20 above 0); greedy_prior minus static_p0 {p['greedy_minus_static_mean_pp']:+.2f} "
              f"(sd {p['greedy_minus_static_sd_pp']:.2f}, {p['greedy_minus_static_below_zero']} of 20 below 0)", flush=True)
    with open(os.path.join(a.out, "result.json"), "w") as fh: json.dump(out, fh, indent=1)
    print(f"wrote {os.path.join(a.out, 'result.json')}", flush=True)


if __name__ == "__main__":
    main()

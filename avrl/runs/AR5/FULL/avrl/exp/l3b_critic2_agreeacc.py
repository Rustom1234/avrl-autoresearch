"""Second critic: independent agree_acc (accept-filtered agreement threshold) recompute, own code. Writes runs/L3b-judge-proxy/critic2_agreeacc.json."""
import json, numpy as np, pandas as pd
R = "runs/L3b-judge-proxy"; df = pd.read_parquet(f"{R}/feats_J.parquet"); cnt = df.cell100.value_counts(); df = df[df.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
assert len(df) == 2000
def hits_agree_acc(d, budget, seed):
    rng = np.random.default_rng(seed); n = len(d); u1 = rng.random(n); acc = d.accept.values.astype(bool)
    # accepted first (agreement asc, n_distinct desc, tie u1), rejected after in same order
    o = np.lexsort((u1, -d.n_distinct.values, d.agreement.values, ~acc))
    return int(d.judge_error.values[o[:budget]].sum())
out = {}
for frac in (0.1, 0.25):
    b = int(frac * len(df)); out[f"full_{frac}"] = float(np.mean([hits_agree_acc(df, b, s) for s in range(8)]))
for tag, frac in (("f0.25", 0.25), ("f0.1", 0.1)):
    j = json.load(open(f"{R}/boot_J_{tag}.json")); brng = np.random.default_rng(12345); ratios = []; ratio_b = []
    aa_all = []
    for r in j["resamples"]:
        d = df.iloc[brng.integers(0, len(df), len(df))].reset_index(drop=True); b = int(frac * len(d)); assert b == r["budget"]
        aa = np.array([hits_agree_acc(d, b, s) for s in range(4)]); assert (np.array([0]) == 0).all()
        assert int(d.judge_error.sum()) == r["n_err"]
        aa_all.append(aa.mean()); ratios.append(np.mean(r["greedy_warm"]) / aa.mean()); ratio_b.append(np.mean(r["blend50"]) / aa.mean())
    out[tag] = dict(agree_acc_boot_mean=float(np.mean(aa_all)), warm_over_agree_acc_pct=[float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))], frac_warm_gt1=float(np.mean(np.array(ratios) > 1)),
                    blend50_over_agree_acc_pct=[float(np.percentile(ratio_b, 2.5)), float(np.percentile(ratio_b, 97.5))])
json.dump(out, open(f"{R}/critic2_agreeacc.json", "w"), indent=1); print(json.dumps(out, indent=1))

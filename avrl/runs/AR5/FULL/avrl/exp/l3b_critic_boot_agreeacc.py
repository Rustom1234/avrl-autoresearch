"""L3b critic check 3 (CPU): paired bootstrap of greedy_warm and greedy_blend against the extra fixed rule agree_acc (agreement then n_distinct among judge-accepted groups, rejected last).
Reproduces the L3b resamples exactly (brng = default_rng(12345), integers(0, 2000, 2000) per resample) and checks blend50 and agree_thr against the stored JSON before using them. NOT pre-registered; informs the reading of claim 2 only."""
import numpy as np, pandas as pd, json
L = "/homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge-proxy"
df = pd.read_parquet(f"{L}/feats_J.parquet")
def rs(o): sc = np.empty(len(o)); sc[o] = (len(o) - np.arange(len(o))) / len(o); return sc
def hits(d, k, budget, seed):
    n = len(d); y = d.judge_error.values.astype(int); agree, nd, sm, acc = d.agreement.values, d.n_distinct.values, d.s_maj.values, d.accept.values.astype(bool)
    rng = np.random.default_rng(seed); u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    o_agree = np.lexsort((u1, -nd, agree)); o_judge = np.lexsort((u2, np.where(acc, sm, np.inf)))
    if k == "agree_thr": o = o_agree
    elif k == "blend50": o = np.lexsort((u3, -(0.5 * rs(o_agree) + 0.5 * rs(o_judge))))
    elif k == "agree_acc": o = np.lexsort((u1, -nd, agree, ~acc))
    return int(y[o[:budget]].sum())
out = {}
for fn, frac in (("boot_J_f0.25.json", 0.25), ("boot_J_f0.1.json", 0.1)):
    b = json.load(open(f"{L}/{fn}")); brng = np.random.default_rng(12345); ok_b = ok_a = True; rows = []
    for i, r in enumerate(b["resamples"]):
        d = df.iloc[brng.integers(0, len(df), len(df))].reset_index(drop=True); bud = int(frac * len(d))
        hb = [hits(d, "blend50", bud, s) for s in range(4)]; ha = [hits(d, "agree_thr", bud, s) for s in range(4)]; aa = [hits(d, "agree_acc", bud, s) for s in range(4)]
        ok_b &= hb == r["blend50"]; ok_a &= ha == r["agree_thr"]; rows.append((np.mean(r["greedy_warm"]), np.mean(r["greedy_blend"]), np.mean(aa), np.mean(r["blend50"])))
    A = np.array(rows); fp = b["full_pool"]; aa_fp = float(np.mean([hits(df, "agree_acc", fp["budget"], s) for s in range(8)]))
    pw, pb = np.mean(fp["greedy_warm"]), np.mean(fp["greedy_blend"]); rw, rb = A[:, 0] / A[:, 2], A[:, 1] / A[:, 2]
    per = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]; bas = lambda v, p: [float(2 * p - np.percentile(v, 97.5)), float(2 * p - np.percentile(v, 2.5))]
    out[fn] = {"reproduces_blend50_all_resamples": bool(ok_b), "reproduces_agree_thr_all_resamples": bool(ok_a), "agree_acc_fullpool_mean": aa_fp, "agree_acc_boot_mean": float(A[:, 2].mean()), "blend50_boot_mean": float(A[:, 3].mean()),
               "greedy_warm_fullpool": float(pw), "greedy_blend_fullpool": float(pb),
               "warm_over_agree_acc": {"plug": float(pw / aa_fp), "boot_mean": float(rw.mean()), "pct": per(rw), "basic": bas(rw, pw / aa_fp), "frac_gt1": float((rw > 1).mean())},
               "blend_over_agree_acc": {"plug": float(pb / aa_fp), "boot_mean": float(rb.mean()), "pct": per(rb), "basic": bas(rb, pb / aa_fp), "frac_gt1": float((rb > 1).mean())}}
json.dump(out, open(f"{L}/critic_agreeacc_boot.json", "w"), indent=1); print(json.dumps(out, indent=1))

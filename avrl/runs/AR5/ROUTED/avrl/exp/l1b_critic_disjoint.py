"""L1b critic: does the tuned-rule gain survive when tune and evaluation documents are disjoint? CPU only.
The 99 cell100 groups (>= 3 docs) are split 49/50 at random (8 partitions). Direction X -> Y: the sweep of
l1b_tuned_threshold.py is tuned on 4 sub-splits of pool X (each drops a random 25% of X's groups, as in L1b), the
winner per budget is scored against plain fixed on 4 sub-splits of pool Y. No document is in both pools. Reuses the
L1b Sim class and SPECS so the rule definitions are identical to the ones under review."""
import sys, json, importlib.util
from multiprocessing import Pool
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("l1b", "/homes/rdubash/avrl_live/ROUTED/avrl/exp/l1b_tuned_threshold.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
FEATS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/feats.parquet"
FR = [0.05, 0.10, 0.25, 0.50]; SEEDS = list(range(8))

def subsplits(df, groups, base):
    out = []
    for j in range(4):
        g = np.array(groups); np.random.default_rng(1000 * base + j).shuffle(g)
        keep = g[len(g) // 4:]
        out.append(L.Sim(df[df.cell100.isin(set(keep))].reset_index(drop=True), "proxy_error"))
    return out

def rate(S, spec_, f, s): b = int(f * S.n); return 100.0 * L.hits_for(S, spec_, b, s) / b

def job(arg):
    p, direction = arg
    df = pd.read_parquet(FEATS); cnt = df.cell100.value_counts(); df = df[df.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = df.cell100.unique(); np.random.default_rng(500 + p).shuffle(g)
    A, B = list(g[:49]), list(g[49:])
    X, Y = (A, B) if direction == 0 else (B, A)
    tune = subsplits(df, X, 10 * p + direction); held = subsplits(df, Y, 10 * p + direction + 5)
    res = {}
    for f in FR:
        sc = {n: np.mean([rate(S, sp, f, s) for S in tune for s in SEEDS]) for n, sp in L.SPECS if n in L.SWEEP}
        sel = max(L.SWEEP, key=lambda n: sc[n]); sp = dict(L.SPECS)[sel]
        gap = np.mean([rate(S, sp, f, s) - rate(S, None, f, s) for S in held for s in SEEDS])
        res[f"{f:.2f}"] = {"selected": sel, "gap_pp": float(gap), "n_held_docs": [S.n for S in held]}
    return {"partition": p, "direction": direction, "res": res}

if __name__ == "__main__":
    jobs = [(p, d) for p in range(8) for d in (0, 1)]
    with Pool(16) as pool: out = pool.map(job, jobs)
    json.dump(out, open("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L1b_critic/disjoint.json", "w"), indent=1)
    for k in ["0.05", "0.10", "0.25", "0.50"]:
        gaps = np.array([o["res"][k]["gap_pp"] for o in out])
        print(k, "mean %+.2f sd %.2f min %+.2f max %+.2f  n>2.0: %d/16  n>0: %d/16" % (gaps.mean(), gaps.std(ddof=1), gaps.min(), gaps.max(), (gaps > 2).sum(), (gaps > 0).sum()))
        from collections import Counter
        print("   selected:", Counter(o["res"][k]["selected"] for o in out).most_common(4))
        print("   per direction:", [round(x, 2) for x in gaps])

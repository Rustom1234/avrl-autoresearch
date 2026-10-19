"""L3b critic check 2 (CPU): does skipping judge-rejected groups alone explain blend50's edge over agree_thr? Extra fixed rules, full pool, 8 seeds, same tie-break style as sim_warm_boot_j.py."""
import numpy as np, pandas as pd, json
d = pd.read_parquet("/homes/rdubash/avrl_live/FULL/avrl/runs/L3b-judge-proxy/feats_J.parquet")
y = d.judge_error.values.astype(int); n = len(d); agree, nd, sm, acc = d.agreement.values, d.n_distinct.values, d.s_maj.values, d.accept.values.astype(bool)
def rs(o): sc = np.empty(len(o)); sc[o] = (len(o) - np.arange(len(o))) / len(o); return sc
def orders(rng):
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    o_agree = np.lexsort((u1, -nd, agree)); o_judge = np.lexsort((u2, np.where(acc, sm, np.inf)))
    o_blend = np.lexsort((u3, -(0.5 * rs(o_agree) + 0.5 * rs(o_judge))))
    # extra: agreement among accepted, rejected last
    o_agree_acc = np.lexsort((u1, -nd, agree, ~acc))
    # extra: blend where the agreement side also sinks rejected
    o_blend_acc = np.lexsort((u3, -(0.5 * rs(o_agree_acc) + 0.5 * rs(o_judge))))
    # extra: 3/4 agreement 1/4 judge, 1/4 agreement 3/4 judge (sensitivity of the 50/50 choice)
    o_b25 = np.lexsort((u3, -(0.25 * rs(o_agree) + 0.75 * rs(o_judge)))); o_b75 = np.lexsort((u3, -(0.75 * rs(o_agree) + 0.25 * rs(o_judge))))
    return {"agree_thr": o_agree, "judge_thr": o_judge, "blend50": o_blend, "agree_acc": o_agree_acc, "blend50_acc": o_blend_acc, "blend25agree": o_b25, "blend75agree": o_b75}
out = {}
for fr in (0.1, 0.25):
    b = int(fr * n); res = {}
    for k in ("agree_thr", "judge_thr", "blend50", "agree_acc", "blend50_acc", "blend25agree", "blend75agree"):
        res[k] = float(np.mean([y[orders(np.random.default_rng(s))[k][:b]].sum() for s in range(8)]))
    res["oracle"] = int(y.sum()); out[str(fr)] = res
print(json.dumps(out, indent=1))

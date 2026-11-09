"""L7a critic check 2 (CPU): 10% secondary numbers from boot_tb_f0.1.json, ratio SDs, and the boundary tier counts straight from feats_J.parquet."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "/homes/rdubash/avrl_live/FULL/avrl/runs/L7a-tiebreak"; A = "/homes/rdubash/avrl_live/FULL/avrl"
res = json.load(open(f"{R}/result.json"))
for p in ["rm", "judge3b", "L3b-judge-proxy"]:
    x = res["pools"][p]["xfit"]["0.25"]["ratios"]["tb_xfit_over_greedy_blend"]; print(p, "25% tb/gb boot sd", round(x["boot_sd"], 4), "boot mean", round(x["boot_mean"], 4), "plug", round(x["plug_in"], 4))
for p in ["rm", "judge3b", "L3b-judge-proxy"]:
    m = json.load(open(f"{R}/{p}/boot_tb_f0.1.json")); fp = m["full_pool"]; rs = m["resamples"]; mean = lambda a: np.array([np.mean(r[a]) for r in rs]); fm = lambda a: np.mean(fp[a])
    print(f"-- {p} 10%: best5 {fp['best_fixed']} best7 {fp['best_fixed7']} fixed_mean_at_frac7 {({k: round(v,1) for k,v in fp['fixed_mean_at_frac7'].items()})}")
    bf = max(fp["fixed_mean_at_frac7"], key=fp["fixed_mean_at_frac7"].get)
    for num, den in [("agree_acc_soth", "agree_acc"), ("agree_acc_sdiff", "agree_acc"), ("greedy_blend", "agree_acc"), ("greedy_blend", fp["best_fixed7"]), ("greedy_blend7", fp["best_fixed7"]), ("greedy_blend", bf), ("greedy_blend7", bf)]:
        r = mean(num) / mean(den); q = np.percentile(r, [2.5, 97.5]); print(f"   {num}/{den}: plug {fm(num)/fm(den):.3f} pct [{q[0]:.3f},{q[1]:.3f}]")
    print("   best fixed at 10% by 8-seed mean:", bf)
# boundary tier from the parquet
for p, d, acc in [("rm", "runs/L5c-proxies/rm", "accept"), ("judge3b", "runs/L5c-proxies/judge3b", "accept"), ("q4", "runs/L3b-judge-proxy", "accept")]:
    df = pd.read_parquet(f"{A}/{d}/feats_J.parquet"); a = df[acc].astype(bool).values; ag = df.agreement.values; nd = df.n_distinct.values; y = df.judge_error.values
    idx = np.flatnonzero(a); o = idx[np.lexsort((-nd[idx], ag[idx]))]; last = o[499]; m = a & (ag == ag[last]); slots = int((ag[o[:500]] == ag[last]).sum())
    print(p, "accepted", int(a.sum()), "errors", int(y.sum()), "tier agreement", ag[last], "tier groups", int(m.sum()), "tier errors", int(y[m].sum()), "slots in tier at 25%", slots, "AUC s_oth", round(roc_auc_score(y[m], df.s_oth.values[m]), 3), "AUC -s_diff", round(roc_auc_score(y[m], -df.s_diff.values[m]), 3))

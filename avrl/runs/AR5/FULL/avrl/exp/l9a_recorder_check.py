"""L9a recorder check: recompute S_tb, per-budget ratios, shard-B sign, stratified AUC, SD gate and the verdict bins from raw boot JSONs and feats_J parquet, written separately from l9a_analyze.py. CPU only."""
import json, os, numpy as np, pandas as pd
from scipy.stats import rankdata
R = "/homes/rdubash/avrl_live/FULL/avrl/runs/L9a-shardB-tiebreak"
FR = ["0.1", "0.15", "0.2", "0.25", "0.3", "0.4"]
ARM = {"rm": "agree_acc_soth", "q3_4b": "agree_acc_sdiff"}
res = json.load(open(f"{R}/result.json"))
out = {}
for k in ("rm", "q3_4b"):
    for tag, d in (("pooled", "boot"), ("shardB", "boot_B")):
        lr = []; pl = []; per = {}
        for f in FR:
            j = json.load(open(f"{R}/{d}/{k}/boot_tb_f{f}.json"))
            assert j["n_boot"] == 200 and j["seeds_per_resample"] == 4 and len(j["resamples"]) == 200
            num = np.array([np.mean(r[ARM[k]]) for r in j["resamples"]]); den = np.array([np.mean(r["agree_acc"]) for r in j["resamples"]])
            lr.append(np.log(num / den)); pl.append(np.log(np.mean(j["full_pool"][ARM[k]]) / np.mean(j["full_pool"]["agree_acc"])))
            per[f] = (round(float(np.exp(pl[-1])), 3), [round(float(x), 3) for x in np.percentile(np.exp(lr[-1]), [2.5, 97.5])])
        S = np.mean(lr, 0); plug = float(np.mean(pl)); q = np.percentile(S, [2.5, 97.5])
        out[(k, tag)] = dict(plug=plug, bm=float(S.mean()), sd=float(S.std(ddof=1)), pct=[float(q[0]), float(q[1])], basic=[2 * plug - float(q[1]), 2 * plug - float(q[0])], npos=int(sum(x > 0 for x in pl)), per=per, pl=pl)
    # stratified AUC on the pooled pool
    F = pd.read_parquet(f"{R}/pool/{k}/feats_J.parquet"); A = F[(F.accept == 1) & (F.agreement < 1.0)]
    y = A.judge_error.values; s = (A.s_oth.values if k == "rm" else -A.s_diff.values); st = A.agreement.values
    def sauc(y, s, st):
        U = T = 0.0
        for g in np.unique(st):
            i = st == g; n1 = int(y[i].sum()); n0 = int(i.sum()) - n1
            if n1 == 0 or n0 == 0: continue
            r = rankdata(s[i]); U += r[y[i] == 1].sum() - n1 * (n1 + 1) / 2; T += n1 * n0
        return U / T
    a = sauc(y, s, st); rng = np.random.default_rng(0); n = len(y); bs = []
    for _ in range(2000):
        ix = rng.integers(0, n, n); bs.append(sauc(y[ix], s[ix], st[ix]))
    out[(k, "auc")] = (a, [float(x) for x in np.percentile(bs, [2.5, 97.5])], n, int(y.sum()))
    # counts
    out[(k, "counts")] = dict(n=len(F), accepted=int(F.accept.sum()), J=int(F.judge_error.sum()), disjoint_shards=bool(F.pid.is_unique), pids=(int(F.pid.min()), int(F.pid.max())), shards=F.shard.value_counts().to_dict())
    vi = res["verdict_inputs"][k]
    P = out[(k, "pooled")]; B = out[(k, "shardB")]; au = out[(k, "auc")]
    pass_k = P["pct"][0] > 0 and au[0] >= 0.62 and au[1][0] > 0.50 and B["plug"] > 0 and P["sd"] <= 0.015
    kill_k = P["plug"] <= 0.005
    print(f"[{k}] pooled S_tb plug {P['plug']:+.4f} bm {P['bm']:+.4f} sd {P['sd']:.4f} pct [{P['pct'][0]:+.4f},{P['pct'][1]:+.4f}] basic [{P['basic'][0]:+.4f},{P['basic'][1]:+.4f}] npos {P['npos']}/6")
    print(f"   per budget ratio (plug, pct CI): {P['per']}")
    print(f"   shardB S_tb plug {B['plug']:+.4f} bm {B['bm']:+.4f} sd {B['sd']:.4f} pct [{B['pct'][0]:+.4f},{B['pct'][1]:+.4f}] basic [{B['basic'][0]:+.4f},{B['basic'][1]:+.4f}] npos {B['npos']}/6; per budget {B['per']}")
    print(f"   strat AUC {au[0]:.4f} CI [{au[1][0]:.4f},{au[1][1]:.4f}] n {au[2]} err {au[3]}; counts {out[(k,'counts')]}")
    print(f"   match result.json: plug {abs(P['plug']-vi['S_tb_plug_in'])<5e-5} lo {abs(P['pct'][0]-vi['S_tb_ci_lo'])<5e-5} sd {abs(P['sd']-vi['S_tb_measured_sd'])<5e-5} auc {abs(au[0]-vi['strat_auc'])<5e-4} aucLo {abs(au[1][0]-vi['strat_auc_ci_lo'])<5e-3} B {abs(B['plug']-vi['S_tb_shardB_plug_in'])<5e-5}")
    print(f"   bins re-derived: PASS {pass_k} KILL {kill_k}; stored pass_k {vi['pass_k']} kill_k {vi['kill_k']}")

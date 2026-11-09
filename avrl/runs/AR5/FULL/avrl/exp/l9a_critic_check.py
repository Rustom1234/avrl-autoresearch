"""L9a critic check (CPU, .venv python). Independent recomputation of the L9a-shardB-tiebreak numbers from raw boot JSONs and feats_J parquets with plain numpy / loops, written separately from l9a_analyze.py."""
import json, os, glob, sys
import numpy as np, pandas as pd
R = "/homes/rdubash/avrl_live/FULL/avrl/runs/L9a-shardB-tiebreak"; FR = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4); fs = lambda f: f"{f:g}"
ARM = {"rm": "agree_acc_soth", "q3_4b": "agree_acc_sdiff"}
out = []
def P(*a):
    s = " ".join(str(x) for x in a); out.append(s); print(s, flush=True)
def curve(path, k):
    lr, pl, rat, ex, bud, ne, hdr = [], [], {}, {}, {}, [], {}
    for f in FR:
        bj = json.load(open(path(f))); a = ARM[k]
        fa = np.mean(bj["full_pool"][a]); fb = np.mean(bj["full_pool"]["agree_acc"]); pl.append(np.log(fa / fb)); rat[fs(f)] = round(fa / fb, 4); ex[fs(f)] = round(fa - fb, 2); bud[fs(f)] = bj["full_pool"]["budget"]
        r = bj["resamples"]; l = np.log(np.array([np.mean(x[a]) for x in r]) / np.array([np.mean(x["agree_acc"]) for x in r])); lr.append(l); ne.append([x["n_err"] for x in r])
        hdr[fs(f)] = (bj["n_pool"], bj["n_boot"], bj["seeds_per_resample"], len(r), sorted(set(len(x[a]) for x in r)), sorted(set(x["budget"] for x in r)), len(bj["full_pool"][a]), bj["full_pool"]["budget"], bj["full_pool"]["n_err"])
    assert all(n == ne[0] for n in ne)
    S = np.mean(lr, 0); pg = float(np.mean(pl)); q = np.percentile(S, [2.5, 97.5])
    return dict(plug=pg, bm=float(S.mean()), sd=float(S.std(ddof=1)), pct=[float(q[0]), float(q[1])], basic=[2 * pg - float(q[1]), 2 * pg - float(q[0])], rat=rat, extra=ex, bud=bud, npos=int(sum(x > 0 for x in pl)), hdr=hdr, S=S)
for k in ("rm", "q3_4b"):
    c = curve(lambda f: f"{R}/boot/{k}/boot_tb_f{fs(f)}.json", k); b = curve(lambda f: f"{R}/boot_B/{k}/boot_tb_f{fs(f)}.json", k)
    P(f"[{k}] pooled plug {c['plug']:+.4f} bm {c['bm']:+.4f} sd {c['sd']:.4f} pct [{c['pct'][0]:+.4f},{c['pct'][1]:+.4f}] basic [{c['basic'][0]:+.4f},{c['basic'][1]:+.4f}] npos {c['npos']}/6")
    P("  ratios", c["rat"]); P("  extra errors vs agree_acc", c["extra"]); P("  budgets", c["bud"]); P("  header (n_pool, n_boot, seeds, nres, len arm, resample budgets, 8seed len, budget, n_err)", c["hdr"])
    P(f"[{k}] shardB plug {b['plug']:+.4f} bm {b['bm']:+.4f} sd {b['sd']:.4f} pct [{b['pct'][0]:+.4f},{b['pct'][1]:+.4f}] basic [{b['basic'][0]:+.4f},{b['basic'][1]:+.4f}] npos {b['npos']}/6"); P("  ratios", b["rat"])
    # leave out 25 / 30
    pl = [np.log(c["rat"][fs(f)]) for f in FR]; P("  S_tb w/o 25,30 (from rounded ratios):", round(float(np.mean([pl[0], pl[1], pl[2], pl[5]])), 4))
    # fraction of resamples S>0, and prob that bootstrap mean differs
    P("  frac S>0:", float((c["S"] > 0).mean()), " plug percentile rank in bootstrap:", float((c["S"] < c["plug"]).mean()))
    # bias-corrected: percentile CI after subtracting (bm - plug)
    P("  bias-centred CI (pct - (bm-plug)):", [round(c["pct"][0] - (c["bm"] - c["plug"]), 4), round(c["pct"][1] - (c["bm"] - c["plug"]), 4)])
    # n_err check across shards of resamples: same indices across budgets already asserted
    # pooled vs mean of shards
    F = pd.read_parquet(f"{R}/pool/{k}/feats_J.parquet")
    P("  pool", len(F), "shards", F.shard.value_counts().to_dict(), "pid unique", F.pid.is_unique, "pid range", int(F.pid.min()), int(F.pid.max()), "accepted", int(F.accept.sum()), "J", int(F.judge_error.sum()), "J by shard", F.groupby("shard").judge_error.sum().to_dict(), "maj wrong", int(F.proxy_error.sum()), F.groupby("shard").proxy_error.sum().to_dict())
    acc = F.s_maj.values > 0 if k == "q3_4b" else None
    if k == "q3_4b": P("  accept == s_maj>0:", bool((F.accept.values == (F.s_maj.values > 0)).all()))
    else:
        K = int(round(0.8575 * 4000)); thr = np.sort(F.s_maj.values)[::-1][K - 1]; P("  rm accept K", K, "n accepted", int(F.accept.sum()), "min accepted s_maj", float(F.s_maj[F.accept == 1].min()), "max rejected", float(F.s_maj[F.accept == 0].max()))
    P("  J == accept & majority_correct==0:", bool((F.judge_error.values == ((F.accept.values == 1) & (F.majority_correct.values == 0)).astype(int)).all()))
    P("  any_truncated groups", int(F.any_truncated.sum()), "of", len(F), "; unanimous accepted J", int(F.judge_error[(F.accept == 1) & (F.agreement >= 1 - 1e-9)].sum()))
    # stratified AUC independent
    m = (F.accept.values == 1) & (F.agreement.values < 1 - 1e-9); y = F.judge_error.values[m].astype(int); sc = (F.s_oth.values if k == "rm" else -F.s_diff.values)[m]; st = np.round(F.agreement.values[m] * 8).astype(int)
    def sauc(y, s, st):
        num = den = 0.0
        for g in np.unique(st):
            i = st == g; yy, ss = y[i], s[i]; pos = ss[yy == 1]; neg = ss[yy == 0]
            if len(pos) == 0 or len(neg) == 0: continue
            d = pos[:, None] - neg[None, :]; num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
        return num / den
    P("  strat AUC", round(sauc(y, sc, st), 4), "n", len(y), "err", int(y.sum()))
    rng = np.random.default_rng(1); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); bs.append(sauc(y[i], sc[i], st[i]))
    P("  AUC CI (my own 1000 boots):", np.percentile(bs, [2.5, 97.5]).round(4).tolist())
    # AUC by shard
    for sh in ("A", "B"):
        ms = m & (F.shard.values == sh); ys = F.judge_error.values[ms].astype(int); ss = (F.s_oth.values if k == "rm" else -F.s_diff.values)[ms]; sts = np.round(F.agreement.values[ms] * 8).astype(int)
        P(f"  shard {sh} strat AUC", round(sauc(ys, ss, sts), 4), "n", len(ys), "err", int(ys.sum()))
    # tie-break using cumulative tiers
    cum = pd.Series(np.round(F.agreement.values[m] * 8).astype(int)).value_counts().sort_index().cumsum().to_dict(); P("  cumulative sub-unanimous accepted by tier", cum)
    # shard overlap of questions
    qa = set(F.question[F.shard == "A"]); qb = set(F.question[F.shard == "B"]); P("  shared questions A,B:", len(qa & qb))
    ref = pd.read_parquet(f"{R}/ref_groups_0_3999.parquet"); P("  shared with ref 0..3999:", len(set(F.question) & set(ref.question)), "ref len", len(ref))
    # features in the replay: any gold-derived?
    P("  columns", list(F.columns))
    P("  corr of s_oth with majority_correct (info only) %.3f" % np.corrcoef(F.s_oth, F.majority_correct)[0, 1])
rm = pd.read_parquet(f"{R}/pool/rm/feats_J.parquet"); q3 = pd.read_parquet(f"{R}/pool/q3_4b/feats_J.parquet")
both = int(((rm.judge_error == 1) & (q3.set_index('pid').loc[rm.pid, 'judge_error'].values == 1)).sum()); a_, b_ = int(rm.judge_error.sum()), int(q3.judge_error.sum())
P("error overlap", both, both / min(a_, b_), both / (a_ + b_ - both))
open(f"{R}/_critic/check.txt", "w").write("\n".join(out))

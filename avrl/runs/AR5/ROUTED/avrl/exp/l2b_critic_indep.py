"""L2b critic: independent checks, CPU only, no import of the L1b or L2b replay modules.
1. Rebuild groups from rollouts.parquet + gold and compare to groups.parquet / feats.parquet.
2. Reimplement the plain fixed rule and configs A and B (own loop, own bookkeeping) and replay splits 0 to 7.
3. Count verifier calls for each policy independently.
4. Pool-level noise: split the pool into two disjoint halves of cell100 buckets and report the gap on each half.
"""
import json, sys
import numpy as np, pandas as pd
RUN = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b"
OUT = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/L2b_critic"
R = pd.read_parquet(f"{RUN}/rollouts.parquet"); G = pd.read_parquet(f"{RUN}/groups.parquet"); F = pd.read_parquet(f"{RUN}/feats.parquet")
print("rollouts", R.shape, "groups", G.shape, "feats", F.shape, "nan feats:", int(F[["agreement","n_distinct","any_truncated","proxy_error","cell100"]].isna().sum().sum()))
print("rollouts per pid", R.groupby("pid").size().unique(), "unique pid", R.pid.nunique(), "dup questions in groups", int(G.question.duplicated().sum()))
# 1. rebuild
agg = R.groupby("pid").agg(n_distinct=("answer", "nunique"), any_trunc=("truncated", "max"), trunc_rate=("truncated", "mean"), empty=("answer", lambda s: (s == "").mean()))
mx = R.groupby(["pid", "answer"]).size().groupby("pid").max() / 8
G2 = G.set_index("pid")
print("agreement match", bool(np.allclose(mx.values, G2.loc[mx.index, "agreement"].values)),
      "n_distinct match", bool((agg.n_distinct.values == G2.loc[agg.index, "n_distinct"].values).all()),
      "any_trunc match", bool((agg.any_trunc.values == G2.loc[agg.index, "any_truncated"].values).all()))
print("rollout truncation rate", float(R.truncated.mean()), "groups with any truncated", float(G.any_truncated.mean()), "empty-answer rollout rate", float((R.answer == "").mean()))
# majority recomputed from answers, error vs gold
def maj(s):
    v = s.value_counts().sort_index(); return v.index[np.argmax(v.values)]
m = R.groupby("pid").answer.apply(maj)
print("majority matches", bool((m.values == G2.loc[m.index, "majority"].values).all()),
      "proxy_error recomputed", float((m.values != G2.loc[m.index, "gold"].values).mean()), "stored", float(G.proxy_error.mean()))
full = pd.read_parquet("/homes/rdubash/avrl_live/ROUTED/avrl/data/gsm8k_train.parquet")
p0q = set(pd.read_parquet("/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/groups.parquet").question)
print("question overlap with p0:", len(set(G.question) & p0q), "of", len(G), "| in gsm8k_train:", int(G.question.isin(set(full.question)).sum()), "| full rows", len(full))
gold_ref = full.set_index("question").answer.str.split("####").str[-1].str.strip().str.replace(",", "")
def norm(x):
    try: return f"{float(x):.6g}"
    except: return x
gg = G.set_index("question").gold
print("gold matches dataset answer", bool(all(norm(gold_ref[q]) == g for q, g in gg.items())))

# 2. independent replay
MIN = 3
def build(df, seed):
    cnt = df.cell100.value_counts(); keep = cnt[cnt >= MIN].index
    d = df[df.cell100.isin(keep)].reset_index(drop=True)
    groups = d.cell100.unique(); np.random.default_rng(seed).shuffle(groups)
    fit = set(groups[: len(groups) // 4]); return d[~d.cell100.isin(fit)].reset_index(drop=True)
def fixed(ag, nd, y, b, s):
    rnd = np.random.default_rng(s).random(len(y)); idx = sorted(range(len(y)), key=lambda i: (ag[i], -nd[i], rnd[i]))
    return int(sum(y[i] for i in idx[:b])), b
def recal(ag, nd, tr, y, b, s, k0, strat):
    n = len(y); rnd = np.random.default_rng(s).random(n)
    key = [(ag[i], nd[i]) if strat == "nd" else (ag[i], nd[i], tr[i]) for i in range(n)]
    prior = {k: 1 - k[0] for k in set(key)}; table = dict(prior)
    Rr = max(b // 8, 10); seen = [False] * n; hits = calls = 0; nxt = Rr
    while calls < b:
        order = sorted(range(n), key=lambda i: (-table[key[i]], ag[i], -nd[i], rnd[i]))
        for i in order:
            if seen[i]: continue
            seen[i] = True; hits += y[i]; calls += 1
            if calls >= nxt:
                v, e = {}, {}
                for j in range(n):
                    if seen[j]: v[key[j]] = v.get(key[j], 0) + 1; e[key[j]] = e.get(key[j], 0) + y[j]
                table = {k: (e.get(k, 0) + k0 * prior[k]) / (v.get(k, 0) + k0) for k in prior}; nxt += Rr
                break
            if calls >= b: break
    return hits, calls
def gap_table(df_for_splits, tag):
    res = {"A": [], "B": []}; fx = {"A": [], "B": []}; cf = {"A": [], "B": []}; callcheck = []
    for sp in range(8):
        d = df_for_splits(sp); ag, nd, tr, y = d.agreement.tolist(), d.n_distinct.tolist(), d.any_truncated.tolist(), d.proxy_error.astype(int).tolist()
        for t, f, k0, st in (("A", 0.05, 2, "nd"), ("B", 0.10, 50, "nd_tr")):
            b = int(f * len(y)); g = []; hf = []; hc = []
            for s in range(8):
                hx, c1 = fixed(ag, nd, y, b, s); hr, c2 = recal(ag, nd, tr, y, b, s, k0, st)
                callcheck += [c1 == b, c2 == b]; g.append(100 * (hr - hx) / b); hf.append(100 * hx / b); hc.append(100 * hr / b)
            res[t].append(float(np.mean(g))); fx[t].append(float(np.mean(hf))); cf[t].append(float(np.mean(hc)))
    out = {t: {"gap_per_split": res[t], "mean": float(np.mean(res[t])), "sd": float(np.std(res[t], ddof=1)), "fixed": float(np.mean(fx[t])), "cfg": float(np.mean(cf[t]))} for t in res}
    print(tag, json.dumps({t: {k: (round(v, 4) if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in o.items()} for t, o in out.items()}), "| all calls equal budget:", all(callcheck))
    return out
main = gap_table(lambda sp: build(F, sp), "indep main (8 splits)")
json.dump(main, open(f"{OUT}/indep_main.json", "w"), indent=1)
stored = json.load(open(f"{RUN}/result.json"))["configs"]
for t in "AB":
    print(t, "max abs diff vs stored per-split gaps (pp):", float(np.max(np.abs(np.array(main[t]["gap_per_split"]) - np.array(stored[t]["gap_pp_per_split"])))))

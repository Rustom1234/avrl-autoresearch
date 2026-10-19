"""L5a critic: independent checks. CPU only. Does NOT import l1b_tuned_threshold, l5a_frozen_table or any L4b module.
(1) Rebuild agreement, n_distinct, any_truncated, majority, proxy_error from rollouts.parquet for p0, L2b, L4b and compare to feats.parquet.
    Check the majority tie rule is not gold-aware and that gold never enters the score.
(2) Question-level disjointness of p0 from L2b and L4b (feats.question), and of L2b from L4b.
(3) Refit the p0 frozen table with own code (k0 50, prior 1 - agreement), compare with result.json["table"].
(4) Reimplement plain fixed and the frozen ranking, replay splits 0 to 7 x 8 tie seeds at 10% on L2b, L4b and p0, compare every
    per-split gap, fixed hit rate and frozen hit rate with result.json. Exact distinct-doc spend per policy.
(5) Bucket bootstrap of the frozen gap (resample 75 of the cell100 buckets with replacement) on L2b and L4b.
(6) Composition: L4b test rows (pid < 1319) and remaining-train rows (pid >= 1319) separately, and random 1,319-prompt halves
    of L4b and L2b, same frozen table, stand-in buckets from the joint cell100 clustering (as in the L4b critic).
Usage: /homes/rdubash/active-audit/.venv/bin/python -I l5a_critic_indep.py
"""
import json
import numpy as np, pandas as pd

RUN = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
S = ["agreement", "n_distinct", "any_truncated"]
K0, FRAC = 50.0, 0.10


def norm(x):
    try: return f"{float(x):.6g}"
    except Exception: return x


def rebuild(pool):
    R = pd.read_parquet(f"{RUN}/{pool}/rollouts.parquet"); F = pd.read_parquet(f"{RUN}/{pool}/feats.parquet")
    G = pd.read_parquet(f"{RUN}/{pool}/groups.parquet")
    rows = []
    for pid, d in R.groupby("pid"):
        ans = d["answer"].astype(str).tolist(); vals, cnt = np.unique(ans, return_counts=True)
        maj = vals[np.argmax(cnt)]; ties = int((cnt == cnt.max()).sum())
        rows.append((pid, maj, cnt.max() / len(ans), len(vals), int(d["tokens"].max() >= 512), ties))
    B = pd.DataFrame(rows, columns=["pid", "maj", "ag", "nd", "trunc", "ties"]).set_index("pid").sort_index()
    F = F.set_index("pid").sort_index(); G = G.set_index("pid").sort_index()
    out = dict(pool=pool, n=len(F), ag_diff=float(np.abs(B.ag - F.agreement).max()), nd_diff=int((B.nd != F.n_distinct).sum()),
               trunc_diff=int((B.trunc != F.any_truncated).sum()), maj_diff=int((B.maj.astype(str) != G.majority.astype(str)).sum()),
               err_diff=int((((B.maj.astype(str) != G.gold.astype(str)).astype(int)) != F.proxy_error).sum()),
               tied_top=int((B.ties > 1).sum()), n_err=int(F.proxy_error.sum()), feats_vs_groups=int((F.proxy_error != G.proxy_error).sum()))
    return out


def fit_table(df):
    g = df.assign(_y=df["proxy_error"].astype(int)).groupby(S)["_y"].agg(["sum", "size"]).reset_index()
    g["prior"] = 1 - g["agreement"]; g["post"] = (g["sum"] + K0 * g["prior"]) / (g["size"] + K0)
    return {(r.agreement, int(r.n_distinct), int(r.any_truncated)): r.post for r in g.itertuples()}, g


def score_of(d, tab):
    return np.array([tab.get((a, int(n), int(t)), 1 - a) for a, n, t in zip(d.agreement, d.n_distinct, d.any_truncated)])


def build(df, split, G="cell100", MIN=3):
    cnt = df[G].value_counts(); d = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    g = d[G].unique().copy(); np.random.default_rng(split).shuffle(g)
    drop = set(g[: len(g) // 4].tolist())
    return d[~d[G].isin(drop)].reset_index(drop=True)


def picks(d, sc, B, seed):
    y = d["proxy_error"].to_numpy().astype(int); ag = d.agreement.to_numpy(float); nd = d.n_distinct.to_numpy(int)
    r = np.random.default_rng(seed).random(len(d))
    f = sorted(range(len(d)), key=lambda i: (ag[i], -nd[i], r[i]))[:B]
    z = sorted(range(len(d)), key=lambda i: (-sc[i], ag[i], -nd[i], r[i]))[:B]
    return y, f, z


def replay(F, tab, splits=range(8), seeds=range(8), frac=FRAC):
    F = F.assign(sc=score_of(F, tab)); res = []; bad = 0
    for s in splits:
        d = build(F, s); B = int(frac * len(d)); gaps, fh, zh = [], [], []
        for sd in seeds:
            y, f, z = picks(d, d.sc.to_numpy(), B, sd)
            if not (len(f) == len(set(f)) == B and len(z) == len(set(z)) == B): bad += 1
            fh.append(100 * y[f].sum() / B); zh.append(100 * y[z].sum() / B); gaps.append(zh[-1] - fh[-1])
        res.append(dict(split=s, n=len(d), B=B, gap=float(np.mean(gaps)), fixed=float(np.mean(fh)), frozen=float(np.mean(zh))))
    return res, bad


def stats(g):
    g = np.asarray(g, float); return f"mean {g.mean():+.2f} sd {g.std(ddof=1):.2f} above2 {(g > 2).sum()} of {len(g)}"


def bootstrap(F, tab, nb=100, seeds=3, nsplit=8):
    """Resample 75 of the cell100 buckets with replacement (pool-level noise). Buckets with fewer than 3 docs dropped as in build()."""
    F = F.assign(sc=score_of(F, tab)); cnt = F.cell100.value_counts(); F = F[F.cell100.isin(cnt[cnt >= 3].index)]
    by = {b: g.reset_index(drop=True) for b, g in F.groupby("cell100")}; keys = list(by); out = []
    for i in range(nb):
        rng = np.random.default_rng(9000 + i); pick = rng.choice(len(keys), size=75, replace=True)
        d = pd.concat([by[keys[j]] for j in pick], ignore_index=True); B = int(FRAC * len(d)); g = []
        for sd in range(seeds):
            y, f, z = picks(d, d.sc.to_numpy(), B, sd); g.append(100 * (y[z].sum() - y[f].sum()) / B)
        out.append(float(np.mean(g)))
    return np.array(out)


if __name__ == "__main__":
    res = json.load(open(f"{RUN}/L5a/result.json"))
    print("== (1) rebuild features from rollouts")
    for p in ["p0", "L2b", "L4b"]: print(rebuild(p))
    print("== (2) question overlap")
    P = {p: pd.read_parquet(f"{RUN}/{p}/feats.parquet") for p in ["p0", "L2b", "L4b"]}
    q = {p: set(P[p].question) for p in P}
    print("p0-L2b", len(q["p0"] & q["L2b"]), "p0-L4b", len(q["p0"] & q["L4b"]), "L2b-L4b", len(q["L2b"] & q["L4b"]),
          "dups within L4b", int(P["L4b"].question.duplicated().sum()), "within L2b", int(P["L2b"].question.duplicated().sum()))
    print("== (3) frozen table")
    tab, g = fit_table(P["p0"])
    rt = {(r["agreement"], int(r["n_distinct"]), int(r["any_truncated"])): r["post"] for r in res["table"]}
    print("strata", len(tab), "result.json", len(rt), "max abs post diff", max(abs(tab[k] - rt[k]) for k in tab), "keys equal", set(tab) == set(rt),
          "errors", int(g["sum"].sum()), "count", int(g["size"].sum()))
    for p in ["L2b", "L4b"]:
        keys = set(zip(P[p].agreement, P[p].n_distinct.astype(int), P[p].any_truncated.astype(int)))
        miss = [k for k in keys if k not in tab]; print(p, "eval strata", len(keys), "missing from p0 table", len(miss))
    print("== (4) independent replay vs result.json")
    for p in ["L2b", "L4b", "p0"]:
        r, bad = replay(P[p], tab); rj = res["pools"][p]
        dg = max(abs(a["gap"] - b) for a, b in zip(r, rj["gap_pp_per_split"]))
        df_ = max(abs(a["fixed"] - b) for a, b in zip(r, rj["fixed_pct_per_split"]))
        dz = max(abs(a["frozen"] - b) for a, b in zip(r, rj["frozen_pct_per_split"]))
        print(p, "spend failures", bad, "max diff gap", dg, "fixed", df_, "frozen", dz, "|", stats([a["gap"] for a in r]),
              "| fixed mean", round(np.mean([a["fixed"] for a in r]), 2), "frozen mean", round(np.mean([a["frozen"] for a in r]), 2))
    print("== (5) bucket bootstrap (100 resamples, 3 tie seeds)")
    for p in ["L2b", "L4b"]:
        b = bootstrap(P[p], tab)
        print(p, f"mean {b.mean():+.2f} sd {b.std(ddof=1):.2f} 95% [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}] above +2.0: {(b > 2).sum()} of {len(b)}")
    print("== (6) composition (L4b test pid<1319 vs remaining train; stand-in joint cell100 buckets)")
    L4 = P["L4b"]
    for name, sub in (("L4b full", L4), ("L4b test only", L4[L4.pid < 1319]), ("L4b remaining train", L4[L4.pid >= 1319])):
        r, bad = replay(sub, tab)
        print(name, "n", len(sub), "err", round(float(sub.proxy_error.mean()), 3), "spend bad", bad, "|", stats([a["gap"] for a in r]),
              "| fixed", round(np.mean([a["fixed"] for a in r]), 2), "frozen", round(np.mean([a["frozen"] for a in r]), 2),
              "| calls", [a["B"] for a in r][:3], "per split", [round(a["gap"], 2) for a in r])
    for p in ["L4b", "L2b"]:
        halves = []
        for i in range(6):
            rng = np.random.default_rng(100 + i); idx = rng.permutation(len(P[p]))[:1319]
            r, _ = replay(P[p].iloc[sorted(idx)].reset_index(drop=True), tab); halves.append(np.mean([a["gap"] for a in r]))
        print(p, "6 random 1,319-prompt halves, frozen gap per half", [round(h, 2) for h in halves], "mean", round(float(np.mean(halves)), 2), "above 2.0:", int(sum(h > 2 for h in halves)))

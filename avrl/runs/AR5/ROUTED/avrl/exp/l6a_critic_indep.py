"""L6a critic: independent recompute of the L6a verdict numbers (no import of l6a_test_rows or l1b), a leakage check, and
eval-side noise checks (bucket bootstrap, 64 fresh bucket cuts) on the registered test rows. CPU only.

Spec re-implemented from the pre-registration: strata (agreement, n_distinct, any_truncated); posterior
(errors + 50 * (1 - agreement)) / (count + 50); score of an unseen stratum = prior; frozen order = score desc, then agreement asc,
n_distinct desc, then a per-tie-seed random draw; plain fixed = agreement asc, n_distinct desc, same random draw.
Sim split: cell100 buckets with at least 3 docs, order of first appearance, default_rng(split).shuffle, first quarter dropped,
the rest is the sim. Budget int(0.10 * n_sim). Gap = frozen hit rate minus fixed hit rate, paired by tie seed, mean over seeds.
"""
import numpy as np, pandas as pd, json, sys

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
p0 = pd.read_parquet(f"{R}/p0/feats.parquet"); l4 = pd.read_parquet(f"{R}/L4b/feats.parquet").reset_index(drop=True)
test = l4[l4.pid <= 1318].reset_index(drop=True)
ST = ["agreement", "n_distinct", "any_truncated"]
res = json.load(open(f"{R}/L6a/result.json"))


def table(fit):
    g = fit.groupby(ST).agg(e=("proxy_error", "sum"), n=("proxy_error", "size")).reset_index()
    g["post"] = (g.e + 50 * (1 - g.agreement)) / (g.n + 50)
    return g[ST + ["post"]]


def score(df, tab):
    m = df[ST].merge(tab, on=ST, how="left")
    return np.where(m.post.isna(), 1 - df.agreement.to_numpy(), m.post.to_numpy())


def sim_of(df, split):
    cnt = df.cell100.value_counts(); d = df[df.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = d.cell100.unique(); np.random.default_rng(split).shuffle(g)
    fit = set(g[: len(g) // 4]); return d[~d.cell100.isin(fit)].reset_index(drop=True)


def gap(sim, scores_list, seeds, y=None):
    """scores_list: list of score arrays (one per table). Returns gap per table (mean over tie seeds, pp) and fixed pct."""
    n = len(sim); b = int(0.10 * n); ag = sim.agreement.to_numpy(); nd = sim.n_distinct.to_numpy()
    y = sim.proxy_error.to_numpy(int) if y is None else y
    out = np.zeros(len(scores_list)); fx = 0.0
    for s in seeds:
        rnd = np.random.default_rng(s).random(n)
        hf = y[np.lexsort((rnd, -nd, ag))[:b]].sum(); fx += 100 * hf / b / len(seeds)
        for j, sc in enumerate(scores_list):
            hz = y[np.lexsort((rnd, -nd, ag, -sc))[:b]].sum(); out[j] += 100 * (hz - hf) / b / len(seeds)
    return out, fx


def with_scores(df, tabs):
    return [score(df, t) for t in tabs]


T3 = table(p0)
tabs200 = [table(p0.iloc[np.random.RandomState(100 + i).choice(len(p0), 200, replace=False)]) for i in range(20)]
print("strata in 3000 table", len(T3), "| 200-label strata counts", [len(t) for t in tabs200])

# 1) exact recompute of the registered statistic, splits 0 to 7, tie seeds 0 to 7
sc_t = with_scores(test, [T3] + tabs200)
gaps = np.zeros((8, 21)); fixed = []
for s in range(8):
    sim = sim_of(test, s); idx = sim.index  # sim reset; recompute scores on sim rows
    scs = with_scores(sim, [T3] + tabs200)
    g, f = gap(sim, scs, range(8)); gaps[s] = g; fixed.append(f)
print("n_sim per split", [len(sim_of(test, s)) for s in range(8)], "budgets", [int(0.1 * len(sim_of(test, s))) for s in range(8)])
m3 = gaps[:, 0]; m200 = gaps[:, 1:].mean(0)
print(f"3000 table: per split {np.round(m3, 2).tolist()} mean {m3.mean():.4f} sd {m3.std(ddof=1):.3f} above2 {(m3 > 2).sum()} fixed {np.mean(fixed):.4f}")
print(f"200 tables: subset means {np.round(m200, 2).tolist()}\n  mean {m200.mean():.4f} sd {m200.std(ddof=1):.3f} above2 {(m200 > 2).sum()} of 20")
r3 = np.array(res["table_3000"]["gap_pp_per_split"]); r200 = np.array(res["table_200"]["subset_mean_pp"])
print(f"max abs diff vs result.json: 3000 per split {np.abs(m3 - r3).max():.2e}, 200 subset means {np.abs(m200 - r200).max():.2e}")
print("verdicts from independent recompute:", "PASS" if m3.mean() > 2 else "KILL", "/", "PASS" if m200.mean() > 2 else "KILL")
print(f"distinct subset means (rounded 1e-6): {len(set(np.round(m200, 6)))} of 20")

# 2) leakage: picks must not depend on eval labels. Permute eval labels, picks must be identical (check by recomputing with random y).
sim = sim_of(test, 0); scs = with_scores(sim, [T3]); n = len(sim); b = int(0.10 * n)
ag = sim.agreement.to_numpy(); nd = sim.n_distinct.to_numpy(); rnd = np.random.default_rng(0).random(n)
pick = np.lexsort((rnd, -nd, ag, -scs[0]))[:b]
ycopy = sim.drop(columns=["proxy_error", "gold", "majority", "majority_correct", "pass_rate"])
sim2 = ycopy.copy(); sc2 = with_scores(sim2, [T3])
pick2 = np.lexsort((rnd, -sim2.n_distinct.to_numpy(), sim2.agreement.to_numpy(), -sc2[0]))[:b]
print("picks identical with gold/label columns removed from the eval frame:", bool((pick == pick2).all()),
      "| fit features used:", ST, "| fit labels from p0 only; p0/test question overlap", int(test.question.isin(set(p0.question)).sum()))
print("p0 fit table rows: proxy_error == (majority != gold):", bool(((p0.majority != p0.gold).astype(int) == p0.proxy_error).all()))

# 3) bucket bootstrap on the test rows (resample cell100 buckets with replacement), splits 0 to 7, 3 tie seeds
rng = np.random.default_rng(12345); bk = {k: v for k, v in test.groupby("cell100")}; keys = np.array(list(bk))
cnt = test.cell100.value_counts(); print("test cell100 buckets", len(keys), "with >=3 docs", int((cnt >= 3).sum()))
B = 200; b3 = np.zeros(B); b200 = np.zeros(B)
for r in range(B):
    ch = rng.choice(len(keys), len(keys), replace=True)
    d = pd.concat([bk[keys[c]].assign(cell100=np.int64(j)) for j, c in enumerate(ch)], ignore_index=True)
    tot = np.zeros(21)
    for s in range(8):
        sim = sim_of(d, s); g, _ = gap(sim, with_scores(sim, [T3] + tabs200), range(3)); tot += g / 8
    b3[r] = tot[0]; b200[r] = tot[1:].mean()
for nm, v in (("3000-label", b3), ("200-label mean of 20", b200)):
    print(f"bucket bootstrap {nm}: mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} 95% [{np.percentile(v, 2.5):+.2f}, {np.percentile(v, 97.5):+.2f}] above +2.0 {int((v > 2).sum())} of {B}")

# 4) fresh cuts: 64 new bucket-shuffle seeds (100 to 163) on the unmodified test rows, 8 tie seeds
c3 = np.zeros(64); c200 = np.zeros(64)
for i, s in enumerate(range(100, 164)):
    sim = sim_of(test, s); g, _ = gap(sim, with_scores(sim, [T3] + tabs200), range(8)); c3[i] = g[0]; c200[i] = g[1:].mean()
for nm, v in (("3000-label", c3), ("200-label mean of 20", c200)):
    print(f"64 fresh cuts {nm}: mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} min {v.min():+.2f} max {v.max():+.2f} above +2.0 {int((v > 2).sum())} of 64")
print("pooled 72 cuts (8 registered + 64 fresh) 3000-label mean", f"{np.concatenate([m3, c3]).mean():+.2f}", "200-label mean", f"{np.concatenate([gaps[:, 1:].mean(1), c200]).mean():+.2f}")

# 5) label cost
print("fit labels: 3000 and 200 versus eval-time budget per split", [int(0.1 * len(sim_of(test, s))) for s in range(8)])

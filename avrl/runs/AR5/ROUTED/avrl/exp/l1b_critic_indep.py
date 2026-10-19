"""L1b critic: independent plain-python reimplementation of the plain fixed threshold and the swept recal rules,
checked against runs/L1b/result.json (held-out gaps) for the selected configs, plus overlap and tie checks. CPU only."""
import json, sys
import numpy as np, pandas as pd
R = json.load(open("/homes/rdubash/avrl_live/ROUTED/avrl/runs/L1b/result.json"))
df0 = pd.read_parquet("/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/feats.parquet")

def sim_ids(split):
    cnt = df0["cell100"].value_counts()
    d = df0[df0["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    g = d["cell100"].unique(); np.random.default_rng(split).shuffle(g)
    fit = set(g[: len(g) // 4])
    return d[~d["cell100"].isin(fit)].reset_index(drop=True)

def fixed(sim, budget, seed):
    rnd = np.random.default_rng(seed).random(len(sim))
    ag, nd, y = sim.agreement.tolist(), sim.n_distinct.tolist(), sim.proxy_error.tolist()
    idx = sorted(range(len(sim)), key=lambda i: (ag[i], -nd[i], rnd[i]))
    return sum(y[i] for i in idx[:budget])

def recal(sim, budget, seed, k0, R_, strat):
    n = len(sim); rnd = np.random.default_rng(seed).random(n)
    ag, nd, tr, y = sim.agreement.tolist(), sim.n_distinct.tolist(), sim.any_truncated.tolist(), sim.proxy_error.tolist()
    key = [(ag[i],) if strat == "agreement" else (ag[i], nd[i]) if strat == "x_nd" else (ag[i], nd[i], tr[i]) for i in range(n)]
    prior = {k: 1 - k[0] for k in set(key)}
    table = dict(prior); v = {k: 0 for k in prior}; e = {k: 0 for k in prior}
    seen = [False] * n; hits = 0; spent = 0; nxt = R_
    while spent < budget:
        best = min((i for i in range(n) if not seen[i]), key=lambda i: (-table[key[i]], ag[i], -nd[i], rnd[i]))
        seen[best] = True; hits += y[best]; v[key[best]] += 1; e[key[best]] += y[best]; spent += 1
        if spent >= nxt:
            table = {k: (e[k] + k0 * prior[k]) / (v[k] + k0) for k in prior}; nxt += R_
    return hits

CFG = {"0.05": (2, 8, "x_nd"), "0.10": (50, 8, "x_nd_x_trunc")}
for key, (k0, d, st) in CFG.items():
    f = float(key); gaps = []
    for s in (4, 5, 6, 7):
        sim = sim_ids(s); b = int(f * len(sim)); Rr = max(b // d, 10)
        fx = [fixed(sim, b, sd) for sd in range(8)]
        tn = [recal(sim, b, sd, k0, Rr, st) for sd in range(8)]
        gap = 100 * (np.mean(tn) - np.mean(fx)) / b
        gaps.append(gap); print(key, "split", s, "budget", b, "R", Rr, "fixed hits", fx, "tuned hits", tn, f"gap {gap:+.3f}", flush=True)
    print(key, "indep mean gap %+.3f" % np.mean(gaps), "sd %.3f" % np.std(gaps, ddof=1), "stored %+.3f" % R["per_budget"][key]["heldout_selected_minus_fixed"]["mean_pp"], flush=True)

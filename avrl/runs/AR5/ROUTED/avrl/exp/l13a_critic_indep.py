"""L13a critic: independent replay from the raw feats files. CPU only, no new rollouts, no GPU job.

Imports nothing from the experiment modules (no l13a_*, l10b_*, l1b_*, g01_*, p0_* import). Own code for: the cell100 sim split,
the frozen 200-label table, plain fixed, the five 160-label fold tables, the out-of-fold p0 posterior, the matched eval posterior
(draw RandomState(400 + i)), the offset-ridge refit solved by Newton's method (not L-BFGS), and the greedy learner.

Replays the two full pools and the two registered L4b pid cuts: 20 subsets x split seeds 0 to 7 x 8 tie seeds. Checks against
runs/L13a/<pool>/result.json, per subset and split, to 1e-9 pp: table_shrunk minus frozen, table_zero minus frozen, frozen minus
fixed. Also checks spend (every arm takes int(0.10 x n_sim) distinct docs) and that every refit solves to max |grad| < 1e-8.
Usage: python -I l13a_critic_indep.py --out OUTDIR [--pools L2b L4b L4b_pid_le1318 L4b_pid_gt1318]
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.special import expit

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
C6 = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"]
ST = ["agreement", "n_distinct", "any_truncated"]
LAB = "proxy_error"; K0 = 50.0; BUDGET = 0.10; NSUB = 20; SUB = 200; NSEED = 8; NSPLIT = 8
TOL_PP = 1e-9
# name: (eval feats, keep pid >= lo, keep pid <= hi, key of the pool inside the result, L13a result dir)
POOLS = {"L2b": (f"{R}/L2b/feats.parquet", None, None, "L2b", f"{R}/L13a/L2b"),
         "L4b": (f"{R}/L4b/feats.parquet", None, None, "L4b", f"{R}/L13a/L4b"),
         "L4b_pid_le1318": (f"{R}/L4b/feats.parquet", None, 1318, "L4b", f"{R}/L13a/L4b_pid_le1318"),
         "L4b_pid_gt1318": (f"{R}/L4b/feats.parquet", 1319, None, "L4b", f"{R}/L13a/L4b_pid_gt1318")}
P0 = pd.read_parquet(f"{R}/p0/feats.parquet")


def table_from(df, k0=K0):
    """Own frozen table: per stratum (agreement, n_distinct, any_truncated), posterior (errors + k0 prior) / (count + k0), prior 1 - agreement."""
    g = df.assign(_y=df[LAB].to_numpy(int)).groupby(ST)["_y"].agg(["sum", "size"])
    prior = 1.0 - g.index.get_level_values("agreement").to_numpy(float)
    post = (g["sum"].to_numpy(float) + k0 * prior) / (g["size"].to_numpy(float) + k0)
    return pd.Series(post, index=g.index)


def lookup(df, tab):
    """Posterior of each row's stratum; a stratum missing from the table scores its prior 1 - agreement."""
    post = tab.reindex(pd.MultiIndex.from_frame(df[ST])).to_numpy(float)
    return np.where(np.isnan(post), 1.0 - df["agreement"].to_numpy(float), post)


def sim_split(df, split):
    """Own cell100 split: keep buckets with at least 3 rows, shuffle the bucket list with default_rng(split), drop the first quarter."""
    cnt = df["cell100"].value_counts()
    df = df[df["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    groups = df["cell100"].unique()
    np.random.default_rng(split).shuffle(groups)
    fit = set(groups[: len(groups) // 4].tolist())
    return df[~df["cell100"].isin(fit)].reset_index(drop=True)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def offset_ridge_newton(Z, y, off, C=1.0):
    """Minimise C * sum_i w_i logloss_i + 0.5 ||beta||^2 over (beta, b) by Newton's method, with linear predictor off + Z beta + b."""
    n, p = Z.shape
    Zb = np.column_stack([Z, np.ones(n)])
    yy = 2.0 * y - 1.0
    w = (n / (2.0 * np.bincount(y, minlength=2).astype(float)))[y]
    pen = np.append(np.ones(p), 0.0)
    th = np.zeros(p + 1)
    for _ in range(200):
        pr = expit(yy * (off + Zb @ th))
        g = Zb.T @ (-C * w * yy * (1.0 - pr)) + pen * th
        if np.max(np.abs(g)) < 1e-10:
            break
        H = (Zb * (C * w * pr * (1.0 - pr))[:, None]).T @ Zb + np.diag(pen)
        th = th - np.linalg.solve(H, g)
    else:
        raise RuntimeError("Newton did not converge")
    return th[:p], th[p], float(np.max(np.abs(g)))


def refit(Xtr, ytr, offtr, Xall, offall):
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd = np.where(sd > 0, sd, 1.0)
    beta, b, gmax = offset_ridge_newton((Xtr - mu) / sd, ytr, offtr)
    return offall + ((Xall - mu) / sd) @ beta + b, gmax


def greedy(rnd, y, ag, nd, X, off, Xp, offp, yp, budget, every, gmaxes):
    """Own greedy learner: take the top unseen doc by (score desc, agreement asc, n_distinct desc, seeded draw asc); after every
    `every` verified labels (and before the budget is spent), refit on the 200 p0 rows plus the verified eval rows."""
    seen = np.zeros(len(y), bool); score = off.copy(); chosen = []; nxt = every
    while len(chosen) < budget:
        cand = np.flatnonzero(~seen)
        k = cand[np.lexsort((rnd[cand], -nd[cand], ag[cand], -score[cand]))[0]]
        seen[k] = True; chosen.append(k)
        if every and len(chosen) >= nxt and len(chosen) < budget:
            score, gmax = refit(np.vstack([Xp, X[seen]]), np.concatenate([yp, y[seen]]),
                                np.concatenate([offp, off[seen]]), X, off)
            gmaxes.append(gmax); nxt += every
    return np.array(chosen)


def subset_job(args):
    ev, i = args
    idx = np.random.RandomState(100 + i).choice(len(P0), SUB, replace=False)
    sub = P0.iloc[idx].reset_index(drop=True)
    tab = table_from(sub)
    perm = np.random.RandomState(300 + i).permutation(SUB)
    fold = np.empty(SUB, int); fold[perm] = np.arange(SUB) % 5
    ftab = [table_from(sub[fold != f]) for f in range(5)]
    oof = np.empty(SUB)
    for f in range(5):
        m = fold == f
        oof[m] = lookup(sub[m], ftab[f])
    Xp = sub[C6].fillna(0).to_numpy(float); offp = logit(oof); yp = sub[LAB].to_numpy(int)
    draw = np.random.RandomState(400 + i).randint(0, 5, size=len(ev))
    post = np.empty(len(ev))
    for f in range(5):
        m = draw == f
        post[m] = lookup(ev[m], ftab[f])
    frz = lookup(ev, tab)
    evx = ev.assign(_post=post, _frz=frz)
    res = {"zero_minus_frozen": [], "shrunk_minus_frozen": [], "frozen_minus_fixed": [], "budget": [], "n_sim": []}
    gmaxes = []
    for s in range(NSPLIT):
        sim = sim_split(evx, s)
        y = sim[LAB].to_numpy(int); ag = sim["agreement"].to_numpy(float); nd = sim["n_distinct"].to_numpy(float)
        X = sim[C6].fillna(0).to_numpy(float); off = logit(sim["_post"].to_numpy(float)); frzs = sim["_frz"].to_numpy(float)
        n = len(sim); budget = int(BUDGET * n); every = max(budget // 8, 40)
        fx, fz, zh, sh = [], [], [], []
        for seed in range(NSEED):
            rnd = np.random.default_rng(seed).random(n)
            sel_fix = np.lexsort((rnd, -nd, ag))[:budget]
            sel_frz = np.lexsort((rnd, -nd, ag, -frzs))[:budget]
            sel_zero = greedy(rnd, y, ag, nd, X, off, Xp, offp, yp, budget, None, gmaxes)
            sel_shr = greedy(rnd, y, ag, nd, X, off, Xp, offp, yp, budget, every, gmaxes)
            for sel in (sel_fix, sel_frz, sel_zero, sel_shr):
                assert len(sel) == budget and len(np.unique(sel)) == budget, "spend check failed"
            fx.append(int(y[sel_fix].sum())); fz.append(int(y[sel_frz].sum()))
            zh.append(int(y[sel_zero].sum())); sh.append(int(y[sel_shr].sum()))
        fixp = 100.0 * np.array(fx, float) / budget; frzp = 100.0 * np.array(fz, float) / budget
        zerop = 100.0 * np.array(zh, float) / budget; shrp = 100.0 * np.array(sh, float) / budget
        res["zero_minus_frozen"].append(float((zerop - frzp).mean()))
        res["shrunk_minus_frozen"].append(float((shrp - frzp).mean()))
        res["frozen_minus_fixed"].append(float((frzp - fixp).mean()))
        res["budget"].append(budget); res["n_sim"].append(n)
    res["gmax"] = float(max(gmaxes)) if gmaxes else 0.0
    res["refits"] = len(gmaxes)
    return i, res


def replay_pool(name, feats, lo, hi):
    ev = pd.read_parquet(feats)
    keep = np.ones(len(ev), bool)
    if lo is not None: keep &= ev["pid"].to_numpy() >= lo
    if hi is not None: keep &= ev["pid"].to_numpy() <= hi
    ev = ev[keep].reset_index(drop=True)
    with Pool(min(NSUB, os.cpu_count(), 8)) as p:
        res = p.map(subset_job, [(ev, i) for i in range(NSUB)])
    return dict(sorted(res, key=lambda t: t[0])), len(ev)


def compare(name, key, rdir, res):
    ref = json.load(open(f"{rdir}/result.json"))
    det = {x["subset"]: x for x in ref["subset_detail"]}
    worst = {"shrunk_minus_frozen": 0.0, "zero_minus_frozen": 0.0, "frozen_minus_fixed": 0.0}
    for i, o in res.items():
        r = det[i]
        for k, rk in (("shrunk_minus_frozen", f"{key}_table_shrunk_minus_frozen_pp_per_split"),
                      ("frozen_minus_fixed", f"{key}_gap_pp_per_split")):
            worst[k] = max(worst[k], max(abs(a - b) for a, b in zip(o[k], r[rk])))
        worst["zero_minus_frozen"] = max(worst["zero_minus_frozen"],
                                         abs(float(np.mean(o["zero_minus_frozen"])) - float(r[f"{key}_table_zero_minus_frozen_pp"])))
    subs = [float(np.mean(o["shrunk_minus_frozen"])) for o in res.values()]
    m = np.asarray(subs)
    ok = all(v <= TOL_PP for v in worst.values())
    return {"pool": name, "result": f"{rdir}/result.json", "subsets": len(res), "replay_mean_pp": float(m.mean()),
            "replay_sd_pp": float(m.std(ddof=1)), "replay_subsets_above_1pp": int((m > 1.0).sum()),
            "result_mean_pp": float(ref["pools"][key]["mean_pp"]), "result_verdict": ref["verdict"],
            "max_abs_diff_pp": {k: float(v) for k, v in worst.items()}, "tol_pp": TOL_PP,
            "max_refit_grad": float(max(o["gmax"] for o in res.values())), "refits": int(sum(o["refits"] for o in res.values())),
            "pass": bool(ok and len(res) == NSUB)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", nargs="+", default=list(POOLS))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    report = {}
    for name in a.pools:
        feats, lo, hi, key, rdir = POOLS[name]
        res, n_eval = replay_pool(name, feats, lo, hi)
        row = compare(name, key, rdir, res)
        row["eval_rows"] = int(n_eval)
        report[name] = row
        print(f"{name}: replay table_shrunk minus frozen {row['replay_mean_pp']:+.4f} (sd {row['replay_sd_pp']:.2f}, "
              f"{row['replay_subsets_above_1pp']} of {row['subsets']} above +1.0) vs result {row['result_mean_pp']:+.4f}; "
              f"max abs diff pp {row['max_abs_diff_pp']}; refits {row['refits']}, max refit |grad| {row['max_refit_grad']:.2e}; "
              f"pass {row['pass']}", flush=True)
    with open(os.path.join(a.out, "replay.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("independent replay all pass:", all(r["pass"] for r in report.values()), flush=True)
    if not all(r["pass"] for r in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

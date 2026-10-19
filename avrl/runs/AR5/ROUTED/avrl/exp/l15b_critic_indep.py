"""L15b critic: independent replay from the raw feats files. CPU only, no GPU job, no new rollouts.

Imports nothing from the experiment modules (no l15b_*, l13a_*, l1b_* import). Own code for the cell100 sim split, the frozen
200-label table, plain fixed, the occupancy weights and the weighted draw.

Checks (writes runs/L15b_critic/result.json):
  A. Redraw each of the 48 registered pools with own weights and RandomState(seed).choice and compare with the stored pool_pids.
  B. Replay frozen minus fixed on the stored pools (20 subsets x split seeds 0-7 x 8 tie seeds) and compare per cut with cut_mean_pp.
  C. Extra (not registered): weighted draws at smaller sizes, with a uniform-draw control of the same size, to see whether the
     90% draw fraction of the registered cuts (1319 of 1473) limited how far the weights could move the mix.
Usage: python -I l15b_critic_indep.py --out DIR [--extra]
"""
import os, sys, json, argparse
from multiprocessing import Pool
import numpy as np, pandas as pd

R = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
ST = ["agreement", "n_distinct", "any_truncated"]
LAB = "proxy_error"; K0 = 50.0; FRAC = 0.10; NSUB = 20; SUB = 200; NSEED = 8; NSPLIT = 8
P0 = pd.read_parquet(f"{R}/p0/feats.parquet")
EV = pd.read_parquet(f"{R}/L4b/feats.parquet")


def table_from(df):
    g = df.assign(_y=df[LAB].to_numpy(int)).groupby(ST)["_y"].agg(["sum", "size"])
    prior = 1.0 - g.index.get_level_values("agreement").to_numpy(float)
    return pd.Series((g["sum"].to_numpy(float) + K0 * prior) / (g["size"].to_numpy(float) + K0), index=g.index)


def lookup(df, tab):
    post = tab.reindex(pd.MultiIndex.from_frame(df[ST])).to_numpy(float)
    return np.where(np.isnan(post), 1.0 - df["agreement"].to_numpy(float), post)


def sim_split(df, split):
    cnt = df["cell100"].value_counts()
    df = df[df["cell100"].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    groups = df["cell100"].unique()
    np.random.default_rng(split).shuffle(groups)
    fit = set(groups[: len(groups) // 4].tolist())
    return df[~df["cell100"].isin(fit)].reset_index(drop=True)


def gap_for_pool(pool):
    """Mean over subsets and splits of (frozen - fixed) hit rate in pp, plus mean fixed and frozen hit rate, plus spend check."""
    tabs = []
    for i in range(NSUB):
        idx = np.random.RandomState(100 + i).choice(len(P0), size=SUB, replace=False)
        tabs.append(table_from(P0.iloc[idx]))
    sims = [sim_split(pool, s) for s in range(NSPLIT)]
    gaps, fx, fz, bad = [], [], [], 0
    for tab in tabs:
        for sim in sims:
            score = lookup(sim, tab); ag = sim["agreement"].to_numpy(float); nd = sim["n_distinct"].to_numpy(float)
            y = sim[LAB].to_numpy(int); n = len(sim); b = int(FRAC * n)
            for s in range(NSEED):
                rnd = np.random.default_rng(s).random(n)
                a = np.lexsort((rnd, -nd, ag))[:b]; z = np.lexsort((rnd, -nd, ag, -score))[:b]
                if len(np.unique(a)) != b or len(np.unique(z)) != b: bad += 1
                fx.append(100.0 * y[a].sum() / b); fz.append(100.0 * y[z].sum() / b); gaps.append(fz[-1] - fx[-1])
    return float(np.mean(gaps)), float(np.mean(fx)), float(np.mean(fz)), bad


def weights(ev):
    code = ev.groupby(ST, sort=True).ngroup().to_numpy()
    nk = code.max() + 1
    is_test = ev["pid"].to_numpy() <= 1318
    ct = np.bincount(code[is_test], minlength=nk); cr = np.bincount(code[~is_test], minlength=nk)
    w = np.where((ct > 0) & (cr > 0), (ct / is_test.sum()) / (np.where(cr > 0, cr, 1) / (~is_test).sum()), 0.0)
    return code, is_test, ct, cr, w


def draw(ev, seed, size=None, weighted=True):
    code, is_test, ct, cr, w = weights(ev)
    elig = np.flatnonzero(~is_test & (w[code] > 0))
    size = int(is_test.sum()) if size is None else size
    p = w[code[elig]] if weighted else np.ones(len(elig))
    pick = np.random.RandomState(seed).choice(len(elig), size=size, replace=False, p=p / p.sum())
    return np.sort(elig[pick])


def tv_to_target(ev, chosen):
    code, is_test, ct, cr, w = weights(ev)
    keep = cr > 0
    tgt = np.where(keep, ct, 0).astype(float); tgt /= tgt.sum()
    d = np.bincount(code[chosen], minlength=len(ct)).astype(float); d /= d.sum()
    return float(0.5 * np.abs(d - tgt).sum())


def floor_tv(ev, size):
    code, is_test, ct, cr, w = weights(ev)
    keep = cr > 0
    tgt = np.where(keep, ct, 0).astype(float); tgt /= tgt.sum()
    return float(np.maximum(size * tgt - cr, 0).sum() / size)


def job_stored(c):
    d = json.load(open(f"{R}/L15b/cut_{c}/result.json"))
    pos = EV.index[EV["pid"].isin(d["pool_pids"])].to_numpy()
    own = draw(EV, c)
    same_draw = bool(np.array_equal(EV["pid"].to_numpy()[own], np.array(d["pool_pids"])))
    gap, fx, fz, bad = gap_for_pool(EV.iloc[own].reset_index(drop=True))
    return {"cut": c, "same_draw": same_draw, "own_gap": gap, "stored_gap": d["cut_mean_pp"], "own_fixed": fx, "own_frozen": fz,
            "stored_fixed": d["cut_fixed_pct_mean"], "stored_frozen": d["cut_frozen_pct_mean"], "spend_bad": bad,
            "tv": tv_to_target(EV, own), "err_rate": float(EV[LAB].to_numpy()[own].mean())}


def quotas(ct, cr, size):
    """Exact occupancy quotas: largest-remainder split of size by test share over strata with test and remaining docs, capped at the
    remaining count, leftover redistributed over uncapped strata."""
    keep = (ct > 0) & (cr > 0); q = np.zeros(len(ct), int); free = keep.copy(); left = size
    while left > 0 and free.any():
        sh = np.where(free, ct, 0).astype(float); sh /= sh.sum(); raw = sh * left; add = np.floor(raw).astype(int)
        rem = left - add.sum(); order = np.argsort(-(raw - add))
        add[order[:rem]] += 1
        q = q + add; over = np.where(q > cr, q - cr, 0); q = np.minimum(q, cr); left = int(over.sum()); free &= (q < cr)
    return q


def draw_strat(ev, seed, size, side):
    """side 'rem': stratified draw from remaining rows at exact test occupancy quotas. side 'test': uniform draw of test rows.
    side 'remuni': uniform draw of remaining rows."""
    code, is_test, ct, cr, w = weights(ev); rs = np.random.RandomState(seed)
    if side == "test": return np.sort(rs.choice(np.flatnonzero(is_test), size=size, replace=False))
    if side == "remuni": return np.sort(rs.choice(np.flatnonzero(~is_test), size=size, replace=False))
    q = quotas(ct, cr, size); out = []
    for j in np.flatnonzero(q > 0):
        out.append(rs.choice(np.flatnonzero((~is_test) & (code == j)), size=q[j], replace=False))
    return np.sort(np.concatenate(out))


def job_strat(args):
    size, c, side = args
    ch = draw_strat(EV, c, size, side)
    gap, fx, fz, bad = gap_for_pool(EV.iloc[ch].reset_index(drop=True))
    return {"size": size, "cut": c, "side": side, "gap": gap, "fixed": fx, "frozen": fz, "spend_bad": bad,
            "tv": tv_to_target(EV, ch), "err_rate": float(EV[LAB].to_numpy()[ch].mean()), "n": int(len(ch))}


def job_extra(args):
    size, c, weighted = args
    ch = draw(EV, c, size, weighted)
    gap, fx, fz, bad = gap_for_pool(EV.iloc[ch].reset_index(drop=True))
    return {"size": size, "cut": c, "weighted": weighted, "gap": gap, "fixed": fx, "frozen": fz, "spend_bad": bad,
            "tv": tv_to_target(EV, ch), "err_rate": float(EV[LAB].to_numpy()[ch].mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--extra", action="store_true"); ap.add_argument("--strat", action="store_true")
    ap.add_argument("--sizes", default="300,600,900"); ap.add_argument("--ncuts", type=int, default=24)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = {}
    with Pool(24) as p:
        if a.strat:
            sizes = [int(x) for x in a.sizes.split(",")]
            res = p.map(job_strat, [(s, c, side) for s in sizes for c in range(100, 100 + a.ncuts) for side in ("rem", "test", "remuni")])
            df = pd.DataFrame(res)
            t = df.groupby(["size", "side"]).agg(n=("n", "mean"), gap=("gap", "mean"), sd=("gap", "std"), mn=("gap", "min"), mx=("gap", "max"),
                                                 fixed=("fixed", "mean"), frozen=("frozen", "mean"), tv=("tv", "mean"),
                                                 err=("err_rate", "mean"), bad=("spend_bad", "sum"))
            print(t.to_string()); out["strat"] = res; out["strat_table"] = json.loads(t.reset_index().to_json(orient="records"))
            json.dump(out, open(os.path.join(a.out, "result_strat.json"), "w"), indent=1); sys.exit(0)
        if not a.extra:
            res = p.map(job_stored, list(range(100, 148)))
            out["stored"] = res
            g = np.array([r["own_gap"] for r in res])
            out["summary"] = {"all_draws_identical": all(r["same_draw"] for r in res), "own_mean": float(g.mean()),
                              "own_sd": float(g.std(ddof=1)), "own_min": float(g.min()), "own_max": float(g.max()),
                              "max_abs_diff_vs_stored": float(max(abs(r["own_gap"] - r["stored_gap"]) for r in res)),
                              "max_abs_diff_fixed": float(max(abs(r["own_fixed"] - r["stored_fixed"]) for r in res)),
                              "max_abs_diff_frozen": float(max(abs(r["own_frozen"] - r["stored_frozen"]) for r in res)),
                              "spend_bad": int(sum(r["spend_bad"] for r in res)),
                              "mean_fixed": float(np.mean([r["own_fixed"] for r in res])),
                              "mean_frozen": float(np.mean([r["own_frozen"] for r in res])),
                              "mean_tv": float(np.mean([r["tv"] for r in res])),
                              "mean_err_rate": float(np.mean([r["err_rate"] for r in res]))}
            print(json.dumps(out["summary"], indent=1))
        else:
            sizes = [int(x) for x in a.sizes.split(",")]
            for s in sizes: print("size", s, "TV floor", round(floor_tv(EV, s), 4), flush=True)
            jobs = [(s, c, w) for s in sizes for c in range(100, 100 + a.ncuts) for w in (True, False)]
            res = p.map(job_extra, jobs)
            out["extra"] = res
            df = pd.DataFrame(res)
            t = df.groupby(["size", "weighted"]).agg(gap=("gap", "mean"), sd=("gap", "std"), mn=("gap", "min"), mx=("gap", "max"),
                                                    fixed=("fixed", "mean"), frozen=("frozen", "mean"), tv=("tv", "mean"),
                                                    err=("err_rate", "mean"), bad=("spend_bad", "sum"))
            print(t.to_string())
            out["extra_table"] = json.loads(t.reset_index().to_json(orient="records"))
            out["floors"] = {str(s): floor_tv(EV, s) for s in sizes}
    json.dump(out, open(os.path.join(a.out, "result_extra.json" if a.extra else "result.json"), "w"), indent=1)

"""L5c critic replay (orchestrator check, CPU only).

Independent of avrl/exp/l5c_batch_learned_class.py: the pool, the batches, both arms and the first bootstrap resamples are
rebuilt here from avrl/runs/p0/feats.parquet. Only avrl/allocators.py (LearnedAllocator) is imported, read-only.

Checks against avrl/runs/L5c/result.json:
  - per-seed caught counts for both arms at 10, 25 and 50 percent (8 seeds each), and verified counts
  - the first N_CHECK group-bootstrap and bucket-cluster-bootstrap draws of D at 25 percent (allocator seed 0)
  - the 95 percent percentile CIs recomputed from all 200 stored draws of each bootstrap
Writes avrl/runs/L5c/critic_replay.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = "/homes/rdubash/avrl_live/CASCADE"
sys.path.insert(0, os.path.join(ROOT, "avrl"))
from allocators import LearnedAllocator  # read-only import of the shipped class

FEATS = os.path.join(ROOT, "avrl/runs/p0/feats.parquet")
STORED = os.path.join(ROOT, "avrl/runs/L5c/result.json")
OUT = os.path.join(ROOT, "avrl/runs/L5c/critic_replay.json")

CELL, LABEL = "cell100", "proxy_error"
COLS = ["agreement", "n_distinct", "mean_tokens", "std_tokens", "any_truncated", "log_qlen"] + [f"pc{i}" for i in range(16)]
BATCH, SEEDS, N_CHECK = 16, list(range(8)), 5
FRACS = {"0.1": (1, 10), "0.25": (1, 4), "0.5": (1, 2)}


def budget(num, den, n):
    """floor(f * n + 1/2) for f = num/den, in exact integer arithmetic."""
    return (2 * num * n + den) // (2 * den)


def build_sim(df):
    cnt = df[CELL].value_counts()
    df = df[df[CELL].isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    buckets = df[CELL].unique()
    np.random.default_rng(0).shuffle(buckets)
    fit = set(buckets[: len(buckets) // 4])
    return df[~df[CELL].isin(fit)].reset_index(drop=True)


def pool_arrays(d):
    return {"X": d[COLS].fillna(0).to_numpy(dtype=float), "y": d[LABEL].to_numpy().astype(int),
            "ag": d["agreement"].to_numpy(), "nd": d["n_distinct"].to_numpy(),
            "cells": pd.factorize(d[CELL])[0], "n": len(d)}


def batches(num, den, seed, n):
    """Consecutive batches of BATCH groups from a seeded permutation; yields (pool rows, b_i)."""
    perm = np.random.default_rng(seed).permutation(n)
    prev = 0
    for start in range(0, n, BATCH):
        rows = perm[start:start + BATCH]
        cum = budget(num, den, start + len(rows))
        yield rows, cum - prev
        prev = cum
    assert prev == budget(num, den, n), "schedule does not end at the exact budget"


def threshold_arm(A, num, den, seed):
    tie = np.random.default_rng(seed)
    hits, nver = 0, 0
    for rows, b in batches(num, den, seed, A["n"]):
        t = tie.random(len(rows))
        order = sorted(range(len(rows)), key=lambda j: (A["ag"][rows[j]], -A["nd"][rows[j]], t[j]))
        chosen = rows[np.array(order[:b], dtype=int)]
        hits += int(A["y"][chosen].sum())
        nver += len(chosen)
    return hits, nver


def learned_arm(A, num, den, seed):
    alloc = LearnedAllocator()
    sel = np.random.default_rng(seed)
    hits, nver = 0, 0
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        for rows, b in batches(num, den, seed, A["n"]):
            recs = [{"idx": int(r), "feat_vec": A["X"][r], "cell": int(A["cells"][r]),
                     "agreement": A["ag"][r], "n_distinct": A["nd"][r]} for r in rows]
            chosen = np.array([int(c) for c in alloc.select(recs, b, sel)], dtype=int)
            assert len(chosen) == b and np.isin(chosen, rows).all(), "selection outside batch or wrong count"
            hits += int(A["y"][chosen].sum())
            nver += len(chosen)
            alloc.update([(A["X"][c], int(A["y"][c]), int(A["cells"][c])) for c in chosen])
    nconv = sum(issubclass(w.category, ConvergenceWarning) for w in wlog)
    return hits, nver, nconv


def pct_ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    stored = json.load(open(STORED))
    sim = build_sim(pd.read_parquet(FEATS))
    A = pool_arrays(sim)
    total = int(A["y"].sum())
    out = {"n_sim": A["n"], "n_buckets": int(A["cells"].max()) + 1, "total_errors": total, "point": {}, "checks": {}}
    print(f"replay pool: n_sim {A['n']} buckets {out['n_buckets']} total_errors {total}", flush=True)

    all_ok, budgets_ok, conv_total = True, True, 0
    for key, (num, den) in FRACS.items():
        exp_total = budget(num, den, A["n"])
        out["point"][key] = {}
        res = {"threshold_batch": ([], []), "learned_batch": ([], [])}
        for s in SEEDS:
            h, v = threshold_arm(A, num, den, s)
            res["threshold_batch"][0].append(h)
            res["threshold_batch"][1].append(v)
            h, v, c = learned_arm(A, num, den, s)
            res["learned_batch"][0].append(h)
            res["learned_batch"][1].append(v)
            conv_total += c
        for arm, (caught, ver) in res.items():
            st = stored["point"][key][arm]["caught_per_seed"]
            match = [float(c) for c in caught] == st
            budgets_ok &= all(v == exp_total for v in ver)
            all_ok &= match
            out["point"][key][arm] = {"caught_per_seed": caught, "stored_per_seed": st, "match": match,
                                      "caught_mean": float(np.mean(caught)), "verified_all_exact": all(v == exp_total for v in ver),
                                      "admitted": 1 - float(np.mean(caught)) / total}
            print(f"{key:>5} {arm:<16} replay mean {np.mean(caught):7.2f} stored {stored['point'][key][arm]['caught_mean']:7.2f} "
                  f"per-seed match {match} verified exact {all(v == exp_total for v in ver)}", flush=True)

    # Bootstrap spot check at 25 percent: first N_CHECK resamples of each stored draw sequence, allocator seed 0.
    num, den = FRACS["0.25"]
    rng_g = np.random.default_rng(0)
    g_mine = []
    for _ in range(N_CHECK):
        idx = np.sort(rng_g.integers(0, A["n"], size=A["n"]))
        Ak = pool_arrays(sim.iloc[idx].reset_index(drop=True))
        h_t, _ = threshold_arm(Ak, num, den, 0)
        h_l, _, _ = learned_arm(Ak, num, den, 0)
        g_mine.append((h_l - h_t) / int(Ak["y"].sum()))
    rng_c = np.random.default_rng(123)
    n_b = int(A["cells"].max()) + 1
    c_mine = []
    for _ in range(N_CHECK):
        draws = rng_c.integers(0, n_b, size=n_b)
        pos, lab = [], []
        for j, b in enumerate(draws):
            m = np.where(A["cells"] == b)[0]
            pos.append(m)
            lab.append(np.full(len(m), j))
        pos, lab = np.concatenate(pos), np.concatenate(lab)
        o = np.argsort(pos, kind="stable")
        pos, lab = pos[o], lab[o]
        Ak = {"X": A["X"][pos], "y": A["y"][pos], "ag": A["ag"][pos], "nd": A["nd"][pos], "cells": lab, "n": len(pos)}
        h_t, _ = threshold_arm(Ak, num, den, 0)
        h_l, _, _ = learned_arm(Ak, num, den, 0)
        c_mine.append((h_l - h_t) / int(Ak["y"].sum()))

    g_st = stored["D_bootstrap_group"]["draws"][:N_CHECK]
    c_st = stored["D_bootstrap_cluster"]["draws"][:N_CHECK]
    g_diff = float(np.max(np.abs(np.array(g_mine) - np.array(g_st))))
    c_diff = float(np.max(np.abs(np.array(c_mine) - np.array(c_st))))
    out["bootstrap_spot_check"] = {"n_checked": N_CHECK, "group_max_abs_diff": g_diff, "cluster_max_abs_diff": c_diff,
                                   "group_replay": g_mine, "cluster_replay": c_mine}
    print(f"bootstrap spot check ({N_CHECK} draws each): group max abs diff {g_diff:.3e}, cluster max abs diff {c_diff:.3e}",
          flush=True)

    ci_ok = True
    for name in ["D_bootstrap_group", "D_bootstrap_cluster"]:
        d = np.array(stored[name]["draws"], float)
        ci = pct_ci(d)
        ci_ok &= np.allclose(ci, stored[name]["ci95"], atol=1e-12) and len(d) == 200 and bool(np.all(d < 0))
        out[f"{name}_recomputed"] = {"ci95": ci, "n": int(len(d)), "max_draw": float(d.max()), "min_draw": float(d.min()),
                                     "draws_above_0": int((d > 0).sum())}
        print(f"{name}: recomputed CI [{ci[0]:+.4f}, {ci[1]:+.4f}] stored {stored[name]['ci95']} "
              f"max draw {d.max():+.4f} draws above 0 {(d > 0).sum()}", flush=True)

    d_pt = (np.mean(out["point"]["0.25"]["learned_batch"]["caught_per_seed"])
            - np.mean(out["point"]["0.25"]["threshold_batch"]["caught_per_seed"])) / total
    out["checks"] = {"point_per_seed_match_all": bool(all_ok), "budgets_exact_all": bool(budgets_ok),
                     "replay_conv_warnings_point": int(conv_total), "bootstrap_spot_match": bool(g_diff < 1e-12 and c_diff < 1e-12),
                     "ci_recompute_match": bool(ci_ok), "D_point_25_replay": float(d_pt)}
    print("checks:", json.dumps(out["checks"]), flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

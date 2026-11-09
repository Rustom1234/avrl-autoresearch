"""L16a independent replay (critic). Imports no experiment module: numpy, pandas and the standard library only.

It re-derives from the raw feats files, without calling avrl/exp/l16a_source_split_null.py or the L13a helpers:
  the 20 frozen 200-label tables (RandomState(100 + i), k0 = 50, prior 1 - agreement, posterior keyed by the three strata),
  the 48 L2b cut positions (RandomState(100 + c).choice(3000, 1319, replace=False), sorted),
  the 200 bootstrap positions of the registered test rows (RandomState(16).randint(0, 1319, size=(200, 1319)), sorted),
  the cell100 split (groups with at least 3 rows, default_rng(split).shuffle, first quarter of the groups held out),
  the fixed and frozen picks (tie key default_rng(seed).random, budget int(0.10 x n_sim), 8 split seeds and 8 tie seeds),
  and the frozen-minus-fixed metric in pp (mean over tie seeds, then splits, then the 20 subsets).
It then checks the stored outputs in avrl/runs/L16a: gate.json (the two L13a references), the 48 cut means and positions, and the
200 bootstrap means and positions. Writes critic/replay.json under --out and exits 1 on any mismatch.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
K0, MIN_ROWS_PER_CELL, BUDGET, TOL_PP = 50.0, 3, 0.10, 1e-6
SPLITS, SEEDS = range(8), range(8)
N_SUB, SUB_SIZE, N_CUTS, CUT_SIZE, N_BOOT, BOOT_SEED = 20, 200, 48, 1319, 200, 16


def posterior_map(p0_rows):
    post = {}
    for key, grp in p0_rows.groupby(["agreement", "n_distinct", "any_truncated"]):
        ag, nd, tr = key
        prior = 1.0 - float(ag)
        post[(float(ag), float(nd), float(tr))] = (int(grp["proxy_error"].sum()) + K0 * prior) / (len(grp) + K0)
    return post


def score_rows(frame, post):
    ag = frame["agreement"].to_numpy(float); nd = frame["n_distinct"].to_numpy(float)
    tr = frame["any_truncated"].to_numpy(float)
    return np.array([post.get((ag[j], nd[j], tr[j]), 1.0 - ag[j]) for j in range(len(frame))], float)


def build_sim(frame, split):
    counts = frame["cell100"].value_counts()
    big = set(counts.index[counts.to_numpy() >= MIN_ROWS_PER_CELL].tolist())
    kept = frame[frame["cell100"].isin(big)].reset_index(drop=True)
    cells = kept["cell100"].unique()
    np.random.default_rng(split).shuffle(cells)
    held = set(cells[: len(cells) // 4].tolist())
    return kept[~kept["cell100"].isin(held)].reset_index(drop=True)


def subset_value(scored):
    """Mean over the 8 splits of the mean over the 8 tie seeds of 100 (frozen hits - fixed hits) / budget."""
    per_split = []
    for split in SPLITS:
        sim = build_sim(scored, split)
        y = sim["proxy_error"].to_numpy(int); ag = sim["agreement"].to_numpy(float)
        nd = sim["n_distinct"].to_numpy(float); fs = sim["frozen_score"].to_numpy(float)
        n = len(y); k = int(BUDGET * n)
        gaps = []
        for s in SEEDS:
            u = np.random.default_rng(s).random(n)
            pick_fixed = np.lexsort((u, -nd, ag))[:k]
            pick_frozen = np.lexsort((u, -nd, ag, -fs))[:k]
            assert len(np.unique(pick_fixed)) == k and len(np.unique(pick_frozen)) == k, "spend failure"
            gaps.append(100.0 * (float(y[pick_frozen].sum()) - float(y[pick_fixed].sum())) / k)
        per_split.append(float(np.mean(gaps)))
    return float(np.mean(per_split))


def frame_value(frame, tables):
    return float(np.mean([subset_value(frame.assign(frozen_score=score_rows(frame, post))) for post in tables]))


def check(name, mine, ref, tol, fails):
    d = abs(mine - ref)
    if not d <= tol:
        fails.append(f"{name}: replay {mine!r} vs stored {ref!r}")
    return {"replay": mine, "stored": ref, "abs_diff_pp": d}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    OUT = a.out
    p0 = pd.read_parquet(f"{RUNS}/p0/feats.parquet")
    l2b = pd.read_parquet(f"{RUNS}/L2b/feats.parquet")
    l4b = pd.read_parquet(f"{RUNS}/L4b/feats.parquet")
    test = l4b[l4b["pid"] <= 1318].reset_index(drop=True)
    tables = [posterior_map(p0.iloc[np.random.RandomState(100 + i).choice(len(p0), size=SUB_SIZE, replace=False)])
              for i in range(N_SUB)]
    p0_q = set(p0["question"])
    fails = []

    gate = json.load(open(f"{OUT}/gate.json"))
    ref_t = json.load(open(f"{RUNS}/L13a/L4b_pid_le1318/result.json"))["pools"]["L4b"]["frozen_vs_fixed_mean_pp"]
    ref_l = json.load(open(f"{RUNS}/L13a/L2b/result.json"))["pools"]["L2b"]["frozen_vs_fixed_mean_pp"]
    refs = {"test_rows": check("test rows vs L13a", frame_value(test, tables), float(ref_t), TOL_PP, fails),
            "l2b_full": check("L2b full vs L13a", frame_value(l2b, tables), float(ref_l), TOL_PP, fails)}
    assert int(test["question"].isin(p0_q).sum()) == 0 and int(l2b["question"].isin(p0_q).sum()) == 0

    cuts = []
    for c in range(N_CUTS):
        pos = np.sort(np.random.RandomState(100 + c).choice(len(l2b), size=CUT_SIZE, replace=False))
        frame = l2b.iloc[pos].reset_index(drop=True)
        assert int(frame["question"].isin(p0_q).sum()) == 0
        stored = json.load(open(f"{OUT}/cuts/cut_{c:02d}.json"))
        cuts.append({"cut": c, "positions_match": stored["positions"] == [int(x) for x in pos],
                     **check(f"cut {c:02d}", frame_value(frame, tables), stored["mean_pp"], TOL_PP, fails)})
        if not cuts[-1]["positions_match"]:
            fails.append(f"cut {c:02d}: positions differ")

    draws = np.random.RandomState(BOOT_SEED).randint(0, len(test), size=(N_BOOT, len(test)))
    boots = []
    for b in range(N_BOOT):
        pos = np.sort(draws[b])
        frame = test.iloc[pos].reset_index(drop=True)
        stored = json.load(open(f"{OUT}/boot/boot_{b:03d}.json"))
        boots.append({"draw": b, "positions_match": stored["positions"] == [int(x) for x in pos],
                      **check(f"boot {b:03d}", frame_value(frame, tables), stored["mean_pp"], TOL_PP, fails)})
        if not boots[-1]["positions_match"]:
            fails.append(f"boot {b:03d}: positions differ")

    cut_diffs = [c["abs_diff_pp"] for c in cuts]; boot_diffs = [b["abs_diff_pp"] for b in boots]
    max_diff = max(cut_diffs + boot_diffs + [refs["test_rows"]["abs_diff_pp"], refs["l2b_full"]["abs_diff_pp"]])
    replay_pass = len(fails) == 0
    out = {"experiment": "L16a", "independent_replay": True, "imports_experiment_modules": False, "pass": replay_pass,
           "tol_pp": TOL_PP, "max_abs_diff_pp": max_diff, "references": refs, "gate_json_pass": gate["pass"],
           "cuts_checked": len(cuts), "boot_draws_checked": len(boots),
           "cut_mean_pp_replay": [c["replay"] for c in cuts], "boot_mean_pp_replay": [b["replay"] for b in boots],
           "failures": fails[:20], "n_failures": len(fails)}
    os.makedirs(f"{OUT}/critic", exist_ok=True)
    with open(f"{OUT}/critic/replay.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"replay pass {replay_pass}: cuts {len(cuts)} boot {len(boots)} max abs diff {max_diff:.2e} pp; "
          f"test rows replay {refs['test_rows']['replay']:+.6f} stored {refs['test_rows']['stored']:+.6f}; "
          f"L2b replay {refs['l2b_full']['replay']:+.6f} stored {refs['l2b_full']['stored']:+.6f}; failures {len(fails)}", flush=True)
    for f in fails[:10]:
        print("FAIL", f, flush=True)
    return 0 if replay_pass else 1


if __name__ == "__main__":
    sys.exit(main())

"""L10b independent check (CPU only, prints to stdout, writes nothing). Recomputes from scratch, without importing the L10b module:

  1. Each pool's mean and sd (ddof 1) over the 20 subset means, and the count of subset means above +1.0 pp, from the per-split
     gaps in result.json (subset_detail), compared with the pool values in result.json.
  2. Per subset: the p0 subset rows (numpy RandomState(100 + i)) and the eval fold draw (RandomState(400 + i).randint(0, 5) per
     eval row) match result.json.
  3. The five fold tables are fitted on exactly the 160 p0 rows outside their fold, and the subset table on the 200 p0 rows.
     Each eval row is scored from the table of its drawn fold; every matched score lies in [0, 1] with no NaN.

Usage: python -I l10b_check.py --result OUTDIR/result.json
Exit 0 only if every check passes.
"""
import sys, json, argparse
import numpy as np, pandas as pd

RUNS = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"
P0 = f"{RUNS}/p0/feats.parquet"
STRATUM = ["agreement", "n_distinct", "any_truncated"]
LABEL = "proxy_error"; K0 = 50.0; SUBSET_SIZE = 200; SUBSETS = 20; PASS_PP = 1.0; TOL = 1e-9


def table(df):
    g = df.groupby(STRATUM)[LABEL].agg(["sum", "size"]).reset_index()
    prior = 1.0 - g["agreement"].to_numpy(float)
    g["post"] = (g["sum"].to_numpy(float) + K0 * prior) / (g["size"].to_numpy(float) + K0)
    return g


def score(df, t):
    m = df[STRATUM].merge(t[STRATUM + ["post"]], on=STRATUM, how="left")
    post = m["post"].to_numpy(float)
    return np.where(np.isnan(post), 1.0 - df["agreement"].to_numpy(float), post)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--result", required=True)
    a = ap.parse_args()
    R = json.load(open(a.result)); detail = R["subset_detail"]
    assert [x["subset"] for x in detail] == list(range(SUBSETS)), "subset_detail must hold subsets 0 to 19"
    p0 = pd.read_parquet(P0)
    results = []

    def check(name, ok, info=""):
        results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name} {info}".rstrip(), flush=True)

    for n in R["pools"]:
        # 1. pool statistics from the per-split gaps
        per_subset = np.asarray([np.mean(x[f"{n}_table_refit_minus_frozen_pp_per_split"]) for x in detail], float)
        n_splits = {len(x[f"{n}_table_refit_minus_frozen_pp_per_split"]) for x in detail}
        mean, sd, above = float(per_subset.mean()), float(per_subset.std(ddof=1)), int((per_subset > PASS_PP).sum())
        rp = R["pools"][n]
        check(f"{n} mean over subset means from per-split gaps", abs(mean - rp["mean_pp"]) < TOL,
              f"recomputed {mean:+.6f} vs result {rp['mean_pp']:+.6f} pp, verdict {'PASS' if mean > PASS_PP else 'KILL'}")
        check(f"{n} sd over subset means (ddof 1)", abs(sd - rp["sd_pp"]) < TOL, f"recomputed {sd:.6f} vs result {rp['sd_pp']:.6f}")
        check(f"{n} subsets above +1.0 pp", above == rp["n_subsets_above_pass"], f"recomputed {above} vs result {rp['n_subsets_above_pass']}")
        check(f"{n} eight split gaps per subset", n_splits == {8}, f"split counts {sorted(n_splits)}")

        # 2 and 3. draws, fold tables and eval scoring, recomputed from the feature files
        ev = pd.read_parquet(f"{RUNS}/{n}/feats.parquet"); n_eval = len(ev)
        n_equal_frozen, draws_ok, counts_ok, tables_ok, scores_ok = 0, True, True, True, True
        mad = []
        for i in range(SUBSETS):
            idx = np.random.RandomState(100 + i).choice(len(p0), size=SUBSET_SIZE, replace=False)
            draws_ok = draws_ok and idx.tolist() == detail[i]["row_positions"]
            sub = p0.iloc[idx]
            fold = np.empty(SUBSET_SIZE, int)
            fold[np.random.RandomState(300 + i).permutation(SUBSET_SIZE)] = np.arange(SUBSET_SIZE) % 5
            tabs = [table(sub[fold != f]) for f in range(5)]
            full = table(sub)
            tables_ok = tables_ok and all(int(t["size"].sum()) == 160 for t in tabs) and int(full["size"].sum()) == SUBSET_SIZE
            draw = np.random.RandomState(400 + i).randint(0, 5, size=n_eval)
            counts_ok = counts_ok and np.bincount(draw, minlength=5).tolist() == detail[i][f"{n}_eval_fold_counts"]
            matched = np.empty(n_eval)
            for f in range(5):
                mk = draw == f
                matched[mk] = score(ev[mk], tabs[f])
            scores_ok = scores_ok and bool(((matched >= 0) & (matched <= 1)).all()) and not np.isnan(matched).any()
            frozen = score(ev, full)
            n_equal_frozen += int(np.isclose(matched, frozen).sum())
            mad.append(float(np.abs(matched - frozen).mean()))
        check(f"{n} p0 subset draws (RandomState(100 + i)) match result.json", draws_ok, f"{SUBSETS} subsets")
        check(f"{n} eval fold counts (RandomState(400 + i)) match result.json", counts_ok, f"{n_eval} eval rows per subset")
        check(f"{n} fold tables fitted on 160 p0 rows, subset table on 200", tables_ok)
        check(f"{n} matched eval scores in [0, 1], no NaN", scores_ok)
        total = n_eval * SUBSETS
        print(f"INFO {n}: eval rows scored with the same value as the 200-label table in {100.0 * n_equal_frozen / total:.1f}% of "
              f"(subset, row) pairs; mean abs difference from the 200-label score {np.mean(mad):.5f} (informational)", flush=True)

    ok_all = all(results)
    print(f"CHECK {'PASS' if ok_all else 'FAIL'}: {sum(results)} of {len(results)} checks passed", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

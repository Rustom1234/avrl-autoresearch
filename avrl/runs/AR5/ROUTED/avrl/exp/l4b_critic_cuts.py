"""L4b secondary: 64 random 75-of-100 bucket cuts of the L4b pool at refit floor 40, configs A at 5% and B at 10%. CPU only.

Cut shape, tie-break seeds and gap definition as in part (a) of avrl/exp/l3a_critic_noise.py: drop a random quarter of the
cell100 buckets (default_rng(7000 + i)), 8 tie-break seeds, gap = config minus plain fixed in pp, paired per seed. Uses the
independent Pool class from l3a_critic_indep.py, as the noise script does. Reported beside the verdict, not part of it.

Usage: python -I l4b_critic_cuts.py --feats FEATS --result_dir RESULT_DIR
Writes <result_dir>/cuts64.json and adds the key secondary_64_cuts to <result_dir>/result.json.
"""
import os, sys, json, argparse
from multiprocessing import Pool as MP
import numpy as np, pandas as pd
sys.path.insert(0, "/homes/rdubash/avrl_live/ROUTED/avrl/exp")
from l3a_critic_indep import Pool


def load(feats):
    F = pd.read_parquet(feats); cnt = F.cell100.value_counts()
    return F[F.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)


def gaps(d, seeds):
    P = Pool(d.reset_index(drop=True)); b5, b10 = int(0.05 * P.n), int(0.10 * P.n); a40, b40 = [], []
    for s in seeds:
        f5 = P.y[P.fixed_set(b5, s)].sum(); f10 = P.y[P.fixed_set(b10, s)].sum()
        a40.append(100 * (P.y[P.recal_set(b5, s, 2, max(b5 // 8, 40), "x")[0]].sum() - f5) / b5)
        b40.append(100 * (P.y[P.recal_set(b10, s, 50, max(b10 // 8, 40), "xt")[0]].sum() - f10) / b10)
    return dict(A40=float(np.mean(a40)), B40=float(np.mean(b40)), n=P.n)


def cut(args):
    F, i = args; B = np.array(sorted(F.cell100.unique()))
    rng = np.random.default_rng(7000 + i); drop = set(rng.permutation(B)[: len(B) // 4].tolist())
    return gaps(F[~F.cell100.isin(drop)], range(8))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feats", required=True); ap.add_argument("--result_dir", required=True)
    a = ap.parse_args()
    F = load(a.feats)
    with MP(16) as mp:
        C = mp.map(cut, [(F, i) for i in range(64)])
    summary = {}
    for k in ("A40", "B40"):
        v = np.array([c[k] for c in C]); sd = float(v.std(ddof=1))
        summary[k] = {"n_cuts": len(v), "mean_pp": float(v.mean()), "sd_pp": sd, "min_pp": float(v.min()), "max_pp": float(v.max()),
                      "n_above_2pp": int((v > 2.0).sum()), "frac_above_2pp": float((v > 2.0).mean())}
        print(f"64 cuts {k}: mean {v.mean():+.2f} sd {sd:.2f} min {v.min():+.2f} max {v.max():+.2f} above +2.0 {(v > 2.0).sum()} of 64", flush=True)
    out = {"experiment": "L4b secondary", "feats": a.feats, "buckets_after_min3": int(F.cell100.nunique()), "docs_after_min3": int(len(F)),
           "note": "secondary, not part of the verdict", "summary": summary, "cuts": C}
    with open(os.path.join(a.result_dir, "cuts64.json"), "w") as fh: json.dump(out, fh, indent=1)
    rj = os.path.join(a.result_dir, "result.json")
    r = json.load(open(rj)); r["secondary_64_cuts"] = summary
    with open(rj, "w") as fh: json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()

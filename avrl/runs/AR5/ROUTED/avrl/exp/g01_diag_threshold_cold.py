"""G01 diagnostic, NOT part of the pre-registered verdict (CPU only).

The replay greedy_online (copied from runs/sim_generic.py) ranks groups randomly until its first refit, which happens after
R = max(budget // 8, 40) verified labels. allocators.GreedyOnlineAllocator instead ranks by the agreement threshold until its
first fit. This script changes only that cold start: before the first refit, groups are ranked by the fixed threshold
(agreement ascending, n_distinct descending, random tie-break). Everything else is the greedy branch of run_online.
It checks whether the G01 KILL survives the production cold start. K = 1, the reference configuration.

Usage (CPU only):
  /homes/rdubash/active-audit/.venv/bin/python -I g01_diag_threshold_cold.py --feats FEATS --cheap ... --out OUTDIR
"""
import os, sys, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir from sys.path
import g01_label_free_baselines as g01


class SimTC(g01.Sim):
    def greedy_threshold_cold(self, budget, seed, K=1):
        rng = np.random.default_rng(seed); R = max(budget // 8, 40); rnd = rng.random(self.n)
        rank = np.empty(self.n); rank[np.lexsort((rnd, -self.nd, self.ag))] = np.arange(self.n)
        score = -rank  # threshold order until the first refit
        seen = np.zeros(self.n, bool); hits = spent = 0; next_fit = R
        while spent < budget:
            take = min(K, budget - spent)
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True
            hits += int(self.y[c].sum()); spent += len(c)
            if spent >= next_fit and 0 < self.y[seen].sum() < seen.sum():
                score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(self.Xs[seen], self.y[seen]).predict_proba(self.Xs)[:, 1]
                next_fit += R
        return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--label", default="proxy_error")
    ap.add_argument("--group", default="cell100"); ap.add_argument("--cheap", required=True)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50"); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--main", required=True, help="G01 result.json (for the recal and replay greedy numbers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sim, Xs = g01.build_sim(a.feats, a.label, a.group, a.cheap.split(","))
    S = SimTC(sim, Xs, a.label, a.group)
    main_res = json.load(open(a.main))["by_k"]["1"]["rows"]
    seeds = list(range(a.seeds)); out = {"note": "diagnostic, not the pre-registered verdict", "rows": {}}
    for bf in [float(b) for b in a.budgets.split(",")]:
        b = int(bf * S.n); tc = [S.greedy_threshold_cold(b, s) for s in seeds]
        key = f"{bf:.2f}"; m = main_res[key]
        hr = lambda h: 100 * np.mean(h) / b
        sd = lambda h: 100 * np.std(h) / b
        gap = hr(tc) - m["recal_threshold"]["hit_rate_pct"]
        out["rows"][key] = {"budget": b, "greedy_threshold_cold_hits": [int(x) for x in tc],
                            "greedy_threshold_cold_hit_rate_pct": float(hr(tc)), "greedy_threshold_cold_sd_pp": float(sd(tc)),
                            "replay_greedy_hit_rate_pct": m["greedy_online"]["hit_rate_pct"],
                            "recal_hit_rate_pct": m["recal_threshold"]["hit_rate_pct"],
                            "gap_vs_recal_pp": float(gap), "max_seed_sd_pp": float(max(sd(tc), m["recal_threshold"]["hit_rate_sd_pp"])),
                            "gap_vs_fixed_pp": float(hr(tc) - m["fixed_agreement_threshold"]["hit_rate_pct"])}
        print(f"budget {key} calls {b}: threshold-cold greedy {hr(tc):.1f} ± {sd(tc):.1f} | replay greedy {m['greedy_online']['hit_rate_pct']:.1f} | "
              f"recal {m['recal_threshold']['hit_rate_pct']:.1f} | fixed {m['fixed_agreement_threshold']['hit_rate_pct']:.1f} | "
              f"gap vs recal {gap:+.2f} pp | gap vs fixed {hr(tc) - m['fixed_agreement_threshold']['hit_rate_pct']:+.2f} pp", flush=True)
    gaps = [r["gap_vs_recal_pp"] for r in out["rows"].values()]
    out["pass_rule_would_hold"] = bool(any(out["rows"][k]["gap_vs_recal_pp"] >= 5 and out["rows"][k]["gap_vs_recal_pp"] > 2 * out["rows"][k]["max_seed_sd_pp"] for k in ("0.10", "0.25")))
    out["kill_rule_would_hold"] = bool(all(g <= 3 for g in gaps))
    print(f"pass rule would hold: {out['pass_rule_would_hold']} | kill rule (all gaps <= 3 pp) would hold: {out['kill_rule_would_hold']}", flush=True)
    with open(os.path.join(a.out, "result.json"), "w") as fh: json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

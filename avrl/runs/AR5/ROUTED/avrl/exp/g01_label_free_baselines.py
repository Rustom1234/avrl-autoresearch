"""G01: label-free baselines on the E01 allocation replay (CPU only).

Rebuilds the sim split of runs/sim_generic.py: groups with >= MIN docs, rng seed 0 shuffle of buckets, first quarter of
buckets is the fit split and is dropped. The uniform, static, static_matched and oracle orders and the greedy branch of
run_online are copied from sim_generic.py. Two label-free baselines are added:
  fixed_agreement_threshold: agreement ascending, then n_distinct descending (allocators.ThresholdAllocator rule), no refit.
  recal_threshold: per (agreement, n_distinct) stratum, score = (errors + 2 * (1 - agreement)) / (verified + 2),
    recomputed every R verified labels from verified groups only. Initial score = 1 - agreement.
Ties in every ranking are broken by a per-seed random draw.

K = docs inspected per verifier visit. runs/sim_generic.py ships K = 4, but the E01 reference JSON
(avrl/runs/p0/sim_cell100_cheap.json) was written by avrl/runs/p0/sim_alloc.py, the same code with K = 1.
K changes only greedy_online and recal_threshold (the fixed-order policies have the same hits for any K), so the script
runs every K in --ks and the reproduction gate is evaluated at K = 1, the K that produced the reference.

Usage (CPU only):
  /homes/rdubash/active-audit/.venv/bin/python -I g01_label_free_baselines.py --feats FEATS --label proxy_error \
      --group cell100 --cheap agreement,n_distinct,mean_tokens,std_tokens,any_truncated,log_qlen \
      --budgets 0.05,0.10,0.25,0.50 --seeds 8 --ks 1,4 --out OUTDIR
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MIN = 3
REF = "/homes/rdubash/avrl_live/ROUTED/avrl/runs/p0/sim_cell100_cheap.json"
REF_KEYS = {0.10: "budget_0.1", 0.25: "budget_0.25", 0.50: "budget_0.5"}
POLICIES = ["uniform", "static", "static_matched", "fixed_agreement_threshold", "recal_threshold", "greedy_online", "oracle"]


def build_sim(path, label, G, cheap):
    """Sim split exactly as runs/sim_generic.py."""
    df = pd.read_parquet(path)
    cnt = df[G].value_counts(); df = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    rng0 = np.random.default_rng(0); groups = df[G].unique(); rng0.shuffle(groups)
    fit_g = set(groups[: len(groups) // 4]); fit = df[df[G].isin(fit_g)]; sim = df[~df[G].isin(fit_g)].reset_index(drop=True)
    X = lambda d: d[cheap].fillna(0).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X(fit), fit[label])
    Xs = X(sim); sim["static"] = clf.predict_proba(Xs)[:, 1]
    return sim, Xs


class Sim:
    def __init__(self, sim, Xs, label, G):
        self.y = sim[label].values.astype(int); self.Xs = Xs; self.n = len(self.y)
        self.n_buckets = int(sim[G].nunique())
        self.ag = sim["agreement"].values; self.nd = sim["n_distinct"].values; self.st = sim["static"].values
        self.s_codes = sim.groupby(["agreement", "n_distinct"]).ngroup().values  # (agreement, n_distinct) strata
        self.n_s = int(self.s_codes.max()) + 1
        s_ag = np.zeros(self.n_s); s_ag[self.s_codes] = self.ag
        self.s_prior = 1 - s_ag  # the fixed threshold's error rate, used as the Beta prior mean

    def _prefix(self, order, budget):
        """Fixed-order policies: visiting K at a time gives the same hits as taking the first budget entries."""
        return int(self.y[order[:budget]].sum())

    def uniform(self, budget, seed):
        return self._prefix(np.random.default_rng(seed).permutation(self.n), budget)

    def static(self, budget, seed):  # free-label classifier, reference only
        return self._prefix(np.argsort(-self.st), budget)

    def static_matched(self, budget, seed):
        """Copied from the static_matched branch of run() in runs/sim_generic.py."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n); tr = perm[: max(budget // 2, 1)]
        if self.y[tr].sum() == 0: order = perm
        else:
            sc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(self.Xs[tr], self.y[tr]).predict_proba(self.Xs)[:, 1]
            sc[tr] = -1; order = np.concatenate([tr, np.argsort(-sc)])
        return self._prefix(order, budget)

    def oracle(self, budget, seed):
        rng = np.random.default_rng(seed)
        return self._prefix(np.argsort(-(self.y + 1e-6 * rng.random(self.n))), budget)

    def fixed_agreement_threshold(self, budget, seed):
        rng = np.random.default_rng(seed)
        return self._prefix(np.lexsort((rng.random(self.n), -self.nd, self.ag)), budget)

    def recal_threshold(self, budget, seed, K):
        rng = np.random.default_rng(seed); R = max(budget // 8, 40); rnd = rng.random(self.n)
        seen = np.zeros(self.n, bool); table = self.s_prior.copy()  # no verified labels yet: table = prior
        hits = spent = 0; next_fit = R
        while spent < budget:
            take = min(K, budget - spent)
            order = np.lexsort((rnd, -self.nd, self.ag, -table[self.s_codes]))
            c = order[~seen[order]][:take]; seen[c] = True
            hits += int(self.y[c].sum()); spent += len(c)
            if spent >= next_fit:
                v = np.bincount(self.s_codes[seen], minlength=self.n_s)
                e = np.bincount(self.s_codes[seen], weights=self.y[seen], minlength=self.n_s)
                table = (e + 2 * self.s_prior) / (v + 2); next_fit += R
        return hits

    def greedy_online(self, budget, seed, K):
        """Greedy branch of run_online in runs/sim_generic.py: no layout, refit every R verified labels,
        top-scored unseen groups first. Before the first fit the score is random noise, as in the reference."""
        rng = np.random.default_rng(seed); R = max(budget // 8, 40)
        seen = np.zeros(self.n, bool); score = rng.random(self.n) * 1e-3
        hits = spent = 0; next_fit = R
        while spent < budget:
            take = min(K, budget - spent)
            mu = np.where(~seen)[0]; c = mu[np.argsort(-score[mu])][:take]; seen[c] = True
            hits += int(self.y[c].sum()); spent += len(c)
            if spent >= next_fit and 0 < self.y[seen].sum() < seen.sum():
                score = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(self.Xs[seen], self.y[seen]).predict_proba(self.Xs)[:, 1]
                next_fit += R
        return hits

    def run(self, policy, budget, seed, K):
        if policy == "uniform": return self.uniform(budget, seed)
        if policy == "static": return self.static(budget, seed)
        if policy == "static_matched": return self.static_matched(budget, seed)
        if policy == "fixed_agreement_threshold": return self.fixed_agreement_threshold(budget, seed)
        if policy == "recal_threshold": return self.recal_threshold(budget, seed, K)
        if policy == "greedy_online": return self.greedy_online(budget, seed, K)
        if policy == "oracle": return self.oracle(budget, seed)
        raise ValueError(policy)


def summarize(per, fracs, S):
    rows = {}
    for bf in fracs:
        b = int(bf * S.n); row = {}
        for p in POLICIES:
            h = np.array(per[bf][p], dtype=float)
            row[p] = {"budget": b, "mean_hits": float(h.mean()), "sd_hits": float(h.std()),
                      "hit_rate_pct": float(100 * h.mean() / b), "hit_rate_sd_pp": float(100 * h.std() / b),
                      "hits_per_seed": [int(x) for x in per[bf][p]]}
        g, r = row["greedy_online"], row["recal_threshold"]; f = row["fixed_agreement_threshold"]
        gap = g["hit_rate_pct"] - r["hit_rate_pct"]; sd_max = max(g["hit_rate_sd_pp"], r["hit_rate_sd_pp"])
        row["contrast"] = {"greedy_minus_recal_pp": gap, "max_seed_sd_pp": sd_max,
                           "greedy_minus_fixed_pp": g["hit_rate_pct"] - f["hit_rate_pct"],
                           "pass_at_budget": bool(gap >= 5 and gap > 2 * sd_max)}
        rows[f"{bf:.2f}"] = row
    return rows


def verdict(rows):
    """Pre-registered rule. PASS: gap >= 5 pp with gap > 2x max seed sd at 10% or 25%, and no gap in (3, 5) at any budget.
    KILL: greedy within 3 pp of recal or below it at all budgets. Anything else: INCONCLUSIVE."""
    gaps = {k: v["contrast"]["greedy_minus_recal_pp"] for k, v in rows.items()}
    pass_cond = any(rows[k]["contrast"]["pass_at_budget"] for k in ("0.10", "0.25") if k in rows)
    mid = any(3 < g < 5 for g in gaps.values())
    if pass_cond and not mid: return "PASS"
    if all(g <= 3 for g in gaps.values()): return "KILL"
    return "INCONCLUSIVE"


def gate(per, S, ref_path):
    """Reproduction gate against the E01 reference (K = 1): greedy_online and static_matched within 1 hit."""
    ref = json.load(open(ref_path)); out = {"ref": ref_path, "rows": {}, "pass": True}
    for bf, key in REF_KEYS.items():
        b = int(bf * S.n); row = {"budget": b}
        for p in ("greedy_online", "static_matched", "static", "oracle"):
            mine = float(np.mean(per[bf][p])); want = float(ref[key][p]["mean"]); diff = mine - want
            row[p] = {"mine": mine, "ref": want, "diff": diff, "gated": p in ("greedy_online", "static_matched")}
            if p in ("greedy_online", "static_matched") and abs(diff) > 1.0: out["pass"] = False
        out["rows"][f"{bf:.2f}"] = row
    return out


def md_tables(name, rows, fracs, S, verd):
    hr = ["uniform", "fixed_agreement_threshold", "recal_threshold", "greedy_online", "static_matched", "oracle", "static"]
    lines = [f"### {name}", "", f"Hit rate (%) = proxy errors caught / verifier calls, mean over seeds, with sd over seeds in pp.", "",
             "| budget | calls | " + " | ".join(hr) + " |", "|---|---|" + "---|" * len(hr)]
    for k, row in rows.items():
        lines.append(f"| {float(k):.0%} | {row['uniform']['budget']} | " + " | ".join(f"{row[p]['hit_rate_pct']:.1f} ± {row[p]['hit_rate_sd_pp']:.1f}" for p in hr) + " |")
    lines += ["", "Raw hits (mean ± sd over seeds):", "", "| budget | " + " | ".join(hr) + " |", "|---|" + "---|" * len(hr)]
    for k, row in rows.items():
        lines.append(f"| {float(k):.0%} | " + " | ".join(f"{row[p]['mean_hits']:.2f} ± {row[p]['sd_hits']:.2f}" for p in hr) + " |")
    lines += ["", "Contrasts (pp of hit rate):", "", "| budget | greedy minus recal | 2 x max seed sd | pass at budget | greedy minus fixed |", "|---|---|---|---|---|"]
    for k, row in rows.items():
        c = row["contrast"]
        lines.append(f"| {float(k):.0%} | {c['greedy_minus_recal_pp']:+.2f} | {2 * c['max_seed_sd_pp']:.2f} | {c['pass_at_budget']} | {c['greedy_minus_fixed_pp']:+.2f} |")
    lines += ["", f"Verdict ({name}): **{verd}**", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--label", default="proxy_error")
    ap.add_argument("--group", default="cell100"); ap.add_argument("--cheap", required=True)
    ap.add_argument("--budgets", default="0.05,0.10,0.25,0.50"); ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--ks", default="1,4"); ap.add_argument("--ref", default=REF); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sim, Xs = build_sim(a.feats, a.label, a.group, a.cheap.split(","))
    S = Sim(sim, Xs, a.label, a.group)
    fracs = [float(b) for b in a.budgets.split(",")]; ks = [int(k) for k in a.ks.split(",")]; seeds = list(range(a.seeds))
    print(f"sim docs {S.n} buckets {S.n_buckets} hits {int(S.y.sum())} base {S.y.mean():.4f} strata {S.n_s} budgets {fracs} K {ks} seeds {seeds}", flush=True)
    result = {"n_sim": S.n, "n_buckets": S.n_buckets, "hits_total": int(S.y.sum()), "seeds": seeds, "label": a.label,
              "group": a.group, "cheap": a.cheap.split(","), "budgets": fracs, "by_k": {}}
    tables = []
    for K in ks:
        per = {bf: {p: [S.run(p, int(bf * S.n), s, K) for s in seeds] for p in POLICIES} for bf in fracs}
        rows = summarize(per, fracs, S); verd = verdict(rows)
        block = {"K": K, "rows": rows, "verdict": verd}
        if K == 1:
            g = gate(per, S, a.ref); block["reproduction_gate"] = g
            block["verdict_valid"] = bool(g["pass"])
        result["by_k"][str(K)] = block
        name = f"K = {K}" + (" (matches E01 reference, gate applies)" if K == 1 else " (sim_generic.py as shipped, gate not applicable)")
        tables.append(md_tables(name, rows, fracs, S, verd))
        for k, row in rows.items():
            print(f"K={K} budget={k} greedy-recal={row['contrast']['greedy_minus_recal_pp']:+.2f}pp sdmax={row['contrast']['max_seed_sd_pp']:.2f} greedy-fixed={row['contrast']['greedy_minus_fixed_pp']:+.2f}pp", flush=True)
        print(f"K={K} verdict {verd}", flush=True)
    if 1 in ks:
        g = result["by_k"]["1"]["reproduction_gate"]
        print(f"reproduction gate (K=1, greedy_online and static_matched within 1 hit): {'PASS' if g['pass'] else 'FAIL'}", flush=True)
        for k, row in g["rows"].items():
            print(f"  budget {k}: " + " | ".join(f"{p} mine {row[p]['mine']:.3f} ref {row[p]['ref']:.3f} diff {row[p]['diff']:+.3f}" for p in ("greedy_online", "static_matched", "static", "oracle")), flush=True)
    head = f"# G01 label-free baselines\n\nSim split: {S.n} groups, {S.n_buckets} {a.group} buckets, {int(S.y.sum())} proxy errors. Seeds {seeds[0]} to {seeds[-1]}.\n"
    with open(os.path.join(a.out, "table.md"), "w") as fh: fh.write(head + "\n" + "\n".join(tables))
    with open(os.path.join(a.out, "result.json"), "w") as fh: json.dump(result, fh, indent=1)
    print(f"wrote {os.path.join(a.out, 'result.json')} and table.md", flush=True)


if __name__ == "__main__":
    main()

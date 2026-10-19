"""L3a critic: independent checks. CPU only. Does NOT import l1b_tuned_threshold or l3a_floor_sensitivity.
(1) Rebuild agreement, n_distinct, any_truncated, majority, proxy_error from rollouts.parquet (fresh pool) and compare to feats.parquet.
(2) Reimplement plain fixed threshold and the recal rule (own code) and replay configs A (5%) and B (10%) at floor 40 on
    both pools, splits 0 to 7, 8 tie-break seeds. Compare per-split gaps with runs/L3a/{fresh,l1b_pool}/result.json.
(3) Spend: count of verifier calls per policy equals int(f * n_sim).
(4) Zero-gap mechanism: for each fresh split and seed, does A's selected set equal fixed's? Count refits at floor 40 and 10.
(5) Gold leakage: ordering keys come only from agreement, n_distinct, any_truncated and labels of already-verified docs.
"""
import sys, json
import numpy as np, pandas as pd
ROOT = "/homes/rdubash/avrl_live/ROUTED/avrl/runs"


def build(df, split, G="cell100", MIN=3):
    cnt = df[G].value_counts(); d = df[df[G].isin(cnt[cnt >= MIN].index)].reset_index(drop=True)
    g = d[G].unique().copy(); np.random.default_rng(split).shuffle(g)
    drop = set(g[: len(g) // 4].tolist())
    return d[~d[G].isin(drop)].reset_index(drop=True)


class Pool:
    def __init__(self, d):
        self.n = len(d); self.y = d["proxy_error"].to_numpy().astype(int)
        self.ag = d["agreement"].to_numpy().astype(float); self.nd = d["n_distinct"].to_numpy().astype(int)
        self.tr = d["any_truncated"].to_numpy().astype(int)

    def strata(self, kind):
        cols = [self.ag, self.nd] + ([self.tr] if kind == "xt" else [])
        keys = list(zip(*cols)); ids = {}
        code = np.array([ids.setdefault(k, len(ids)) for k in keys])
        prior = np.zeros(len(ids)); prior[code] = 1.0 - self.ag
        return code, prior

    def fixed_set(self, B, seed):
        r = np.random.default_rng(seed).random(self.n)
        order = sorted(range(self.n), key=lambda i: (self.ag[i], -self.nd[i], r[i]))
        return order[:B]

    def recal_set(self, B, seed, k0, R, kind):
        code, prior = self.strata(kind); S = len(prior)
        r = np.random.default_rng(seed).random(self.n)
        table = prior.copy(); seen = np.zeros(self.n, bool); chosen = []; nref = 0
        while len(chosen) < B:
            cand = np.flatnonzero(~seen)
            o = np.lexsort((r[cand], -self.nd[cand], self.ag[cand], -table[code[cand]]))
            i = int(cand[o[0]]); seen[i] = True; chosen.append(i)
            if len(chosen) % R == 0:
                v = np.zeros(S); e = np.zeros(S)
                for j in np.flatnonzero(seen): v[code[j]] += 1; e[code[j]] += self.y[j]
                table = (e + k0 * prior) / (v + k0); nref += 1
        return chosen, nref


def run(feats_path, tag, floor=40, splits=range(8), seeds=range(8)):
    F = pd.read_parquet(feats_path); out = {"A": [], "B": [], "spend_bad": 0, "zero_equal": {}, "refits": {}, "A_hit": [], "fix5_hit": []}
    for s in splits:
        P = Pool(build(F, s)); b5, b10 = int(0.05 * P.n), int(0.10 * P.n)
        gA, gB, same, refs = [], [], 0, set()
        for sd in seeds:
            f5 = P.fixed_set(b5, sd); f10 = P.fixed_set(b10, sd)
            a, nr = P.recal_set(b5, sd, 2, max(b5 // 8, floor), "x"); refs.add(nr)
            b, _ = P.recal_set(b10, sd, 50, max(b10 // 8, floor), "xt")
            if not (len(f5) == len(a) == b5 and len(f10) == len(b) == b10 and len(set(a)) == b5 and len(set(b)) == b10): out["spend_bad"] += 1
            gA.append(100 * (P.y[a].sum() - P.y[f5].sum()) / b5); gB.append(100 * (P.y[b].sum() - P.y[f10].sum()) / b10)
            same += int(set(a) == set(f5))
            out["fix5_hit"].append(100 * P.y[f5].sum() / b5); out["A_hit"].append(100 * P.y[a].sum() / b5)
        out["A"].append(float(np.mean(gA))); out["B"].append(float(np.mean(gB)))
        out["zero_equal"][s] = same; out["refits"][s] = sorted(refs)
    return out


if __name__ == "__main__":
    # (1) feature rebuild on the fresh pool
    R = pd.read_parquet(f"{ROOT}/L2b/rollouts.parquet"); F = pd.read_parquet(f"{ROOT}/L2b/feats.parquet").set_index("pid")
    rows = {}
    for pid, g in R.groupby("pid"):
        vc = g["answer"].value_counts(); mx = vc.max(); maj_cands = sorted(vc[vc == mx].index)
        rows[pid] = dict(agreement=mx / len(g), n_distinct=len(vc), any_truncated=int(g["truncated"].max()), nroll=len(g), ntie=len(maj_cands),
                         majority=maj_cands[0])
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    J = F.loc[X.index]
    print("rollouts per prompt:", sorted(X.nroll.unique()), "prompts", len(X))
    print("agreement max abs diff", float(np.abs(X.agreement - J.agreement).max()), "n_distinct diff", int((X.n_distinct != J.n_distinct).sum()),
          "any_truncated diff", int((X.any_truncated != J.any_truncated).sum()))
    tie_ok = (X.majority == J.majority) | (X.ntie > 1)
    print("majority diff (excluding ties)", int((~tie_ok).sum()), "prompts with a majority tie", int((X.ntie > 1).sum()))
    pe = (J.majority != J.gold).astype(int)
    print("proxy_error == (majority != gold):", int((pe != J.proxy_error).sum()), "mismatches; proxy_error rate", float(J.proxy_error.mean()))
    print("agreement identical to a gold-based quantity? corr(agreement, pass_rate)", float(np.corrcoef(J.agreement, J.pass_rate)[0, 1]),
          "| fraction agreement == pass_rate", float((J.agreement == J.pass_rate).mean()))
    # (2)-(4)
    for name, path, res in (("fresh", f"{ROOT}/L2b/feats.parquet", f"{ROOT}/L3a/fresh/result.json"),
                            ("l1b_pool", f"{ROOT}/p0/feats.parquet", f"{ROOT}/L3a/l1b_pool/result.json")):
        o = run(path, name); ref = json.load(open(res))["configs"]
        dA = max(abs(a - b) for a, b in zip(o["A"], ref["A"]["gap_pp_per_split"])); dB = max(abs(a - b) for a, b in zip(o["B"], ref["B"]["gap_pp_per_split"]))
        print(f"[{name}] mine A {[round(x, 3) for x in o['A']]} mean {np.mean(o['A']):+.4f} sd {np.std(o['A'], ddof=1):.4f}")
        print(f"[{name}] mine B {[round(x, 3) for x in o['B']]} mean {np.mean(o['B']):+.4f} sd {np.std(o['B'], ddof=1):.4f}")
        print(f"[{name}] max abs diff vs result.json: A {dA:.2e} pp, B {dB:.2e} pp; spend/uniqueness failures {o['spend_bad']}")
        print(f"[{name}] seeds (of 8) where A's selected set == fixed's set, per split: {o['zero_equal']}; refits at 5%: {o['refits']}")
        print(f"[{name}] mean fixed 5% hit {np.mean(o['fix5_hit']):.2f}, mean A 5% hit {np.mean(o['A_hit']):.2f}")

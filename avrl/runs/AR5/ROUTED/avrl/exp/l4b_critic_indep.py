"""L4b critic: independent checks. CPU only. Does NOT import l1b_tuned_threshold, l4b_replay or l3a_critic_indep.
(1) Rebuild agreement, n_distinct, any_truncated, majority, proxy_error from rollouts.parquet and compare to feats.parquet;
    recompute gold from the GSM8K parquet files and compare to groups.parquet.
(2) Question-level disjointness of the L4b pool from the p0 (L1B) and L2b pools.
(3) Reimplement plain fixed and the recal rule (own code) and replay configs A (5%) and B (10%) at floors 40 and 10 on the L4b pool,
    splits 0 to 7, 8 tie-break seeds. Compare per-split gaps and hit rates with result.json. Spend and uniqueness per policy.
(4) Zero-gap mechanism: does A's selected set equal fixed's at floor 40, per split and seed? Refit counts.
(5) Headroom: how much of the 5% budget the fixed rule already fills, per pool.
(6) Bucket bootstrap for B at 10% and floor 40 (pool-level noise).
Usage: python -I l4b_critic_indep.py
"""
import json, re, sys
import numpy as np, pandas as pd

ROOT = "/homes/rdubash/avrl_live/ROUTED/avrl"
RUN = f"{ROOT}/runs"


def norm(x):
    try: return f"{float(x):.6g}"
    except Exception: return x


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
        ids = {}; code = np.array([ids.setdefault(k, len(ids)) for k in zip(*cols)])
        prior = np.zeros(len(ids)); prior[code] = 1.0 - self.ag
        return code, prior

    def fixed_set(self, B, seed):
        r = np.random.default_rng(seed).random(self.n)
        order = sorted(range(self.n), key=lambda i: (self.ag[i], -self.nd[i], r[i]))
        return order[:B]

    def recal_set(self, B, seed, k0, R, kind, labels_visible=None):
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


def replay(F, floor, splits=range(8), seeds=range(8)):
    out = dict(A=[], B=[], A_fix=[], A_cfg=[], B_fix=[], B_cfg=[], bad=0, same={}, refits={}, calls=[], n=[])
    for s in splits:
        P = Pool(build(F, s)); b5, b10 = int(0.05 * P.n), int(0.10 * P.n)
        gA, gB, hf5, ha, hf10, hb, same, refs = [], [], [], [], [], [], 0, set()
        for sd in seeds:
            f5 = P.fixed_set(b5, sd); f10 = P.fixed_set(b10, sd)
            a, nr = P.recal_set(b5, sd, 2, max(b5 // 8, floor), "x"); refs.add(nr)
            b, _ = P.recal_set(b10, sd, 50, max(b10 // 8, floor), "xt")
            if not (len(f5) == len(a) == b5 and len(f10) == len(b) == b10 and len(set(a)) == b5 and len(set(b)) == b10 and len(set(f5)) == b5 and len(set(f10)) == b10):
                out["bad"] += 1
            hf5.append(100 * P.y[f5].sum() / b5); ha.append(100 * P.y[a].sum() / b5)
            hf10.append(100 * P.y[f10].sum() / b10); hb.append(100 * P.y[b].sum() / b10)
            gA.append(ha[-1] - hf5[-1]); gB.append(hb[-1] - hf10[-1]); same += int(set(a) == set(f5))
        out["A"].append(float(np.mean(gA))); out["B"].append(float(np.mean(gB)))
        out["A_fix"].append(float(np.mean(hf5))); out["A_cfg"].append(float(np.mean(ha)))
        out["B_fix"].append(float(np.mean(hf10))); out["B_cfg"].append(float(np.mean(hb)))
        out["same"][s] = same; out["refits"][s] = sorted(refs); out["calls"].append((b5, b10)); out["n"].append(P.n)
    return out


def main():
    # (1) rebuild
    R = pd.read_parquet(f"{RUN}/L4b/rollouts.parquet"); G = pd.read_parquet(f"{RUN}/L4b/groups.parquet").set_index("pid")
    F = pd.read_parquet(f"{RUN}/L4b/feats.parquet")
    print("rollouts rows", len(R), "prompts", R.pid.nunique(), "rollouts per prompt", sorted(R.groupby("pid").size().unique()),
          "feats rows", len(F), "groups rows", len(G))
    rows = {}
    for pid, g in R.groupby("pid"):
        vc = g["answer"].value_counts(); mx = vc.max(); cands = sorted(vc[vc == mx].index)
        rows[pid] = dict(agreement=mx / len(g), n_distinct=len(vc), any_truncated=int(g["truncated"].max()), ntie=len(cands), majority=cands[0])
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index(); J = G.loc[X.index]
    print("agreement max abs diff", float(np.abs(X.agreement - J.agreement).max()), "| n_distinct diff", int((X.n_distinct != J.n_distinct).sum()),
          "| any_truncated diff", int((X.any_truncated != J.any_truncated).sum()), "| majority diff", int((X.majority != J.majority).sum()),
          "| prompts with majority tie", int((X.ntie > 1).sum()))
    print("proxy_error != (majority != gold):", int((((J.majority != J.gold).astype(int)) != J.proxy_error).sum()))
    Fi = F.set_index("pid").loc[X.index]
    for c in ("agreement", "n_distinct", "any_truncated", "proxy_error", "gold", "majority"):
        print(f"feats vs groups column {c}: mismatches", int((Fi[c].astype(str) != J[c].astype(str)).sum()))
    # truncation of the rollouts, tokens at the cap
    print("rollout truncated fraction %.4f" % R.truncated.mean(), "| rollouts with empty answer", int((R.answer == "").sum()),
          "| max tokens", int(R.tokens.max()))
    # gold recomputed from the dataset
    tr = pd.read_parquet(f"{ROOT}/data/gsm8k_train.parquet"); te = pd.read_parquet(f"{ROOT}/data/gsm8k_test.parquet")
    amap = {**dict(zip(tr.question, tr.answer)), **dict(zip(te.question, te.answer))}
    gold2 = G.question.map(lambda q: norm(amap[q].split("####")[-1].strip().replace(",", "")))
    print("gold recomputed from dataset vs groups.gold mismatches:", int((gold2 != G.gold).sum()), "| questions not in dataset", int((~G.question.isin(amap)).sum()))
    print("is test row for pids < 1319:", bool(G.question.iloc[:1319].isin(set(te.question)).all()), "| pids >= 1319 are train:", bool(G.question.iloc[1319:].isin(set(tr.question)).all()))
    # (2) disjointness
    q0 = set(pd.read_parquet(f"{RUN}/p0/groups.parquet").question); q2 = set(pd.read_parquet(f"{RUN}/L2b/groups.parquet").question); q4 = set(G.question)
    print("pool sizes p0/L2b/L4b", len(q0), len(q2), len(q4), "| overlaps p0&L4b", len(q0 & q4), "L2b&L4b", len(q2 & q4), "p0&L2b", len(q0 & q2), "| L4b unique", G.question.is_unique)
    # (3) replay
    res = json.load(open(f"{RUN}/L4b/result.json"))["configs"]
    for floor, ka, kb in ((40, "A40", "B40"), (10, "A10", "B10")):
        o = replay(F, floor)
        dA = max(abs(a - b) for a, b in zip(o["A"], res[ka]["gap_pp_per_split"]))
        dB = max(abs(a - b) for a, b in zip(o["B"], res[kb]["gap_pp_per_split"]))
        dfa = max(abs(a - b) for a, b in zip(o["A_fix"], res[ka]["fixed_hit_pct_per_split"])); dca = max(abs(a - b) for a, b in zip(o["A_cfg"], res[ka]["config_hit_pct_per_split"]))
        dfb = max(abs(a - b) for a, b in zip(o["B_fix"], res[kb]["fixed_hit_pct_per_split"])); dcb = max(abs(a - b) for a, b in zip(o["B_cfg"], res[kb]["config_hit_pct_per_split"]))
        print(f"[floor {floor}] mine A gaps {[round(x, 3) for x in o['A']]} mean {np.mean(o['A']):+.4f} sd {np.std(o['A'], ddof=1):.4f}")
        print(f"[floor {floor}] mine B gaps {[round(x, 3) for x in o['B']]} mean {np.mean(o['B']):+.4f} sd {np.std(o['B'], ddof=1):.4f}")
        print(f"[floor {floor}] max abs diff vs result.json: A gap {dA:.2e}, B gap {dB:.2e}, A fixed hit {dfa:.2e}, A cfg hit {dca:.2e}, B fixed hit {dfb:.2e}, B cfg hit {dcb:.2e}; spend/uniqueness failures {o['bad']}")
        print(f"[floor {floor}] mean hit rates: fixed5 {np.mean(o['A_fix']):.2f} A {np.mean(o['A_cfg']):.2f} fixed10 {np.mean(o['B_fix']):.2f} B {np.mean(o['B_cfg']):.2f}")
        print(f"[floor {floor}] calls (5%,10%) per split {o['calls']} n_sim {o['n']}")
        print(f"[floor {floor}] seeds (of 8) where A's set == fixed's set, per split: {o['same']}; refits at 5%: {o['refits']}")
    # (5) headroom per pool
    for name in ("p0", "L2b", "L4b"):
        D = pd.read_parquet(f"{RUN}/{name}/feats.parquet")
        o = replay(D, 40, splits=range(2), seeds=range(2))
        print(f"[headroom {name}] proxy_error rate {D.proxy_error.mean():.4f}; docs with agreement <= 0.25: {int((D.agreement <= 0.25).sum())} of {len(D)}; "
              f"fixed 5% hit on splits 0,1: {np.mean(o['A_fix']):.2f}; headroom to 100: {100 - np.mean(o['A_fix']):.2f} pp")
    # (6) bucket bootstrap for B at 10% floor 40, and A at 5% floor 40
    rng = np.random.default_rng(12345); cnt = F.cell100.value_counts(); Fm = F[F.cell100.isin(cnt[cnt >= 3].index)].reset_index(drop=True)
    buckets = np.array(sorted(Fm.cell100.unique())); byb = {b: Fm[Fm.cell100 == b] for b in buckets}
    resB, resA = [], []
    for it in range(100):
        keep = rng.choice(buckets, size=int(0.75 * len(buckets)), replace=True)
        d = pd.concat([byb[b] for b in keep], ignore_index=True); P = Pool(d); b5, b10 = int(0.05 * P.n), int(0.10 * P.n); gb, ga = [], []
        for sd in range(3):
            gb.append(100 * (P.y[P.recal_set(b10, sd, 50, max(b10 // 8, 40), "xt")[0]].sum() - P.y[P.fixed_set(b10, sd)].sum()) / b10)
            ga.append(100 * (P.y[P.recal_set(b5, sd, 2, max(b5 // 8, 40), "x")[0]].sum() - P.y[P.fixed_set(b5, sd)].sum()) / b5)
        resB.append(np.mean(gb)); resA.append(np.mean(ga))
    resB = np.array(resB); resA = np.array(resA)
    print(f"[bootstrap 100 x 75-of-100 buckets with replacement, 3 seeds] B 10% floor 40: mean {resB.mean():+.2f} sd {resB.std(ddof=1):.2f} 2.5-97.5% {np.percentile(resB, 2.5):+.2f} to {np.percentile(resB, 97.5):+.2f}, above +2.0: {(resB > 2).sum()} of 100")
    print(f"[bootstrap] A 5% floor 40: mean {resA.mean():+.2f} sd {resA.std(ddof=1):.2f} 2.5-97.5% {np.percentile(resA, 2.5):+.2f} to {np.percentile(resA, 97.5):+.2f}, above +2.0: {(resA > 2).sum()} of 100")


if __name__ == "__main__":
    main()

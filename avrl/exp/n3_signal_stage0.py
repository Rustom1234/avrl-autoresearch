"""N3 Stage 0: why does "cheap passes agree" route well for the RL verifier and badly for the ledger critic?

CPU only, no model calls. Reads avrl/runs/{p0,E06,E10p0} and avrl/runs/AR3 and computes, per stratum
(role x regime or plant kind): pass accuracy a, within-item error ICC, wrong-output collision c_w, right-output
collision c_r, AUROC and recall at 25 percent escalation of the signals AGREE, VERB (critic only) and DIFF
(a difficulty-only control), with a cluster bootstrap. Then the tests T1 (collision vs ICC), T2 (reversal across
roles) and T3 (power gate) exactly as frozen in avrl/runs/N3/PREREG.md (sha256 in PREREG.sha256).

  .venv/bin/python avrl/exp/n3_signal_stage0.py [--B 10000] [--check-only]
Writes avrl/runs/N3/results.json and avrl/runs/N3/tables.md. Unit tests: avrl/exp/n3_tests/.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # avrl/
sys.path.insert(0, str(HERE))
import ar3_features as F  # noqa: E402
import ar3_stats as S  # noqa: E402

RUNS = ROOT / "runs"
OUT = RUNS / "N3"
SEED = 20261105
FRAC = 0.25
MIN_ITEMS, MIN_WRONG, MIN_RIGHT = 40, 10, 10
MIN_STRATA_T1 = 6
T1_MAE, T1_GAP, T1_KILL_GAP = 0.06, 0.20, 0.10
T3_HALF = 0.15
Q_LO, Q_HI = 5.0, 95.0  # 90 percent percentile interval

# ------------------------------------------------------------------ batch statistics (weights W: B x n_items)


def _tie_groups(s):
    """For a sorted array s: start and end index (inclusive) of the run of equal values around each position."""
    n = len(s)
    new = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, n - 1]
    return starts[gid], ends[gid]


def _cum0(a):
    return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)


def auroc_batch(score, y, W):
    """AUROC of `score` for y == 1 (higher score = more suspect) under each weight row of W (B x n). Ties count half."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    order = np.argsort(score, kind="stable")
    s, yy, Ws = score[order], y[order], W[:, order]
    gs, ge = _tie_groups(s)
    neg, pos = Ws * (1 - yy), Ws * yy
    c = _cum0(neg)
    below, upto = c[:, gs], c[:, ge + 1]
    num = (pos * (below + 0.5 * (upto - below))).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def recall_batch(score, y, W, frac=FRAC):
    """Share of y == 1 weight escalated when the top `frac` of the weight (highest score first) is escalated.
    Ties at the boundary are included fractionally."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    order = np.argsort(-score, kind="stable")
    s, yy, Ws = score[order], y[order], W[:, order]
    gs, ge = _tie_groups(s)
    c = _cum0(Ws)
    before = c[:, gs]
    gw = c[:, ge + 1] - before
    budget = frac * Ws.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(gw > 0, np.clip((budget - before) / gw, 0.0, 1.0), 0.0)
    num = (Ws * yy * f).sum(axis=1)
    den = (Ws * yy).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def icc_batch(m, n, W):
    """One-way ANOVA ICC(1) of 0/1 pass-error indicators. m: errors per item out of n passes; W: B x n_items weights."""
    m = np.asarray(m, float)
    N = W.sum(axis=1)
    pbar = (W * m).sum(axis=1) / (n * N)
    ssb = (W * (m[None, :] - n * pbar[:, None]) ** 2).sum(axis=1) / n
    ssw = (W * (m * (n - m))[None, :]).sum(axis=1) / n
    msb, msw = ssb / (N - 1), ssw / (N * (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (msb - msw) / (msb + (n - 1) * msw)


def ratio_batch(num, den, W):
    """sum_i W_i num_i / sum_i W_i den_i for each weight row (nan if the denominator is 0)."""
    a, b = W @ np.asarray(num, float), W @ np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


# ------------------------------------------------------------------ pass level -> item level


def pair_stats(wrong, same):
    """Per-item counts of pass pairs. wrong: (n, P) bool; same: (n, P*(P-1)/2) bool in itertools.combinations order.
    Returns ww, sww, rr, srr: pairs both wrong, both wrong with the same output, both right, both right with the same output."""
    wrong = np.asarray(wrong, bool)
    pairs = list(combinations(range(wrong.shape[1]), 2))
    bw = np.stack([wrong[:, a] & wrong[:, b] for a, b in pairs], axis=1)
    br = np.stack([~wrong[:, a] & ~wrong[:, b] for a, b in pairs], axis=1)
    same = np.asarray(same, bool)
    return bw.sum(1), (bw & same).sum(1), br.sum(1), (br & same).sum(1)


def plurality_correct(ids, correct):
    """Cheap decision on a few passes: plurality output class (ties to the earliest pass); correct iff its members' mean
    `correct` rounds to 1 (the rule of code_rewards.group_signature_stats). Returns 0 or 1."""
    top = Counter(ids).most_common(1)[0][0]
    mem = [c for i, c in zip(ids, correct) if i == top]
    return int(round(sum(mem) / len(mem)))


def merge_kinds(counts, min_items=MIN_ITEMS):
    """kind -> stratum name for the plant kinds. A family (N or V) with any kind under min_items is merged into one stratum."""
    fams = {k[0] for k in counts if counts[k] < min_items}
    return {k: (k[0] if k[0] in fams else k) for k in counts}


@dataclass
class Stratum:
    name: str
    role: str
    n_pass: int
    m: np.ndarray  # wrong passes per item
    ww: np.ndarray
    sww: np.ndarray
    rr: np.ndarray
    srr: np.ndarray
    y: np.ndarray  # 1 if the cheap decision is wrong
    cluster: np.ndarray  # int cluster id per item
    scores: dict = field(default_factory=dict)  # signal -> suspicion score per item
    extra: dict = field(default_factory=dict)
    weight: object = None  # optional fixed per-item weight (used only by the exploratory design-weighted row)

    @property
    def n_items(self):
        return len(self.y)

    @property
    def n_clusters(self):
        return int(self.cluster.max()) + 1


def stratum_stats(st, W):
    """All per-stratum statistics under weight rows W (B x n_items). Returns {name: array(B)}."""
    if st.weight is not None:
        W = W * st.weight[None, :]
    o = {"a": 1.0 - (W @ st.m) / (st.n_pass * W.sum(axis=1)),
         "icc": icc_batch(st.m, st.n_pass, W),
         "c_w": ratio_batch(st.sww, st.ww, W),
         "c_r": ratio_batch(st.srr, st.rr, W),
         "wrong_dec": (W @ st.y) / W.sum(axis=1)}
    for nm, sc in st.scores.items():
        o[f"auc_{nm}"] = auroc_batch(sc, st.y, W)
        o[f"rec_{nm}"] = recall_batch(sc, st.y, W)
    return o


def point_stats(st):
    return {k: float(v[0]) for k, v in stratum_stats(st, np.ones((1, st.n_items))).items()}


def bootstrap_stratum(st, B, seed, chunk=250):
    """Cluster bootstrap: {name: array(B)} of every statistic. Each draw resamples clusters with replacement."""
    rng = np.random.default_rng(seed)
    C = st.n_clusters
    p = np.full(C, 1.0 / C)
    out = {}
    for lo in range(0, B, chunk):
        nb = min(chunk, B - lo)
        W = rng.multinomial(C, p, size=nb)[:, st.cluster].astype(float)
        for k, v in stratum_stats(st, W).items():
            out.setdefault(k, np.full(B, np.nan))[lo:lo + nb] = v
    return out


def ci(x, lo=Q_LO, hi=Q_HI):
    x = np.asarray(x, float)
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


# ------------------------------------------------------------------ strata kept or dropped


def keep_stratum(st):
    """(kept, reason). Floors: >= 40 items, >= 10 wrong and >= 10 correct cheap decisions."""
    nw = int(st.y.sum())
    nr = st.n_items - nw
    if st.n_items < MIN_ITEMS:
        return False, f"{st.n_items} items < {MIN_ITEMS}"
    if nw < MIN_WRONG:
        return False, f"{nw} wrong decisions < {MIN_WRONG}"
    if nr < MIN_RIGHT:
        return False, f"{nr} correct decisions < {MIN_RIGHT}"
    return True, ""


# ------------------------------------------------------------------ T1, T2, T3


def loso_predict(X, y):
    """Leave-one-stratum-out OLS predictions with an intercept. X: (S, p), y: (S,)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), X])
    pred = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        pred[i] = A[i] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    return pred


def loso_scores(X, y):
    """(MAE, R2) of leave-one-out predictions; R2 = 1 - SSE / SST with SST around the mean of y."""
    y = np.asarray(y, float)
    err = loso_predict(X, y) - y
    sst = float(((y - y.mean()) ** 2).sum())
    return float(np.abs(err).mean()), (1.0 - float((err ** 2).sum()) / sst) if sst > 0 else float("nan")


def t1_models(a, c_r, c_w, icc, auc):
    """Both models on one set of per-stratum values. Returns dict with MAE and R2 for each and the R2 gap."""
    mae_c, r2_c = loso_scores(np.column_stack([a, c_r, c_w]), auc)
    mae_i, r2_i = loso_scores(np.column_stack([a, icc]), auc)
    return {"mae_coll": mae_c, "r2_coll": r2_c, "mae_icc": mae_i, "r2_icc": r2_i, "gap": r2_c - r2_i}


def t1_verdict(m, n_strata):
    """PASS / KILL / INCONCLUSIVE from the frozen rule."""
    if n_strata < MIN_STRATA_T1:
        return "INCONCLUSIVE"
    if m["mae_coll"] <= T1_MAE and m["gap"] >= T1_GAP:
        return "PASS"
    if m["r2_icc"] >= m["r2_coll"] - T1_KILL_GAP:
        return "KILL"
    return "INCONCLUSIVE"


def t3_verdict(half_widths):
    """UNPOWERED iff more than half of the strata have a 90 percent AUROC interval half-width above 0.15."""
    hw = np.asarray(half_widths, float)
    n_bad = int((hw > T3_HALF).sum())
    return ("UNPOWERED" if n_bad > len(hw) / 2 else "POWERED"), n_bad


def role_signals(point, boot, names, sigs):
    """Macro average over strata `names` of recall for each signal: (point dict, boot dict of arrays)."""
    return ({s: float(np.mean([point[n][f"rec_{s}"] for n in names])) for s in sigs},
            {s: np.mean([boot[n][f"rec_{s}"] for n in names], axis=0) for s in sigs})


def t2_verdict(pt_v, bt_v, pt_c, bt_c):
    """Frozen T2 rule. pt_*: {signal: macro recall}; bt_*: {signal: array(B)}. Verifier role has AGREE and DIFF; critic has more.
    PASS iff AGREE beats DIFF in the verifier (90 percent paired interval above 0), the critic's best signal is not AGREE and beats AGREE
    (90 percent paired interval above 0), and the best signals differ between the roles."""
    best_v = max(pt_v, key=pt_v.get)
    best_c = max(pt_c, key=pt_c.get)
    d_v = bt_v["AGREE"] - bt_v["DIFF"]
    d_c = bt_c[best_c] - bt_c["AGREE"] if best_c != "AGREE" else np.zeros(len(next(iter(bt_c.values()))))
    lo_v, hi_v = ci(d_v)
    lo_c, hi_c = ci(d_c)
    ok_v = best_v == "AGREE" and lo_v > 0
    ok_c = best_c != "AGREE" and lo_c > 0
    return {"verdict": "PASS" if (ok_v and ok_c and best_v != best_c) else "FAIL",
            "best_verifier": best_v, "best_critic": best_c,
            "verifier_agree_minus_diff": {"point": pt_v["AGREE"] - pt_v["DIFF"], "ci90": [lo_v, hi_v]},
            "critic_best_minus_agree": {"point": pt_c[best_c] - pt_c["AGREE"], "ci90": [lo_c, hi_c]}}


def final_verdict(t1, t2, t3):
    """PROMOTE iff T1 PASS and T2 PASS; otherwise STOP. UNPOWERED T3 limits a promotion to enlarging n."""
    if t1 == "PASS" and t2 == "PASS":
        return "PROMOTE (to enlarge n only, no mechanism claim)" if t3 == "UNPOWERED" else "PROMOTE"
    return "STOP"


# ------------------------------------------------------------------ loaders


def verify_prereg():
    want = (OUT / "PREREG.sha256").read_text().split()[0]
    got = hashlib.sha256((OUT / "PREREG.md").read_bytes()).hexdigest()
    if want != got:
        raise SystemExit(f"PREREG.md sha256 {got[:12]} != PREREG.sha256 {want[:12]}: refusing to run")
    return got


def _ids(a):
    """Integer id per distinct output string (NaN and None are real values here, not missing)."""
    return pd.factorize(pd.Series(a).astype(str).to_numpy(), use_na_sentinel=False)[0]


def load_verifier(name, run, kind):
    """One verifier stratum from rollouts.parquet + groups.parquet. kind 'answer' (GSM8K) or 'sig' (code)."""
    d = RUNS / run
    R, G = pd.read_parquet(d / "rollouts.parquet"), pd.read_parquet(d / "groups.parquet")
    key = "pid" if kind == "answer" else "task_id"
    R = R.sort_values([key, "k"]).reset_index(drop=True)
    G = G.sort_values(key).reset_index(drop=True)
    P = int(R.groupby(key).size().iloc[0])
    assert (R.groupby(key).size() == P).all() and len(R) == len(G) * P, f"{run}: unbalanced rollouts"
    assert (R[key].to_numpy()[::P] == G[key].to_numpy()).all()
    out = R["answer"].astype(str).to_numpy() if kind == "answer" else np.array(["\x1f".join(map(str, s)) for s in R["sig"]])
    ids = _ids(out).reshape(len(G), P)
    correct = R["correct"].to_numpy().reshape(len(G), P).astype(int)
    wrong = correct == 0
    pairs = list(combinations(range(P), 2))
    same = np.stack([ids[:, a] == ids[:, b] for a, b in pairs], axis=1)
    ww, sww, rr, srr = pair_stats(wrong, same)
    dec = np.array([plurality_correct(ids[i, :3].tolist(), correct[i, :3].tolist()) for i in range(len(G))])
    agree3 = np.mean([ids[:, a] == ids[:, b] for a, b in combinations(range(3), 2)], axis=0)
    scores = {"AGREE": -agree3}
    if "q_len" in G:
        scores["DIFF"] = _signed_diff(G["q_len"].to_numpy(float), 1 - dec)
    agree_all = np.array([Counter(ids[i]).most_common(1)[0][1] / P for i in range(len(G))])
    extra = {"native_k": P, "native_agree_matches_groups": bool(np.allclose(agree_all, G["agreement"].to_numpy())),
             "native_auc_agreement": float(S.weighted_auroc(G["proxy_error"].to_numpy(), -G["agreement"].to_numpy())),
             "native_proxy_error": float(G["proxy_error"].mean()),
             "plurality8_vs_groups_majority_correct_agree": float(np.mean(
                 [plurality_correct(ids[i].tolist(), correct[i].tolist()) == G["majority_correct"].iloc[i] for i in range(len(G))]))}
    return Stratum(name, "verifier", P, wrong.sum(1), ww, sww, rr, srr, (1 - dec).astype(int), np.arange(len(G)), scores, extra)


def _signed_diff(x, y):
    """Difficulty control: +x if its point AUROC for y is at least 0.5, else -x. The sign is then fixed."""
    a = float(auroc_batch(x, y, np.ones((1, len(y))))[0])
    return x if a >= 0.5 else -x


def load_critic():
    """AR3 strata (burn + test, plants and clean controls) from the raw Haiku calls. Returns (list of Stratum, extras)."""
    run = RUNS / "AR3"
    items = F.load_items(run)
    pool = [it for it in items if it["split"] in ("burn", "test") and (it["truth"] == "clean" or it["kind"] in F.PLANT_KINDS)]
    calls = F.load_calls(run, ["burn", "test"], {it["item_id"] for it in pool})
    counts = Counter(it["kind"] for it in pool if it["kind"] in F.PLANT_KINDS)
    kmap = merge_kinds(counts)
    kmap["control"] = "clean"
    entries = {e: i for i, e in enumerate(sorted({it["entry_key"] for it in pool}))}
    by = {}
    for it in pool:
        by.setdefault(kmap[it["kind"]], []).append(it)
    pairs5 = list(combinations(range(5), 2))
    strata, chk = [], {"conf_agree_maxdiff": 0.0, "conf_verb_maxdiff": 0.0, "h3_correct_mismatch": 0}
    df = pd.read_parquet(run / "items.parquet").set_index("item_id")
    for name in sorted(by):
        its = by[name]
        wrong, same, y, agree, verb, ln, cl = [], [], [], [], [], [], []
        for it in its:
            hp = calls[it["item_id"]]["haiku"]
            assert all(i in hp for i in range(1, 6)), f"{it['item_id']}: missing Haiku pass"
            ps = [hp[i] for i in range(1, 6)]
            lines = [p["lines"] for p in ps]
            wrong.append([not F.judge(it["truth"], it["plant"], F.supported([l], 1), F.TOL)[1] for l in lines])
            same.append([F.sim(lines[a], lines[b]) == 1.0 for a, b in pairs5])
            cor3 = F.judge(it["truth"], it["plant"], F.supported(lines[:3], 2), F.TOL)[1]
            y.append(int(not cor3))
            ag = F.mean_pair_sim(lines[:3])
            cs = [p["conf"] for p in ps[:3] if p["conf"] is not None]
            vb = sum(cs) / len(cs) / 100
            agree.append(-ag)
            verb.append(-vb)
            ln.append(float(it["covariates"]["n_chars"]))
            cl.append(entries[it["entry_key"]])
            r = df.loc[it["item_id"]]
            chk["conf_agree_maxdiff"] = max(chk["conf_agree_maxdiff"], abs(ag - r["conf_agree"]))
            chk["conf_verb_maxdiff"] = max(chk["conf_verb_maxdiff"], abs(vb - r["conf_verb"]))
            chk["h3_correct_mismatch"] += int(bool(cor3) != bool(r["h3_correct"]))
        wrong, y = np.array(wrong, bool), np.array(y)
        ww, sww, rr, srr = pair_stats(wrong, np.array(same, bool))
        _, cl_ids = np.unique(cl, return_inverse=True)
        scores = {"AGREE": np.array(agree), "VERB": np.array(verb), "DIFF": _signed_diff(np.array(ln), y)}
        strata.append(Stratum(name, "critic", 5, wrong.sum(1), ww, sww, rr, srr, y, cl_ids, scores,
                              {"item_ids": [it["item_id"] for it in its], "kinds": sorted({it["kind"] for it in its}), "entry": np.array(cl)}))
    return strata, chk


def repro_checks(crit_chk, p0):
    """Loader gate (frozen in PREREG section 8)."""
    res = json.loads((RUNS / "p0" / "result.json").read_text())
    r = {"p0_native_auc": p0.extra["native_auc_agreement"], "p0_result_json_auc": res["auc_agreement_only"],
         "p0_agree_matches_groups": p0.extra["native_agree_matches_groups"], **crit_chk}
    r["p0_ok"] = abs(r["p0_native_auc"] - r["p0_result_json_auc"]) < 1e-6 and r["p0_agree_matches_groups"]
    r["critic_ok"] = crit_chk["conf_agree_maxdiff"] < 1e-9 and crit_chk["h3_correct_mismatch"] == 0
    df = pd.read_parquet(RUNS / "AR3" / "items.parquet")
    t = df[(df.split == "test") & df.valid.astype(bool) & (df.w > 0)]
    pt = json.loads((RUNS / "AR3" / "result.json").read_text())["point"]
    for sig in ("agree", "verb"):
        for tgt, col in (("s", "proxy_error_sonnet"), ("t", "proxy_error_truth")):
            mine = S.weighted_auroc(t[col].to_numpy(float), -t[f"conf_{sig}"].to_numpy(float), t.w.to_numpy(float))
            r[f"ar3_weighted_auc_{sig}_{tgt}"] = {"mine": float(mine), "result_json": float(pt[f"auc_{tgt}_{sig}"])}
    r["ar3_weighted_ok"] = all(abs(v["mine"] - v["result_json"]) < 1e-6 for k, v in r.items() if k.startswith("ar3_weighted_auc"))
    r["all_ok"] = bool(r["p0_ok"] and r["critic_ok"] and r["ar3_weighted_ok"])
    return r


# ------------------------------------------------------------------ exploratory (post hoc, never part of the verdict)


def pool_strata(name, sts, weight=None):
    """Concatenate critic strata (they carry global entry ids in extra["entry"]) into one. Only signals present in every
    stratum are kept. DIFF is dropped: its sign was chosen per stratum, so a pooled DIFF would be a different control."""
    assert len({s.role for s in sts}) == 1 and len({s.n_pass for s in sts}) == 1
    keys = [k for k in sts[0].scores if k != "DIFF" and all(k in s.scores for s in sts)]
    cat = lambda f: np.concatenate([getattr(s, f) for s in sts])
    cl = np.concatenate([s.extra["entry"] for s in sts])  # global entry ids, so an entry in two strata stays one cluster
    return Stratum(name, sts[0].role, sts[0].n_pass, cat("m"), cat("ww"), cat("sww"), cat("rr"), cat("srr"), cat("y"),
                   np.unique(cl, return_inverse=True)[1], {k: np.concatenate([s.scores[k] for s in sts]) for k in keys},
                   {"item_ids": sum([s.extra.get("item_ids", []) for s in sts], [])}, weight)


def unanimity(st):
    """Share of items whose first three outputs all agree, among wrong cheap decisions and among correct ones, and the
    share of unanimous items whose decision is wrong. Uses the stratum's fixed weights when it has them."""
    w = np.ones(st.n_items) if st.weight is None else np.asarray(st.weight, float)
    una, wrong = st.scores["AGREE"] == -1.0, st.y == 1

    def share(num, den):
        return float((w * num * den).sum() / (w * den).sum()) if (w * den).sum() > 0 else float("nan")

    return {"unanimous_given_wrong": share(una, wrong), "unanimous_given_right": share(una, ~wrong),
            "wrong_given_unanimous": share(wrong, una)}


def run_t1(kept, point, boot, B):
    """T1 block. Below MIN_STRATA_T1 kept strata nothing is fitted: 3 points cannot carry a 4-parameter model."""
    names = [s.name for s in kept]
    if len(kept) < MIN_STRATA_T1:
        return {"verdict": "INCONCLUSIVE", "reason": f"{len(kept)} kept strata < {MIN_STRATA_T1}; models not fitted", "strata": names}
    cols = ["a", "c_r", "c_w", "icc", "auc_AGREE"]
    P = np.array([[point[n][c] for c in cols] for n in names])
    m1 = t1_models(*P.T)
    gaps, maes = np.full(B, np.nan), np.full(B, np.nan)
    stack = np.stack([np.stack([boot[n][c] for c in cols], axis=1) for n in names], axis=1)  # B x S x 5
    for b in range(B):
        if np.isfinite(stack[b]).all():
            r = t1_models(*stack[b].T)
            gaps[b], maes[b] = r["gap"], r["mae_coll"]
    return {"verdict": t1_verdict(m1, len(kept)), **m1, "gap_ci90": list(ci(gaps)), "mae_coll_ci90": list(ci(maes)),
            "n_boot_ok": int(np.isfinite(gaps).sum()), "strata": names, "inputs": {c: P[:, j].tolist() for j, c in enumerate(cols)}}


def exploratory(cands, B, seed0):
    """Descriptive table for every candidate stratum, kept or not, plus pooled critic rows. Floors are not applied;
    the counts of wrong and correct decisions are in the table so that fragile AUROCs can be seen as such."""
    crit = [s for s in cands if s.role == "critic"]
    plants = [s for s in crit if s.name != "clean"]
    df = pd.read_parquet(RUNS / "AR3" / "items.parquet").set_index("item_id")
    allc = pool_strata("critic: all items, unweighted", crit)
    wts = df.loc[allc.extra["item_ids"], "w"].to_numpy(float)
    wts = wts * len(wts) / wts.sum()  # mean weight 1, so N-1 in the ICC is not near 0; the weighted ICC is then only a rough figure
    rows = list(cands) + [pool_strata("critic: all plants", plants), allc,
                          pool_strata("critic: all items, design-weighted (75% clean)", crit, weight=wts)]
    out = []
    for i, st in enumerate(rows):
        p, b = point_stats(st), bootstrap_stratum(st, B, seed0 + i)
        r = {"name": st.name, "role": st.role, "items": st.n_items, "clusters": st.n_clusters, "wrong_dec": int(st.y.sum()),
             "right_dec": int(st.n_items - st.y.sum()), "point": p, "ci90": {k: ci(v) for k, v in b.items()}}
        r.update(unanimity(st))
        out.append(r)
    return out


def spearman_vs_auc(rows, names):
    """Spearman correlation across the named strata of the AGREE AUROC with each pass-level statistic (descriptive only)."""
    from scipy.stats import spearmanr
    sel = [r for r in rows if r["name"] in names]
    auc = [r["point"]["auc_AGREE"] for r in sel]
    return {"strata": [r["name"] for r in sel],
            **{k: float(spearmanr([r["point"][k] for r in sel], auc)[0]) for k in ("c_w", "icc", "a", "c_r")}}


def build_explore_table(rows):
    f = lambda r, k: f"{fmt(r['point'][k])} [{fmt(r['ci90'][k][0])}, {fmt(r['ci90'][k][1])}]" if k in r["point"] else "n/a"
    L = ["| stratum | items | wrong / right dec. | a | ICC | c_r | c_w | unanimous if wrong | unanimous if right | AUROC AGREE | AUROC VERB | AUROC DIFF | recall25 AGREE | recall25 VERB | recall25 DIFF |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['items']} | {r['wrong_dec']} / {r['right_dec']} | {f(r, 'a')} | {f(r, 'icc')} | {f(r, 'c_r')} | {f(r, 'c_w')} | "
                 f"{fmt(r['unanimous_given_wrong'], 2)} | {fmt(r['unanimous_given_right'], 2)} | {f(r, 'auc_AGREE')} | {f(r, 'auc_VERB')} | {f(r, 'auc_DIFF')} | "
                 f"{f(r, 'rec_AGREE')} | {f(r, 'rec_VERB')} | {f(r, 'rec_DIFF')} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main


def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def build_tables(kept, point, boot):
    L = ["| stratum | role | items | clusters | wrong dec. | a | ICC | c_r | c_w | AUROC AGREE (90% CI) | half-width | AUROC VERB | AUROC DIFF | recall25 AGREE | recall25 VERB | recall25 DIFF |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in kept:
        p, b = point[st.name], boot[st.name]
        lo, hi = ci(b["auc_AGREE"])
        g = lambda k: fmt(p[k]) if k in p else "n/a"
        L.append(f"| {st.name} | {st.role} | {st.n_items} | {st.n_clusters} | {int(st.y.sum())} | {fmt(p['a'])} | {fmt(p['icc'])} | {fmt(p['c_r'])} | {fmt(p['c_w'])} | "
                 f"{fmt(p['auc_AGREE'])} [{fmt(lo)}, {fmt(hi)}] | {fmt((hi - lo) / 2)} | {g('auc_VERB')} | {g('auc_DIFF')} | {g('rec_AGREE')} | {g('rec_VERB')} | {g('rec_DIFF')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    sha = verify_prereg()
    print("PREREG sha256", sha[:16], flush=True)

    cands = [load_verifier("p0", "p0", "answer"), load_verifier("E06", "E06", "answer"),
             load_verifier("E10p0/instruct", "E10p0/instruct", "sig"), load_verifier("E10p0/base", "E10p0/base", "sig")]
    crit, chk = load_critic()
    cands += crit
    repro = repro_checks(chk, cands[0])
    print("repro all_ok", repro["all_ok"], flush=True)
    if not repro["all_ok"]:
        raise SystemExit("reproduction gate failed: loader bug, stopping before any test statistic")
    if a.check_only:
        return

    kept, dropped, seeds = [], [], {}
    for i, st in enumerate(cands):
        ok, why = keep_stratum(st)
        if ok:
            kept.append(st)
            seeds[st.name] = SEED + i
        else:
            dropped.append({"name": st.name, "reason": why, "items": st.n_items, "wrong_dec": int(st.y.sum())})
    print("kept", [s.name for s in kept], "dropped", [d["name"] for d in dropped], flush=True)

    point = {st.name: point_stats(st) for st in kept}
    boot = {st.name: bootstrap_stratum(st, a.B, seeds[st.name]) for st in kept}
    print(f"bootstrap done {time.time() - t0:.0f}s", flush=True)

    hw = {n: (ci(boot[n]["auc_AGREE"])[1] - ci(boot[n]["auc_AGREE"])[0]) / 2 for n in boot}
    t3, n_bad = t3_verdict(list(hw.values()))
    t1 = run_t1(kept, point, boot, a.B)

    vnames = [s.name for s in kept if s.role == "verifier" and "DIFF" in s.scores]
    cnames = [s.name for s in kept if s.role == "critic"]
    if vnames and cnames:
        pv, bv = role_signals(point, boot, vnames, ["AGREE", "DIFF"])
        pc, bc = role_signals(point, boot, cnames, ["AGREE", "VERB", "DIFF"])
        t2 = t2_verdict(pv, bv, pc, bc)
        t2.update(verifier_recall=pv, critic_recall=pc, verifier_strata=vnames, critic_strata=cnames)
    else:
        t2 = {"verdict": "FAIL", "note": f"a role has no kept stratum (verifier {vnames}, critic {cnames})"}

    verdict = final_verdict(t1["verdict"], t2["verdict"], t3)
    explore = exploratory(cands, a.B, SEED + 100)
    res = {"prereg_sha256": sha, "B": a.B, "seed": SEED, "repro": repro, "kept": [s.name for s in kept], "dropped": dropped,
           "point": point, "ci90": {n: {k: ci(v) for k, v in boot[n].items()} for n in boot},
           "native_verifier": {s.name: s.extra for s in cands if s.role == "verifier"},
           "t1": t1, "t2": t2, "t3": {"verdict": t3, "n_wide": n_bad, "n_strata": len(kept), "half_widths": hw,
                                     "note": "computed on the kept strata only; none of them is a critic stratum"},
           "verdict": verdict, "exploratory_post_hoc": explore,
           "exploratory_spearman_verifier4": spearman_vs_auc(explore, ["p0", "E06", "E10p0/instruct", "E10p0/base"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "tables.md").write_text("## Kept strata (preregistered floors)\n\n" + build_tables(kept, point, boot) +
                                   "\n## All candidate strata and pooled critic rows (post hoc, descriptive, no floors)\n\n" + build_explore_table(explore))
    print(json.dumps({"T1": t1["verdict"], "T2": t2["verdict"], "T3": t3, "verdict": verdict}))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

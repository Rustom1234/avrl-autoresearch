"""Budgeted sensitive-content auditing simulator: does ADAPTIVE beat STATIC TRIAGE and UNIFORM?

Single file. Reads feats.parquet (built by prep.py, which reproduces
aa.data.load_fineweb(n=400000, seed=0, ...) exactly and applies aa.data.add_regex_features).

Label = strong_pii (stage-0 proxy for the expensive detector).
Bucket ("field of view") = registered domain with >= 3 docs; everything else is one
"singleton" pool that any policy may also sample from uniformly.
One inspection = pick a bucket, inspect up to 8 uninspected docs, pay 1 detector call per
doc actually inspected, observe those docs' labels.

Usage: python sim.py            (writes result.json)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aa.data import stable_hash  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = os.path.join(HERE, "feats.parquet")
EMB_TEXTS = os.path.join(HERE, "emb_texts.parquet")
CLUSTER_CACHE = os.path.join(HERE, "doc_clusters.npz")

MIN_BUCKET = 3
FOV = 8                    # docs inspected per bucket visit
BUDGETS = [0.01, 0.03, 0.10]
SEEDS = [0, 1, 2]
WARMUP_FRAC = 0.05         # of the budget, for the static model
TS_PRIOR_STRENGTH = 8.0    # pseudo-counts for the static-informed Thompson prior
KNN_K = 10
N_CLUSTERS = 256
POOL = "__POOL__"


# ---------------------------------------------------------------- field of view

def make_field(y, bcode, nb, rng, avail_mask=None):
    """Group available docs by bucket, shuffled within bucket. O(1) inspections."""
    idx = np.arange(len(y)) if avail_mask is None else np.nonzero(avail_mask)[0]
    idx = rng.permutation(idx)
    idx = idx[np.argsort(bcode[idx], kind="stable")]
    b_sorted = bcode[idx]
    start = np.searchsorted(b_sorted, np.arange(nb + 1))
    return {"doc": idx, "start": start, "size": np.diff(start),
            "cum": np.concatenate([[0], np.cumsum(y[idx].astype(np.int64))]),
            "ptr": np.zeros(nb, dtype=np.int64), "y": y}


def inspect(F, b, k=FOV):
    """Returns (n_calls, n_hits). Cost = docs actually inspected."""
    avail = F["size"][b] - F["ptr"][b]
    k = int(min(k, avail))
    if k <= 0:
        return 0, 0
    s = F["start"][b] + F["ptr"][b]
    h = int(F["cum"][s + k] - F["cum"][s])
    F["ptr"][b] += k
    return k, h


def stream_from(events, warm_stream=None):
    """events: list of (n_calls, n_hits). Returns a per-call 0/1 hit stream (order within an
    inspection is irrelevant for all reported metrics, which are call-indexed cumulative sums)."""
    parts = [] if warm_stream is None else [np.asarray(warm_stream, dtype=np.int8)]
    for k, h in events:
        if k:
            a = np.zeros(k, dtype=np.int8)
            a[:h] = 1
            parts.append(a)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int8)


# ---------------------------------------------------------------- cheap features

_HV_TRI = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=512,
                            norm=None, alternate_sign=False, lowercase=True)
_HV_TOK = HashingVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", n_features=256,
                            norm=None, alternate_sign=False, lowercase=True)
_HV_TLD = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=64,
                            norm=None, alternate_sign=False, lowercase=True)


def domain_stats(df):
    """Label-free per-domain aggregates. NOTE: no regex counts, no labels."""
    d = pd.DataFrame({"domain": df["domain"].to_numpy(),
                      "_lt": np.log1p(df["token_count"].to_numpy(float)),
                      "_ls": df["language_score"].to_numpy(float),
                      "_en": (df["language"].to_numpy() == "en").astype(float)})
    g = d.groupby("domain", sort=True)
    s = pd.DataFrame({
        "n": g.size(),
        "mean_ltok": g["_lt"].mean(),
        "std_ltok": g["_lt"].std().fillna(0.0),
        "mean_lscore": g["_ls"].mean(),
        "frac_en": g["_en"].mean(),
    })
    s["dom_len"] = [len(d) for d in s.index]
    s["tld"] = [d.split(".", 1)[1] if "." in d else "" for d in s.index]
    return s


def featurize(stats):
    doms = list(stats.index)
    num = np.column_stack([
        np.log1p(stats["n"].to_numpy(float)), stats["mean_ltok"].to_numpy(float),
        stats["std_ltok"].to_numpy(float), stats["mean_lscore"].to_numpy(float),
        stats["frac_en"].to_numpy(float), np.log1p(stats["dom_len"].to_numpy(float))])
    num = np.nan_to_num(num)
    num = (num - num.mean(0)) / (num.std(0) + 1e-9)
    names = [d.split(".", 1)[0].replace("-", " ") for d in doms]
    return sp.hstack([sp.csr_matrix(num), _HV_TRI.transform(doms),
                      _HV_TOK.transform(names), _HV_TLD.transform(stats["tld"])]).tocsr()


# ---------------------------------------------------------------- embeddings (optional)

def doc_clusters(seed=0, log=None, timing=None):
    """MiniLM(50k-doc subsample) -> MiniBatchKMeans(256). Cached. Returns (orders, labels) or None."""
    if os.path.exists(CLUSTER_CACHE):
        z = np.load(CLUSTER_CACHE)
        if timing is not None and "wall" in z:
            timing["wall_clock_sec_embedding_step"] = float(z["wall"])
        return z["order"], z["cluster"]
    te = time.time()
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans
        if not torch.cuda.is_available():
            return None
        t = pd.read_parquet(EMB_TEXTS)
        m = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        E = m.encode(t["text"].tolist(), batch_size=256, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=3, batch_size=4096)
        lab = km.fit_predict(E).astype(np.int32)
        order = t["order"].to_numpy(np.int64)
        w = round(time.time() - te, 1)
        np.savez(CLUSTER_CACHE, order=order, cluster=lab, wall=np.array(w))
        if timing is not None:
            timing["wall_clock_sec_embedding_step"] = w
        return order, lab
    except Exception as e:  # report as failure, do not fake it
        if log is not None:
            log.append(f"embedding step failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------- setup per evaluation set

def build(df, fit_df=None, clusters=None, seed=0):
    """Buckets, per-bucket cheap-feature vectors, kNN neighbour graph, oracle rates."""
    dom = df["domain"].to_numpy()
    vc = pd.Series(dom).value_counts()
    keep = set(vc[vc >= MIN_BUCKET].index)
    key = np.where(pd.Series(dom).isin(keep).to_numpy(), dom, POOL)
    bcode, buckets = pd.factorize(key)
    nb = len(buckets)
    y = df["strong_pii"].to_numpy(np.int8)

    stats = domain_stats(df)
    dom_index = {d: i for i, d in enumerate(stats.index)}
    doc_dom = np.array([dom_index[d] for d in dom])
    Xd = featurize(stats)                                   # per distinct domain

    # bucket-level cheap vectors: mean of member docs' own-domain rows
    M = sp.csr_matrix((np.ones(len(df)), (bcode, doc_dom)), shape=(nb, len(stats)))
    M = M.multiply(1.0 / np.asarray(M.sum(1)))
    Xb = sp.csr_matrix(M) @ Xd

    # embedding block: bucket histogram over MiniBatchKMeans clusters of a 50k-doc subsample
    Hb, cov = None, 0.0
    if clusters is not None:
        order, lab = clusters
        pos = pd.Index(df["order"].to_numpy()).get_indexer(order)
        ok = pos >= 0
        if ok.sum() > 0:
            Hb = sp.csr_matrix((np.ones(int(ok.sum())), (bcode[pos[ok]], lab[ok])),
                               shape=(nb, N_CLUSTERS))
            rs = np.asarray(Hb.sum(1)).ravel()
            cov = float((rs > 0).mean())
            Hb = sp.csr_matrix(Hb.multiply(1.0 / np.maximum(rs, 1)[:, None]))

    fit_svd = Xb if fit_df is None else _fit_side(fit_df)
    nbr = {"emb": _knn(Xb, Hb, fit_svd, seed), "noemb": _knn(Xb, None, fit_svd, seed)}

    rate = np.bincount(bcode, weights=y, minlength=nb) / np.bincount(bcode, minlength=nb)
    size = np.bincount(bcode, minlength=nb)
    return {"df": df, "y": y, "bcode": bcode, "nb": nb, "buckets": buckets, "stats": stats,
            "Xd": Xd, "dom_index": dom_index, "doc_dom": doc_dom, "Xb": Xb,
            "nbr": nbr, "rate": rate, "size": size, "emb_cov": cov,
            "pool_id": int(np.nonzero(buckets == POOL)[0][0]) if POOL in set(buckets) else -1}


def _fit_side(fit_df):
    s = domain_stats(fit_df)
    return featurize(s)


def _knn(Xb, Hb, Xfit, seed):
    """Top-K cosine neighbours in a cheap, label-free space (SVD of hashed domain features,
    plus SVD of the cluster histogram when embeddings are available)."""
    blocks = []
    sv = TruncatedSVD(n_components=64, random_state=seed).fit(Xfit)
    A = sv.transform(Xb)
    blocks.append(A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9))
    if Hb is not None:
        B = TruncatedSVD(n_components=32, random_state=seed).fit_transform(Hb)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        blocks.append(B)
    V = np.hstack(blocks)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine", algorithm="brute").fit(V)
    d, i = nn.kneighbors(V)
    return i[:, 1:].astype(np.int32), (1.0 - d[:, 1:]).astype(np.float32)


# ---------------------------------------------------------------- policies

def pol_uniform(S, budget, seed, fit_S=None):
    rng = np.random.default_rng(1000 + seed)
    p = rng.permutation(len(S["y"]))[:budget]
    return S["y"][p].astype(np.int8)


def pol_oracle(S, budget, seed, fit_S=None):
    rng = np.random.default_rng(2000 + seed)
    F = make_field(S["y"], S["bcode"], S["nb"], rng)
    order = np.lexsort((-S["size"], -S["rate"]))
    ev, spent = [], 0
    for b in order:
        while spent < budget:
            k, h = inspect(F, b, min(FOV, budget - spent))
            if k == 0:
                break
            ev.append((k, h)); spent += k
        if spent >= budget:
            break
    return stream_from(ev)


def _warmup(S, budget, seed):
    rng = np.random.default_rng(3000 + seed)
    w = int(round(WARMUP_FRAC * budget))
    p = rng.permutation(len(S["y"]))[:w]
    mask = np.ones(len(S["y"]), bool); mask[p] = False
    return p, S["y"][p].astype(np.int8), mask, rng


def _fit_static(S, train_doc_idx, train_y, seed):
    X = S["Xd"][S["doc_dom"][train_doc_idx]]
    if train_y.max() == train_y.min():
        return np.full(S["nb"], 0.5)
    lr = LogisticRegression(max_iter=300, C=1.0, random_state=seed).fit(X, train_y)
    p_dom = lr.predict_proba(S["Xd"])[:, 1]
    p_doc = p_dom[S["doc_dom"]]
    return (np.bincount(S["bcode"], weights=p_doc, minlength=S["nb"])
            / np.bincount(S["bcode"], minlength=S["nb"]))


def _fit_static_transfer(S, fit_S, seed):
    """Train on ALL labels of the fit half (already-audited shard), score the eval half."""
    if "_p_transfer" in S:
        return S["_p_transfer"]
    X = fit_S["Xd"][fit_S["doc_dom"]]
    lr = LogisticRegression(max_iter=300, C=1.0, random_state=seed).fit(X, fit_S["y"])
    p_dom = lr.predict_proba(S["Xd"])[:, 1]
    p_doc = p_dom[S["doc_dom"]]
    S["_p_transfer"] = (np.bincount(S["bcode"], weights=p_doc, minlength=S["nb"])
                        / np.bincount(S["bcode"], minlength=S["nb"]))
    return S["_p_transfer"]


def _static_scores(S, budget, seed, fit_S):
    if fit_S is None:
        idx, warm_stream, mask, rng = _warmup(S, budget, seed)
        pb = _fit_static(S, idx, S["y"][idx], seed)
        return pb, warm_stream, mask, rng, len(idx)
    rng = np.random.default_rng(3000 + seed)
    return _fit_static_transfer(S, fit_S, seed), None, None, rng, 0


def pol_static(S, budget, seed, fit_S=None):
    pb, warm_stream, mask, rng, spent = _static_scores(S, budget, seed, fit_S)
    F = make_field(S["y"], S["bcode"], S["nb"], rng, mask)
    ev = []
    for b in np.lexsort((-S["size"], -pb)):
        while spent < budget:
            k, h = inspect(F, b, min(FOV, budget - spent))
            if k == 0:
                break
            ev.append((k, h)); spent += k
        if spent >= budget:
            break
    return stream_from(ev, warm_stream)


def _ts(S, budget, seed, a0, b0, warm_stream, mask, rng, spent):
    F = make_field(S["y"], S["bcode"], S["nb"], rng, mask)
    a, b = a0.copy(), b0.copy()
    dead = F["size"] <= 0
    ev = []
    while spent < budget and not dead.all():
        d = rng.beta(a, b)
        d[dead] = -1.0
        bi = int(np.argmax(d))
        k, h = inspect(F, bi, min(FOV, budget - spent))
        if k == 0:
            dead[bi] = True
            continue
        a[bi] += h; b[bi] += k - h; spent += k
        ev.append((k, h))
        if F["size"][bi] - F["ptr"][bi] <= 0:
            dead[bi] = True
    return stream_from(ev, warm_stream)


def pol_ts_flat(S, budget, seed, fit_S=None):
    rng = np.random.default_rng(4000 + seed)
    return _ts(S, budget, seed, np.ones(S["nb"]), np.ones(S["nb"]), None, None, rng, 0)


def pol_ts_static(S, budget, seed, fit_S=None):
    pb, warm_stream, mask, rng, spent = _static_scores(S, budget, seed, fit_S)
    pb = np.clip(pb, 1e-3, 1 - 1e-3)
    return _ts(S, budget, seed, TS_PRIOR_STRENGTH * pb, TS_PRIOR_STRENGTH * (1 - pb),
               warm_stream, mask, rng, spent)


def _neighbor(S, budget, seed, which):
    """After a hit, re-prioritise the hit bucket and its cheap-similarity neighbours."""
    import heapq
    rng = np.random.default_rng(5000 + seed)
    F = make_field(S["y"], S["bcode"], S["nb"], rng)
    nbr_i, nbr_s = S["nbr"][which]
    pri = np.zeros(S["nb"])
    heap, dead = [], F["size"] <= 0
    explore = rng.permutation(S["nb"]); cur = 0
    ev, spent = [], 0
    while spent < budget:
        bi = None
        while heap:
            negp, cand = heapq.heappop(heap)
            if dead[cand] or pri[cand] != -negp:
                continue
            bi = cand; pri[cand] = 0.0; break
        if bi is None:
            while cur < len(explore) and dead[explore[cur]]:
                cur += 1
            if cur >= len(explore):
                break
            bi = int(explore[cur]); cur += 1
        k, h = inspect(F, bi, min(FOV, budget - spent))
        if k == 0:
            dead[bi] = True; continue
        spent += k; ev.append((k, h))
        if F["size"][bi] - F["ptr"][bi] <= 0:
            dead[bi] = True
        if h > 0:
            cand = [(bi, 1.0)] + list(zip(nbr_i[bi], nbr_s[bi]))
            for c, s in cand:
                c = int(c)
                if dead[c]:
                    continue
                p = h * float(s)
                if p > pri[c]:
                    pri[c] = p
                    heapq.heappush(heap, (-p, c))
    return stream_from(ev)


def pol_neighbor(S, budget, seed, fit_S=None):
    return _neighbor(S, budget, seed, "emb")


def pol_neighbor_noemb(S, budget, seed, fit_S=None):
    return _neighbor(S, budget, seed, "noemb")


POLICIES = {"uniform": pol_uniform, "static_triage": pol_static, "ts_flat": pol_ts_flat,
            "ts_static_prior": pol_ts_static, "neighbor_expansion": pol_neighbor,
            "neighbor_expansion_noemb": pol_neighbor_noemb, "oracle": pol_oracle}


# ---------------------------------------------------------------- diagnostics

def diagnostics(S, budget, fit_S=None, seed=0):
    """Why the policies land where they do: static-model ranking quality, where the budget goes,
    and whether the cheap kNN neighbourhood carries any label signal."""
    from sklearn.metrics import roc_auc_score
    pb, _, _, _, _ = _static_scores(S, budget, seed, fit_S)
    order = np.lexsort((-S["size"], -pb))
    cs = np.cumsum(S["size"][order])
    ntop = int(np.searchsorted(cs, budget, side="left")) + 1
    top = order[:ntop]
    pool_calls = int(S["size"][S["pool_id"]]) if S["pool_id"] in set(top.tolist()) else 0
    d = {
        "static_doc_level_auc": float(roc_auc_score(S["y"], pb[S["bcode"]])),
        "static_bucket_rank_of_pool": int(np.nonzero(order == S["pool_id"])[0][0]),
        "n_buckets": int(S["nb"]),
        "static_top_ranked_buckets_covering_budget": ntop,
        "static_frac_budget_in_pool": round(min(pool_calls, budget) / budget, 4),
        "uniform_frac_budget_in_pool": round(float(S["size"][S["pool_id"]]) / len(S["y"]), 4),
        "true_hit_rate_of_static_top_buckets": float(
            np.average(S["rate"][top], weights=np.maximum(S["size"][top], 1))),
    }
    # cheap-kNN autocorrelation of "bucket contains >=1 hit", real buckets only
    hot = (S["rate"] > 0)
    real = np.ones(S["nb"], bool); real[S["pool_id"]] = False
    for which in ("emb", "noemb"):
        nbr_i = S["nbr"][which][0]
        nb_hot = hot[nbr_i].mean(axis=1)
        base = float(hot[real].mean())
        sel = real & hot
        d[f"knn_{which}_P(nbr_hot|self_hot)"] = float(nb_hot[sel].mean())
        d[f"knn_{which}_P(nbr_hot)"] = base
        d[f"knn_{which}_ratio"] = float(nb_hot[sel].mean() / base) if base > 0 else None
    return d


# ---------------------------------------------------------------- experiment driver

def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0)), "n": len(v)}


def calls_to(stream, target):
    c = np.cumsum(stream.astype(np.int64))
    i = int(np.searchsorted(c, target, side="left"))
    return i + 1 if i < len(c) else None


def run_setting(S, budgets_abs, seeds, fit_S=None, curve_at=None):
    out, curves, streams = {}, {}, {}
    for name, fn in POLICIES.items():
        out[name] = {}
        for bname, B in budgets_abs.items():
            hits, per_call = [], []
            for sd in seeds:
                st = fn(S, B, sd, fit_S)
                streams[(name, bname, sd)] = st
                hits.append(int(st.sum()))
                per_call.append(float(st.sum()) / max(len(st), 1))
            out[name][bname] = {"hits": agg(hits), "hits_per_call": agg(per_call),
                                "calls": B, "hits_all_seeds": hits}
        if curve_at is not None:
            st = streams[(name, curve_at, seeds[0])]
            c = np.cumsum(st.astype(np.int64))
            step = max(1, len(c) // 100)
            curves[name] = {"calls": list(range(step, len(c) + 1, step)),
                            "cum_hits": [int(c[i - 1]) for i in range(step, len(c) + 1, step)]}
    return out, curves, streams


def main():
    t0 = time.time()
    log, assumptions = [], []
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    df = pd.read_parquet(FEATS)
    n = len(df)

    timing = {}
    cl = doc_clusters(seed=0, log=log, timing=timing)
    try:
        timing["wall_clock_sec_prep"] = json.load(open(os.path.join(HERE, "prep_meta.json")))["wall_clock_sec"]
    except Exception:
        timing["wall_clock_sec_prep"] = None
    emb_ok = cl is not None

    S = build(df, clusters=cl, seed=0)
    budgets_abs = {f"{int(b*100)}pct": int(round(b * n)) for b in BUDGETS}
    t_build = time.time() - t0

    main_res, curves, streams = run_setting(S, budgets_abs, SEEDS, None, curve_at="3pct")

    # calls to reach 50% of oracle hits at the 3% budget, measured on each policy's 10% run
    target = 0.5 * main_res["oracle"]["3pct"]["hits"]["mean"]
    c2r = {}
    for name in POLICIES:
        vals = [calls_to(streams[(name, "10pct", sd)], target) for sd in SEEDS]
        c2r[name] = {"agg": agg(vals), "per_seed": vals,
                     "not_reached_seeds": int(sum(v is None for v in vals))}

    # ---- transfer: fit on one half, evaluate on the other
    transfer = {}
    for split in ["row_half", "domain_disjoint"]:
        if split == "row_half":
            a, b = df.iloc[: n // 2], df.iloc[n // 2:]
        else:
            h = np.array([stable_hash(d) % 2 for d in df["domain"]])
            a, b = df[h == 0], df[h == 1]
        Sa = build(a.reset_index(drop=True), clusters=cl, seed=0)
        Sb = build(b.reset_index(drop=True), fit_df=a, clusters=cl, seed=0)
        ba = {"3pct": int(round(0.03 * len(b)))}
        r, _, _ = run_setting(Sb, ba, SEEDS, fit_S=Sa)
        transfer[split] = {"n_fit": len(a), "n_eval": len(b), "n_buckets_eval": Sb["nb"],
                           "eval_bucket_embedding_coverage": round(float(Sb["emb_cov"]), 4),
                           "diagnostics": diagnostics(Sb, ba["3pct"], Sa, 0),
                           "overlap_domains": int(len(set(a["domain"]) & set(b["domain"]))),
                           "results": r}

    assumptions += [
        "prep.py streams parquet row-groups but reproduces aa.data.load_fineweb(n=400000, seed=0) "
        "exactly (same RandomState(0).choice positional indices, same order); done to stay under the RAM cap.",
        "Label = strong_pii from aa.data.add_regex_features logic (email|phone|ssn|credit|secret > 0).",
        "Bucket = registered domain with >= 3 docs. All docs in domains with 1-2 docs form ONE "
        "'singleton' pool bucket that every policy may inspect like any other bucket.",
        "An inspection costs one detector call per doc actually inspected (<= 8), not a flat 8; "
        "the last inspection of a run is truncated to the remaining budget.",
        "Static-triage warm-up = 5% of the budget, spent as uniform random DOC inspections "
        "(not bucket visits): more labelled points per call. Warm-up hits count toward the total "
        "and warm-up docs are removed from the buckets.",
        "The logistic regression is trained at doc level on the doc's own registered-domain cheap "
        "features; a bucket's score is the mean predicted probability over its docs' domains "
        "(identical to a per-domain score for real buckets, a mixture for the pool).",
        "Cheap features: log bucket size, mean/std log token_count, mean language_score, frac "
        "language=='en', log domain-string length, hashed char_wb 3-grams of the domain (512), "
        "hashed domain-name word tokens (256), hashed TLD (64). No labels, no regex counts.",
        f"Thompson static prior = Beta({TS_PRIOR_STRENGTH}*p, {TS_PRIOR_STRENGTH}*(1-p)), i.e. one "
        "inspection's worth of pseudo-counts; it pays the same 5%-of-budget warm-up as static triage.",
        "Neighbour expansion: on a hit, the hit bucket itself and its top-10 cosine neighbours get "
        "priority h*sim; on a miss the bucket is dropped from the frontier; empty frontier -> uniform "
        "random bucket. Neighbour space = TruncatedSVD(64) of the hashed domain features "
        "(+ TruncatedSVD(32) of the bucket's MiniBatchKMeans(256) cluster histogram when embeddings ran).",
        "Oracle = buckets sorted by true hit rate desc, ties by size desc, inspected top-down.",
        "'Calls to reach 50% of oracle hits at 3% budget' is measured on each policy's 10%-budget run, "
        "so a policy may spend more than the 3% budget to get there.",
        "Transfer: the static model is trained on ALL labels of the fit half (an already-audited "
        "shard), so transfer-static pays NO warm-up; the SVD neighbour space is fit on the fit half. "
        "row_half = first/second half of the loaded (already randomly permuted) sample; "
        "domain_disjoint = md5(domain)%2, the only split with no domain overlap.",
        "Seeds 0,1,2 control within-bucket doc order, warm-up sampling, Thompson draws and "
        "exploration order; the data subsample itself is fixed at seed 0 as specified.",
    ]
    assumptions.append(
        "feats.parquet is stored in load_fineweb sample order, so the row_half transfer split puts "
        "the entire 50k embedding subsample in the FIT half: eval-side bucket embedding coverage is "
        "0 there and neighbor_expansion == neighbor_expansion_noemb by construction. The "
        "domain_disjoint split keeps ~53% eval-side coverage. See transfer[*].eval_bucket_embedding_coverage.")
    if not emb_ok:
        assumptions.append("EMBEDDING STEP DID NOT RUN; neighbor_expansion == neighbor_expansion_noemb.")

    res = {
        "run": {"script": os.path.join(HERE, "sim.py"), "prep": os.path.join(HERE, "prep.py"),
                "n_docs": int(n), "data_seed": 0, "policy_seeds": SEEDS,
                "wall_clock_sec_total": None, "wall_clock_sec_setup": round(t_build, 1), **timing,
                "budgets_calls": budgets_abs, "fov_docs_per_inspection": FOV},
        "corpus": {
            "n_docs": int(n), "strong_pii_rate": float(df["strong_pii"].mean()),
            "total_hits": int(df["strong_pii"].sum()),
            "distinct_domains": int(df["domain"].nunique()),
            "n_buckets_ge3": int((S["size"] > 0).sum() - 1),
            "docs_in_real_buckets": int(n - S["size"][S["pool_id"]]),
            "pool_docs": int(S["size"][S["pool_id"]]),
            "pool_hit_rate": float(S["rate"][S["pool_id"]]),
            "real_bucket_hit_rate": float(
                (df["strong_pii"].sum() - S["rate"][S["pool_id"]] * S["size"][S["pool_id"]])
                / max(n - S["size"][S["pool_id"]], 1)),
            "hits_reachable_in_real_buckets": int(round(
                float(df["strong_pii"].sum()) - S["rate"][S["pool_id"]] * S["size"][S["pool_id"]])),
            "median_bucket_size": float(np.median(S["size"][S["size"] > 0])),
            "embedding_step_ran": bool(emb_ok),
            "bucket_embedding_coverage": round(float(S["emb_cov"]), 4),
        },
        "main": main_res,
        "calls_to_50pct_of_oracle_hits_at_3pct": {"target_hits": float(target), **c2r},
        "curves_at_3pct_seed0": curves,
        "diagnostics_main_3pct_seed0": diagnostics(S, budgets_abs["3pct"], None, 0),
        "transfer": transfer,
        "assumptions": assumptions,
        "log": log,
    }
    res["run"]["wall_clock_sec_total"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("run", "corpus")}, indent=1))
    for p in POLICIES:
        print(p, {b: round(main_res[p][b]["hits_per_call"]["mean"], 4) for b in budgets_abs})


if __name__ == "__main__":
    sys.exit(main())

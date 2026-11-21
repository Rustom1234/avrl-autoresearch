"""Curator: rank the pool by similarity to the target sample, then pick documents
greedily with a density-based diversity rule, up to about T tokens.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Measure the word count and simple quality statistics of target_sample and the
     pool, and derive the junk thresholds from the target's own statistics.
  2. Drop damaged, junk and spam-like documents, and exact or near duplicates.
  3. Score each remaining document by bag-of-words TF-IDF similarity to the target:
     a "local centroid" score plus the nearest target documents, with a mild
     penalty for lengths far from the target's.
     Local centroid score: the pool is divided into K clusters (K-Means on the
     TF-IDF vectors). Each cluster centroid gets a weight from its distance to the
     target centroid (closer -> larger weight). A document's score is the weighted
     average of its cosine similarities to the cluster centroids.
  4. Density-based greedy selection: documents are taken one by one, each time the
     highest-scoring document whose cosine similarity to the centroid of the
     already selected set (the "growing" centroid) is at least SIM_THRESH.
     The first SEED_N documents by score are taken unconditionally to start the
     centroid. Documents that fail the test are deferred and re-tested
     periodically as the centroid moves. If nothing passes any more and the
     budget is not filled, the threshold is relaxed step by step.
     The selected documents are then shuffled with the seed so training sees a
     stationary mixture.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import math
import os
import random
import re
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WS = re.compile(r"\s+")
_NONALPHA = re.compile(r"[\W\d_]")
_WORD = re.compile(r"\w+")
_SPAM = re.compile(
    r"click here|buy now|free shipping|subscribe now|limited time offer|order now|"
    r"sign up (?:now|today)|act now|100% free|best price|cheap \w+ online|"
    r"viagra|casino|unsubscribe|all rights reserved|add to cart|http[s]?://|www\.",
    re.I,
)

HEAD = 1500      # characters inspected for the quality statistics
VEC_CHARS = 4000  # characters used for the bag-of-words similarity
N_CLUSTERS = 32   # K for the local-centroid score
KM_FIT_MAX = 30000  # at most this many documents are used to fit K-Means

SIM_THRESH = 0.3   # minimum cosine similarity to the growing centroid
SEED_N = 200       # top-scoring documents taken unconditionally to start the centroid
RECHECK_EVERY = 1000  # deferred documents are re-tested after this many acceptances
RELAX = 0.85       # threshold multiplier when no deferred document passes
WINDOW = 1.6       # only the best WINDOW * budget tokens (by score) are considered


def _stats(s):
    """Return (n_words, alpha_ratio, unique_ratio, zlib_ratio, spam_per_1k_words)."""
    n_words = s.count(" ") + s.count("\n") + 1
    h = s[:HEAD]
    L = len(h)
    if L == 0:
        return 0, 0.0, 0.0, 1.0, 0.0
    alpha = (len(_NONALPHA.sub("", h)) + h.count(" ") + h.count("\n")) / L
    w = h.lower().split()[:200]
    uniq = len(set(w)) / max(1, len(w))
    raw = h.encode("utf-8", "ignore")
    zr = len(zlib.compress(raw, 6)) / max(1, len(raw))
    h2 = s[:VEC_CHARS]
    spam = len(_SPAM.findall(h2)) * 1000.0 / max(1, h2.count(" ") + 1)
    return n_words, alpha, uniq, zr, spam


def _tokens_per_char(texts):
    """Calibrate tokens per character on a small sample (cheap), default 0.25."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        chars = sum(len(t) for t in texts)
        toks = sum(len(x) for x in enc.encode_ordinary_batch(texts))
        if chars > 0 and toks > 0:
            return toks / chars
    except Exception:
        pass
    return 0.25


def _fallback_scores(cand_texts, target_sample):
    """Simple keyword overlap with the target's high-frequency vocabulary."""
    from collections import Counter

    c = Counter()
    for t in target_sample:
        c.update(set(_WORD.findall(t.lower()[:VEC_CHARS])))
    vocab = {w for w, _ in c.most_common(3000)}
    out = np.zeros(len(cand_texts), dtype=np.float64)
    for i, t in enumerate(cand_texts):
        ws = _WORD.findall(t.lower()[:VEC_CHARS])
        if ws:
            out[i] = sum(1 for w in ws if w in vocab) / len(ws)
    return out


def _local_centroid_scores(Xc, centroid, seed):
    """Weighted average of similarities to K-Means cluster centroids of the pool.

    Xc: L2-normalised sparse TF-IDF rows of the pool candidates.
    centroid: unit-norm dense target centroid.
    Each cluster's weight comes from the cosine distance between its centroid and
    the target centroid (closer clusters weigh more).
    """
    from sklearn.cluster import MiniBatchKMeans

    nc = Xc.shape[0]
    k = int(min(N_CLUSTERS, nc))
    if k < 2:
        raise ValueError("too few documents to cluster")

    rs = np.random.RandomState(seed % (2 ** 32))
    if nc > KM_FIT_MAX:
        fit_idx = np.sort(rs.choice(nc, KM_FIT_MAX, replace=False))
        Xfit = Xc[fit_idx]
    else:
        Xfit = Xc

    km = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=30,
        batch_size=4096,
        random_state=seed % (2 ** 32),
    )
    km.fit(Xfit)

    C = np.asarray(km.cluster_centers_, dtype=np.float32)
    norms = np.linalg.norm(C, axis=1)
    valid = norms > 1e-8
    if not valid.any():
        raise ValueError("degenerate clustering")
    Cn = np.zeros_like(C)
    Cn[valid] = C[valid] / norms[valid, None]

    # distance of each cluster centroid to the target centroid -> weight
    cos_ct = (Cn @ centroid.astype(np.float32)).astype(np.float64)
    dist = 1.0 - cos_ct
    dv = dist[valid]
    tau = float(max(dv.std(), 1e-6))
    w = np.zeros(k, dtype=np.float64)
    w[valid] = np.exp(-(dv - dv.min()) / tau)
    w /= w.sum()

    # cluster-wise similarities of every document, then the weighted average
    out = np.zeros(nc, dtype=np.float64)
    CnT = np.ascontiguousarray(Cn.T)
    step = 20000
    for a in range(0, nc, step):
        S = np.asarray(Xc[a:a + step] @ CnT, dtype=np.float64)
        out[a:a + step] = S @ w
    return out


def _density_greedy(Xc, order, costs, budget):
    """Greedy selection against the growing centroid of the selected set.

    Xc: sparse CSR TF-IDF rows (candidates). order: candidate positions sorted by
    decreasing score. costs: estimated tokens per candidate. Returns the list of
    selected candidate positions.
    """
    Xc = Xc.tocsr()
    indptr = Xc.indptr
    indices = Xc.indices
    data = Xc.data.astype(np.float64)
    nfeat = Xc.shape[1]

    csum = np.zeros(nfeat, dtype=np.float64)  # sum of the selected vectors
    state = {"ss": 0.0, "total": 0.0, "thr": float(SIM_THRESH)}
    chosen = []

    def row(j):
        a, b = indptr[j], indptr[j + 1]
        return indices[a:b], data[a:b]

    def accept(j, idx, val, dot):
        xx = float(val @ val) if len(val) else 0.0
        state["ss"] += 2.0 * dot + xx
        if len(idx):
            csum[idx] += val
        chosen.append(j)
        state["total"] += costs[j]

    def sim_of(j):
        idx, val = row(j)
        if len(idx) == 0:
            return idx, val, 0.0, 0.0
        dot = float(val @ csum[idx])
        xx = float(val @ val)
        den = math.sqrt(max(state["ss"], 1e-12) * max(xx, 1e-12))
        return idx, val, dot, dot / den

    def recheck(deferred):
        """Re-test deferred documents (in score order) against the current centroid."""
        if not deferred or state["ss"] <= 0:
            return deferred, 0
        sims = np.asarray(Xc @ csum).ravel() / math.sqrt(state["ss"])
        keep = []
        n_acc = 0
        for pos, j in enumerate(deferred):
            if state["total"] >= budget:
                keep.extend(deferred[pos:])
                break
            if sims[j] >= state["thr"] * 0.9:  # cheap prefilter (rows are ~unit norm)
                idx, val, dot, s = sim_of(j)
                if s >= state["thr"]:
                    accept(j, idx, val, dot)
                    n_acc += 1
                    continue
            keep.append(j)
        return keep, n_acc

    deferred = []
    since = 0
    for j in order:
        if state["total"] >= budget:
            break
        j = int(j)
        idx, val, dot, s = sim_of(j)
        if len(chosen) < SEED_N or s >= state["thr"]:
            accept(j, idx, val, dot)
            since += 1
        else:
            deferred.append(j)
        if since >= RECHECK_EVERY and deferred:
            since = 0
            deferred, _ = recheck(deferred)

    # budget not filled: keep re-testing, relaxing the threshold when stuck
    while state["total"] < budget and deferred:
        deferred, n_acc = recheck(deferred)
        if state["total"] >= budget or not deferred:
            break
        if n_acc == 0:
            state["thr"] *= RELAX
            if state["thr"] < 1e-3:
                for j in deferred:
                    if state["total"] >= budget:
                        break
                    chosen.append(j)
                    state["total"] += costs[j]
                break
    return chosen


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not target_sample:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # ---- 1. statistics of the target and the pool -------------------------------
    t_stats = np.array([_stats(t) for t in target_sample], dtype=np.float64)
    p_stats = np.array(
        [_stats(d) if isinstance(d, str) else (0, 0, 0, 1, 0) for d in docs],
        dtype=np.float64,
    )
    t_words = t_stats[:, 0]
    p_words = p_stats[:, 0]
    avg_t_words = float(t_words.mean())

    t_logw = np.log1p(t_words)
    mu_logw = float(t_logw.mean())
    sd_logw = float(max(t_logw.std(), 0.25))

    # thresholds taken from the target itself, with some margin
    min_words = max(15.0, min(0.25 * float(np.percentile(t_words, 2)), 0.1 * avg_t_words))
    alpha_lo = float(np.percentile(t_stats[:, 1], 1)) - 0.06
    uniq_lo = float(np.percentile(t_stats[:, 2], 1)) * 0.75
    z_lo = float(np.percentile(t_stats[:, 3], 1)) * 0.80
    z_hi = float(np.percentile(t_stats[:, 3], 99)) * 1.15
    spam_hi = max(float(np.percentile(t_stats[:, 4], 99)) * 2.0 + 2.0, 4.0)

    # ---- 2. junk filter and deduplication ---------------------------------------
    seen_full = set()
    seen_edge = set()
    cand = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        nw, alpha, uniq, zr, spam = p_stats[i]
        if nw < min_words:
            continue
        if alpha < alpha_lo or uniq < uniq_lo or zr < z_lo or zr > z_hi:
            continue
        if spam > spam_hi:
            continue
        if d.count("\ufffd") > 2:
            continue
        norm = _WS.sub(" ", d.lower()).strip()
        if not norm:
            continue
        hf = hash(norm)
        if hf in seen_full:
            continue
        seen_full.add(hf)
        he = hash((norm[:200], norm[-200:]))
        if he in seen_edge:
            continue
        seen_edge.add(he)
        cand.append(i)

    if not cand:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    cand_texts = [docs[i][:VEC_CHARS] for i in cand]

    # ---- 3. similarity to the target --------------------------------------------
    Xc = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            max_features=100000,
            dtype=np.float32,
        )
        t_texts = [t[:VEC_CHARS] for t in target_sample]
        X = vec.fit_transform(cand_texts + t_texts)
        nc = len(cand_texts)
        Xc = X[:nc]
        Xt = X[nc:]

        centroid = np.asarray(Xt.mean(axis=0)).ravel()
        cn = float(np.linalg.norm(centroid))
        if cn > 0:
            centroid = centroid / cn

        # local-centroid score (falls back to the global centroid if clustering fails)
        try:
            s_cent = _local_centroid_scores(Xc, centroid, seed)
        except Exception:
            s_cent = np.asarray(Xc @ centroid).ravel().astype(np.float64)

        # mean similarity to the 5 nearest target documents
        k = min(5, Xt.shape[0])
        s_knn = np.zeros(nc, dtype=np.float64)
        Xt_T = Xt.T.tocsr()
        step = 20000
        for a in range(0, nc, step):
            S = (Xc[a:a + step] @ Xt_T).toarray()
            if k < S.shape[1]:
                S = np.partition(S, S.shape[1] - k, axis=1)[:, -k:]
            s_knn[a:a + step] = S.mean(axis=1)

        def _z(v):
            sd = v.std()
            return (v - v.mean()) / sd if sd > 0 else v * 0.0

        score = 0.6 * _z(s_cent) + 0.4 * _z(s_knn)
    except Exception:
        Xc = None
        score = _fallback_scores(cand_texts, target_sample)
        sd = score.std()
        score = (score - score.mean()) / sd if sd > 0 else score * 0.0

    # mild penalties: lengths far from the target's, and spam-like patterns
    cand_arr = np.array(cand, dtype=np.int64)
    zlen = np.abs((np.log1p(p_words[cand_arr]) - mu_logw) / sd_logw)
    score = score - 0.15 * np.maximum(0.0, zlen - 2.0)
    score = score - 0.05 * p_stats[cand_arr, 4]

    # ---- 4. density-based greedy selection up to about T tokens ------------------
    sample_ids = rng.sample(cand, min(300, len(cand)))
    tpc = _tokens_per_char([docs[i] for i in sample_ids])
    budget = T * 1.03  # small margin because token counts are estimated

    order = np.argsort(-score, kind="stable")
    costs = np.array([len(docs[i]) * tpc + 1 for i in cand], dtype=np.float64)

    def _top_n():
        out = []
        total = 0.0
        for j in order:
            out.append(int(j))
            total += costs[int(j)]
            if total >= budget:
                break
        return out

    chosen_pos = None
    if Xc is not None:
        try:
            # only the best-scoring part of the pool takes part in the greedy pass
            cum = np.cumsum(costs[order])
            n_win = int(np.searchsorted(cum, budget * WINDOW, side="left")) + 1
            window = order[:n_win]
            chosen_pos = _density_greedy(Xc, window, costs, budget)
            if not chosen_pos:
                chosen_pos = None
        except Exception:
            chosen_pos = None
    if chosen_pos is None:
        chosen_pos = _top_n()

    chosen = [cand[j] for j in chosen_pos]
    rng.shuffle(chosen)
    return chosen

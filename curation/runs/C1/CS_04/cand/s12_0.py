"""Curator: diversity-aware max-similarity selection of pool documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Measure the word count and simple quality statistics of target_sample and the
     pool, and derive the junk thresholds from the target's own statistics.
  2. Drop damaged, junk and spam-like documents, and exact or near duplicates.
  3. Score each remaining document with the "local centroid" score on bag-of-words
     TF-IDF vectors and keep the top 10% as the shortlist.
     Local centroid score: the pool is divided into K clusters (K-Means on the
     TF-IDF vectors). Each cluster centroid gets a weight from its distance to the
     target centroid (closer -> larger weight). A document's score is the weighted
     average of its cosine similarities to the cluster centroids.
  4. Within the shortlist, greedily pick documents in order of "max-similarity"
     (the highest cosine similarity to any single target document, with mild
     length and spam penalties). A document is only accepted if its cosine
     similarity to the centroid of the already-selected documents is at least
     SIM_THRESHOLD. Picking stops at about T tokens or when the shortlist is
     exhausted (a shorter list is cycled by the trainer).
  5. Shuffle the chosen documents with the seed so training sees a stationary
     mixture.
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
TOP_FRAC = 0.10   # fraction of candidates shortlisted by the local-centroid score
SIM_THRESHOLD = 0.2  # minimum cosine similarity to the already-selected centroid


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


def _z(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


def _greedy_diverse(Xtop, order, costs, budget):
    """Greedy pick in the given order with a coherence constraint.

    Xtop: L2-normalised sparse CSR rows of the shortlist.
    order: shortlist positions sorted by decreasing max-similarity score.
    costs: estimated token cost of each shortlist document.
    A document is accepted only if its cosine similarity to the centroid of the
    already-selected documents is at least SIM_THRESHOLD.
    Returns the list of accepted shortlist positions.
    """
    Xtop = Xtop.tocsr()
    indptr, indices, data = Xtop.indptr, Xtop.indices, Xtop.data
    sumvec = np.zeros(Xtop.shape[1], dtype=np.float64)
    sum_sq = 0.0
    picked = []
    total = 0.0
    for j in order:
        j = int(j)
        s, e = indptr[j], indptr[j + 1]
        if e <= s:
            continue
        idx = indices[s:e]
        val = data[s:e].astype(np.float64)
        xsq = float(val @ val)
        if xsq <= 0.0:
            continue
        dot = float(val @ sumvec[idx])
        if picked:
            cos = dot / (math.sqrt(max(sum_sq, 1e-12)) * math.sqrt(xsq))
            if cos < SIM_THRESHOLD:
                continue
        picked.append(j)
        sumvec[idx] += val
        sum_sq += 2.0 * dot + xsq
        total += costs[j]
        if total >= budget:
            break
    return picked


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
    cand_arr = np.array(cand, dtype=np.int64)
    nc = len(cand)

    # token cost estimate and budget
    sample_ids = rng.sample(cand, min(300, len(cand)))
    tpc = _tokens_per_char([docs[i] for i in sample_ids])
    budget = T * 1.03  # small margin because token counts are estimated
    costs_all = np.array([len(docs[i]) * tpc + 1 for i in cand], dtype=np.float64)

    # mild penalties: lengths far from the target's, and spam-like patterns
    zlen = np.abs((np.log1p(p_words[cand_arr]) - mu_logw) / sd_logw)
    penalty = 0.15 * np.maximum(0.0, zlen - 2.0) + 0.05 * p_stats[cand_arr, 4]

    chosen = None

    # ---- 3 + 4. shortlist by local-centroid score, greedy max-similarity --------
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
        X = vec.fit_transform(cand_texts + t_texts).tocsr()
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

        # shortlist: top 10% by the local-centroid score
        n_top = int(max(1, min(nc, math.ceil(TOP_FRAC * nc))))
        top = np.argsort(-s_cent, kind="stable")[:n_top]
        Xtop = Xc[top]

        # max-similarity: highest similarity to any individual target document
        s_max = np.zeros(n_top, dtype=np.float64)
        Xt_T = Xt.T.tocsr()
        step = 20000
        for a in range(0, n_top, step):
            S = (Xtop[a:a + step] @ Xt_T).toarray()
            s_max[a:a + step] = S.max(axis=1)

        pick_score = _z(s_max) - penalty[top]
        order = np.argsort(-pick_score, kind="stable")

        picked = _greedy_diverse(Xtop, order, costs_all[top], budget)
        if picked:
            chosen = [cand[int(top[j])] for j in picked]
    except Exception:
        chosen = None

    # ---- fallback: plain keyword-overlap ranking up to the budget ---------------
    if not chosen:
        score = _fallback_scores(cand_texts, target_sample)
        score = _z(score) - penalty
        order = np.argsort(-score, kind="stable")
        chosen = []
        total = 0.0
        for j in order:
            j = int(j)
            chosen.append(cand[j])
            total += costs_all[j]
            if total >= budget:
                break

    # ---- 5. shuffle so training sees a stationary mixture -----------------------
    rng.shuffle(chosen)
    return chosen

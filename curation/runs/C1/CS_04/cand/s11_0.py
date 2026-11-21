"""Curator: rank the pool by similarity to the target sample and keep the best T tokens.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Measure the word count and simple quality statistics of target_sample and the
     pool, and derive the junk thresholds from the target's own statistics.
  2. Drop damaged, junk and spam-like documents, and exact or near duplicates.
  3. Vocabulary pre-filter: keep only documents for which at least 5% of their
     100 most frequent words occur in the vocabulary of target_sample. This prunes
     unrelated noise before any similarity is computed.
  4. Score each remaining document by bag-of-words TF-IDF similarity to the target:
     a Gaussian kernel on the cosine distance to the target centroid,
         exp(-d^2 / (2 sigma^2)),  d = 1 - cos(doc, target centroid),
     with sigma taken from the target documents' own distances to their centroid,
     plus the nearest target documents, with a mild penalty for lengths far from
     the target's.
  5. Keep the highest-scoring documents up to about T tokens, then shuffle them with
     the seed so training sees a stationary mixture.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import zlib
from collections import Counter

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
TOP_WORDS = 100   # most frequent words of a document used by the vocabulary filter
MIN_SHARED = 0.05  # minimum fraction of those words found in the target vocabulary
SIGMA_FLOOR = 0.05  # lower bound for the Gaussian kernel bandwidth


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


def _target_vocab(target_sample):
    """All distinct lower-cased words that occur in the target sample."""
    vocab = set()
    for t in target_sample:
        vocab.update(_WORD.findall(t.lower()))
    return vocab


def _shared_top_fraction(text, vocab):
    """Fraction of the document's TOP_WORDS most frequent words found in vocab."""
    ws = _WORD.findall(text.lower())
    if not ws:
        return 0.0
    top = Counter(ws).most_common(TOP_WORDS)
    hit = sum(1 for w, _ in top if w in vocab)
    return hit / len(top)


def _gaussian_kernel_scores(Xc, Xt, centroid):
    """Gaussian kernel on the cosine distance to the target centroid.

    Xc, Xt: L2-normalised sparse TF-IDF rows of the pool candidates / the target.
    centroid: unit-norm dense target centroid.
    The bandwidth sigma is the median cosine distance of the target documents to
    their own centroid, so a document as close as a typical target document
    scores exp(-0.5).
    """
    c = centroid.astype(np.float32)
    d_pool = 1.0 - np.asarray(Xc @ c).ravel().astype(np.float64)
    d_tgt = 1.0 - np.asarray(Xt @ c).ravel().astype(np.float64)
    sigma = float(np.median(d_tgt)) if d_tgt.size else 0.0
    if not np.isfinite(sigma):
        sigma = 0.0
    sigma = max(sigma, SIGMA_FLOOR)
    d_pool = np.maximum(d_pool, 0.0)
    return np.exp(-(d_pool * d_pool) / (2.0 * sigma * sigma))


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

    # ---- 3. vocabulary pre-filter -------------------------------------------------
    # keep documents that share at least 5% of their top-100 words with the target
    vocab = _target_vocab(target_sample)
    if vocab:
        kept = [
            i for i in cand
            if _shared_top_fraction(docs[i][:VEC_CHARS], vocab) >= MIN_SHARED
        ]
        if kept:  # never let the filter empty the pool
            cand = kept

    cand_texts = [docs[i][:VEC_CHARS] for i in cand]

    # ---- 4. similarity to the target --------------------------------------------
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

        # Gaussian kernel on the cosine distance to the target centroid
        s_cent = _gaussian_kernel_scores(Xc, Xt, centroid)

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
        score = _fallback_scores(cand_texts, target_sample)
        sd = score.std()
        score = (score - score.mean()) / sd if sd > 0 else score * 0.0

    # mild penalties: lengths far from the target's, and spam-like patterns
    cand_arr = np.array(cand, dtype=np.int64)
    zlen = np.abs((np.log1p(p_words[cand_arr]) - mu_logw) / sd_logw)
    score = score - 0.15 * np.maximum(0.0, zlen - 2.0)
    score = score - 0.05 * p_stats[cand_arr, 4]

    # ---- 5. keep the best documents up to about T tokens ------------------------
    sample_ids = rng.sample(cand, min(300, len(cand)))
    tpc = _tokens_per_char([docs[i] for i in sample_ids])
    budget = T * 1.03  # small margin because token counts are estimated

    order = np.argsort(-score, kind="stable")
    chosen = []
    total = 0.0
    for j in order:
        i = cand[int(j)]
        chosen.append(i)
        total += len(docs[i]) * tpc + 1
        if total >= budget:
            break

    rng.shuffle(chosen)
    return chosen

"""Curator: target-likeness filter using a bigram language model fitted to target_sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Tokenise target_sample and docs with tiktoken gpt2 (score on the first 2,000
     tokens per doc, record full token counts).
  2. Fit an interpolated bigram model on the target tokens:
     p = 0.7 * bigram + 0.3 * add-1 unigram.
  3. Score each doc by mean per-token negative log-likelihood.
  4. Drop duplicates (hash of normalised first 500 chars) and docs under 50 tokens.
  5. Threshold by 1-D 2-means on the scores, keep the low-NLL cluster, with a floor
     of 0.5 T tokens and a cap of T tokens (best-scoring first).
  6. Return the kept indices shuffled with CURATE_SEED.
"""
import os
import random

import numpy as np
import tiktoken
from sklearn.cluster import KMeans

from common import T

V = 50257
SCORE_CAP = 2000
MIN_TOKENS = 50
DEDUP_CHARS = 500
LAMBDA_BI = 0.7
LAMBDA_UNI = 0.3
TOK_BATCH = 1000
SCORE_BATCH_TOKENS = 2_000_000


def _fallback(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def _tokenise(enc, texts, cap=None):
    """Return (list of int64 arrays, possibly capped; full token counts)."""
    arrays = []
    lengths = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), TOK_BATCH):
        batch = [t if isinstance(t, str) else "" for t in texts[start:start + TOK_BATCH]]
        try:
            toks = enc.encode_ordinary_batch(batch, num_threads=8)
        except Exception:
            toks = []
            for t in batch:
                try:
                    toks.append(enc.encode_ordinary(t))
                except Exception:
                    toks.append([])
        for j, tk in enumerate(toks):
            lengths[start + j] = len(tk)
            if cap is not None and len(tk) > cap:
                tk = tk[:cap]
            arrays.append(np.asarray(tk, dtype=np.int64))
    return arrays, lengths


def _fit(target_arrays):
    uni = np.zeros(V, dtype=np.float64)
    ctx = np.zeros(V, dtype=np.float64)
    key_parts = []
    for a in target_arrays:
        if a.size == 0:
            continue
        uni += np.bincount(a, minlength=V)
        if a.size > 1:
            ctx += np.bincount(a[:-1], minlength=V)
            key_parts.append(a[:-1] * V + a[1:])
    if key_parts:
        keys, counts = np.unique(np.concatenate(key_parts), return_counts=True)
    else:
        keys = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.int64)
    n_total = uni.sum()
    p_uni = (uni + 1.0) / (n_total + V)
    return p_uni, ctx, keys, counts.astype(np.float64)


def _score_chunk(arrs, p_uni, ctx, keys, counts):
    """Mean per-token NLL for each array in arrs (all non-empty)."""
    lens = np.array([a.size for a in arrs], dtype=np.int64)
    toks = np.concatenate(arrs)
    n = toks.size
    doc_id = np.repeat(np.arange(len(arrs)), lens)

    bi = np.zeros(n, dtype=np.float64)
    if n > 1 and keys.size > 0:
        a = toks[:-1]
        b = toks[1:]
        same = doc_id[:-1] == doc_id[1:]
        q = a * V + b
        pos = np.searchsorted(keys, q)
        np.minimum(pos, keys.size - 1, out=pos)
        match = same & (keys[pos] == q)
        denom = np.maximum(ctx[a], 1.0)
        bi[1:] = np.where(match, counts[pos] / denom, 0.0)

    p = LAMBDA_BI * bi + LAMBDA_UNI * p_uni[toks]
    nll = -np.log(p)
    sums = np.bincount(doc_id, weights=nll, minlength=len(arrs))
    return sums / lens


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []
    if not target_sample:
        return _fallback(n_docs, seed)

    enc = tiktoken.get_encoding("gpt2")

    # 1-2. Tokenise the target sample and fit the bigram model.
    target_arrays, _ = _tokenise(enc, list(target_sample))
    if sum(a.size for a in target_arrays) == 0:
        return _fallback(n_docs, seed)
    p_uni, ctx, keys, counts = _fit(target_arrays)

    # 4a. Dedup on normalised first 500 characters (keep first occurrence).
    seen = set()
    unique = np.zeros(n_docs, dtype=bool)
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        k = hash(" ".join(d[:DEDUP_CHARS].lower().split()))
        if k in seen:
            continue
        seen.add(k)
        unique[i] = True

    # 1. Tokenise the pool (capped arrays for scoring, full lengths recorded).
    doc_arrays, full_len = _tokenise(enc, docs, cap=SCORE_CAP)

    # 4b. Length filter.
    valid = unique & (full_len >= MIN_TOKENS)
    cand = np.nonzero(valid)[0]
    if cand.size == 0:
        return _fallback(n_docs, seed)

    # 3. Score candidates in chunks.
    scores = np.empty(cand.size, dtype=np.float64)
    start = 0
    while start < cand.size:
        end = start
        tok_sum = 0
        while end < cand.size and (tok_sum < SCORE_BATCH_TOKENS or end == start):
            tok_sum += doc_arrays[cand[end]].size
            end += 1
        arrs = [doc_arrays[i] for i in cand[start:end]]
        scores[start:end] = _score_chunk(arrs, p_uni, ctx, keys, counts)
        start = end
    del doc_arrays

    # Sort candidates best (lowest NLL) first; stable for determinism.
    order = np.argsort(scores, kind="stable")
    cand = cand[order]
    scores = scores[order]
    cum = np.cumsum(full_len[cand])

    # 5. Threshold by 1-D 2-means; keep the low-NLL cluster.
    n_keep = cand.size
    if cand.size >= 2 and scores[-1] > scores[0]:
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=seed)
            labels = km.fit_predict(scores.reshape(-1, 1))
            low = int(np.argmin(km.cluster_centers_.ravel()))
            n_low = int(np.sum(labels == low))
            if n_low > 0:
                n_keep = n_low
        except Exception:
            n_keep = cand.size

    # Floor: at least 0.5 T tokens, widening to the best-scoring docs.
    half = 0.5 * T
    if cum[n_keep - 1] < half:
        pos = int(np.searchsorted(cum, half, side="left"))
        n_keep = min(pos + 1, cand.size)

    # Cap: at most T tokens' worth of the best-scoring docs.
    if cum[n_keep - 1] > T:
        pos = int(np.searchsorted(cum, T, side="left"))
        n_keep = min(pos + 1, n_keep)

    # 6. Shuffle the kept indices with the seed.
    kept = [int(i) for i in cand[:n_keep]]
    kept.sort()
    random.Random(seed).shuffle(kept)
    return kept

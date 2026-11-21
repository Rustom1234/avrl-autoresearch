"""Curator: two-pass (self-trained) target-likeness filter using a bigram language model.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Tokenise target_sample and docs with tiktoken gpt2 (score on the first 2,000
     tokens per doc, record full token counts).
  2. Drop duplicates (hash of normalised first 500 chars, or hash of tokens 100..300
     already seen) and docs under 50 tokens.
  3. Pass 1: fit an interpolated bigram model on the target tokens
     (p = 0.7 * bigram + 0.3 * add-1 unigram), score each doc by mean per-token NLL,
     and take the low-NLL cluster of a 1-D 2-means as pseudo-target.
  4. Pass 2: refit the same model on target_sample plus the first 2,000 tokens of every
     pseudo-target doc, and rescore every candidate leave-one-out (a doc in the fit set
     has its own bigram, context and unigram counts subtracted).
  5. Also compute the worst mean NLL over non-overlapping 256-token windows; reject docs
     whose worst window exceeds the pass-2 2-means threshold by more than 1.0 nat.
  6. Keep the pass-2 low-NLL cluster, with a floor of 0.5 T tokens and a cap of T tokens
     (best-scoring first).
  7. Return the kept indices shuffled with CURATE_SEED.
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
BODY_START = 100
BODY_END = 300
BODY_MIN = 50            # minimum tokens in the body slice for the body hash to apply
LAMBDA_BI = 0.7
LAMBDA_UNI = 0.3
TOK_BATCH = 1000
SCORE_BATCH_TOKENS = 2_000_000
WINDOW = 256
MIN_WINDOW = 128         # trailing partial windows shorter than this are ignored
WINDOW_MARGIN = 1.0      # nats above the pass-2 threshold


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


def _fit(arrays):
    """Raw counts for the interpolated bigram model: (uni, n_total, ctx, keys, counts)."""
    uni = np.zeros(V, dtype=np.float64)
    ctx = np.zeros(V, dtype=np.float64)
    key_parts = []
    for a in arrays:
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
    n_total = float(uni.sum())
    return uni, n_total, ctx, keys, counts.astype(np.float64)


def _own_counts(composite):
    """For each element, how many times its value occurs in the array."""
    _, inv, c = np.unique(composite, return_inverse=True, return_counts=True)
    return c[np.ravel(inv)].astype(np.float64)


def _score_chunk(arrs, model, in_fit=None):
    """Per-doc (mean NLL, worst 256-token-window mean NLL) for non-empty arrays.

    in_fit: optional bool array (one per doc); docs flagged True were part of the
    model's fit set and are scored leave-one-out.
    """
    uni, n_total, ctx, keys, counts = model
    n_docs = len(arrs)
    lens = np.array([a.size for a in arrs], dtype=np.int64)
    toks = np.concatenate(arrs)
    n = toks.size
    doc_id = np.repeat(np.arange(n_docs, dtype=np.int64), lens)
    loo = in_fit is not None and bool(np.any(in_fit))

    # Unigram part (add-1), leave-one-out where needed.
    u_num = uni[toks]
    u_den = np.full(n, n_total + V, dtype=np.float64)
    if loo:
        fit_tok = in_fit[doc_id]
        own_u = _own_counts(doc_id * V + toks)
        u_num = np.maximum(u_num - np.where(fit_tok, own_u, 0.0), 0.0)
        u_den = u_den - np.where(fit_tok, lens[doc_id].astype(np.float64), 0.0)
    p_u = (u_num + 1.0) / u_den

    # Bigram part, leave-one-out where needed.
    bi = np.zeros(n, dtype=np.float64)
    if n > 1 and keys.size > 0:
        pi = np.nonzero(doc_id[:-1] == doc_id[1:])[0]
        if pi.size > 0:
            a = toks[pi]
            b = toks[pi + 1]
            q = a * V + b
            pos = np.searchsorted(keys, q)
            np.minimum(pos, keys.size - 1, out=pos)
            match = keys[pos] == q
            num = np.where(match, counts[pos], 0.0)
            den = ctx[a].copy()
            if loo:
                d = doc_id[pi]
                fit_pair = in_fit[d]
                own_bi = _own_counts(d * (V * V) + q)
                own_ctx = _own_counts(d * V + a)
                num = num - np.where(fit_pair, own_bi, 0.0)
                den = den - np.where(fit_pair, own_ctx, 0.0)
            num = np.maximum(num, 0.0)
            bi[pi + 1] = num / np.maximum(den, 1.0)

    p = LAMBDA_BI * bi + LAMBDA_UNI * p_u
    nll = -np.log(p)
    means = np.bincount(doc_id, weights=nll, minlength=n_docs) / lens

    # Worst mean NLL over non-overlapping WINDOW-token windows.
    starts = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1:
        starts[1:] = np.cumsum(lens)[:-1]
    n_win = (lens + WINDOW - 1) // WINDOW
    win_off = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1:
        win_off[1:] = np.cumsum(n_win)[:-1]
    total_win = int(n_win.sum())
    pos_in_doc = np.arange(n, dtype=np.int64) - starts[doc_id]
    local_win = pos_in_doc // WINDOW
    gwin = win_off[doc_id] + local_win
    w_sum = np.bincount(gwin, weights=nll, minlength=total_win)
    w_cnt = np.bincount(gwin, minlength=total_win)
    w_mean = w_sum / np.maximum(w_cnt, 1)
    is_first = np.zeros(total_win, dtype=bool)
    is_first[win_off] = True
    w_valid = (w_cnt >= MIN_WINDOW) | is_first
    w_mean = np.where(w_valid, w_mean, -np.inf)
    worst = np.maximum.reduceat(w_mean, win_off)
    return means, worst


def _score_all(cand, doc_arrays, model, in_fit_global=None):
    """Score every candidate (in cand order) in token-bounded chunks."""
    scores = np.empty(cand.size, dtype=np.float64)
    worst = np.empty(cand.size, dtype=np.float64)
    start = 0
    while start < cand.size:
        end = start
        tok_sum = 0
        while end < cand.size and (tok_sum < SCORE_BATCH_TOKENS or end == start):
            tok_sum += doc_arrays[cand[end]].size
            end += 1
        ids = cand[start:end]
        arrs = [doc_arrays[i] for i in ids]
        fit_flags = in_fit_global[ids] if in_fit_global is not None else None
        scores[start:end], worst[start:end] = _score_chunk(arrs, model, fit_flags)
        start = end
    return scores, worst


def _two_means(scores, seed):
    """1-D 2-means. Returns (low-cluster mask, threshold) or (all True, inf) on failure."""
    all_true = np.ones(scores.size, dtype=bool)
    if scores.size < 2 or not np.all(np.isfinite(scores)):
        return all_true, float("inf")
    if not scores.max() > scores.min():
        return all_true, float("inf")
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=seed)
        labels = km.fit_predict(scores.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        low = int(np.argmin(centers))
        mask = labels == low
        if not mask.any():
            return all_true, float("inf")
        return mask, float(0.5 * (centers[0] + centers[1]))
    except Exception:
        return all_true, float("inf")


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []
    if not target_sample:
        return _fallback(n_docs, seed)

    enc = tiktoken.get_encoding("gpt2")

    # Tokenise the target sample and fit the pass-1 bigram model.
    target_arrays, _ = _tokenise(enc, list(target_sample))
    if sum(a.size for a in target_arrays) == 0:
        return _fallback(n_docs, seed)
    model1 = _fit(target_arrays)

    # Tokenise the pool (capped arrays for scoring, full lengths recorded).
    doc_arrays, full_len = _tokenise(enc, docs, cap=SCORE_CAP)

    # Dedup: normalised first 500 chars, plus body tokens 100..300 (keep first occurrence).
    seen_head = set()
    seen_body = set()
    unique = np.zeros(n_docs, dtype=bool)
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        dup = False
        k = hash(" ".join(d[:DEDUP_CHARS].lower().split()))
        if k in seen_head:
            dup = True
        else:
            seen_head.add(k)
        body = doc_arrays[i][BODY_START:BODY_END]
        if body.size >= BODY_MIN:
            kb = hash(body.tobytes())
            if kb in seen_body:
                dup = True
            else:
                seen_body.add(kb)
        if not dup:
            unique[i] = True

    # Length filter.
    valid = unique & (full_len >= MIN_TOKENS)
    cand = np.nonzero(valid)[0]
    if cand.size == 0:
        return _fallback(n_docs, seed)

    # Pass 1: score with the target-only model, 2-means, low cluster = pseudo-target.
    scores1, _ = _score_all(cand, doc_arrays, model1)
    low1, _ = _two_means(scores1, seed)
    pseudo = cand[low1]

    # Pass 2: refit on target + pseudo-target (first 2,000 tokens), rescore leave-one-out.
    in_fit = np.zeros(n_docs, dtype=bool)
    in_fit[pseudo] = True
    model2 = _fit(list(target_arrays) + [doc_arrays[i] for i in pseudo])
    scores2, worst2 = _score_all(cand, doc_arrays, model2, in_fit)
    del doc_arrays

    low2, thr2 = _two_means(scores2, seed)

    # Reject partly damaged docs: worst window well above the pass-2 threshold.
    ok = worst2 <= thr2 + WINDOW_MARGIN
    if not ok.any():
        ok = np.ones(cand.size, dtype=bool)

    pool = cand[ok]
    pool_scores = scores2[ok]
    pool_low = low2[ok]

    # Sort best (lowest NLL) first; stable for determinism.
    order = np.argsort(pool_scores, kind="stable")
    pool = pool[order]
    cum = np.cumsum(full_len[pool])

    n_keep = int(pool_low.sum())
    if n_keep <= 0:
        n_keep = pool.size

    # Floor: at least 0.5 T tokens, widening to the best-scoring docs.
    half = 0.5 * T
    if cum[n_keep - 1] < half:
        pos = int(np.searchsorted(cum, half, side="left"))
        n_keep = min(pos + 1, pool.size)

    # Cap: at most T tokens' worth of the best-scoring docs.
    if cum[n_keep - 1] > T:
        pos = int(np.searchsorted(cum, T, side="left"))
        n_keep = min(pos + 1, n_keep)

    # Shuffle the kept indices with the seed.
    kept = [int(i) for i in pool[:n_keep]]
    kept.sort()
    random.Random(seed).shuffle(kept)
    return kept

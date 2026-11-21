```python
"""Curator: two-pass (self-trained) target-likeness filter with weighted NLL.
Modified to use a weighted score (scores2 + 0.5 * worst2) for mean NLL filtering
while maintaining a hard cutoff on worst-window NLL.
"""
import os
import random
import sys

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
WINDOW_MARGIN = 1.0      # nats above the pass-2 2-means threshold (fallback rule only)
CAL_PERCENTILE = 97.5    # percentile of the target leave-one-out scores
CAL_MEAN_MARGIN = 0.05   # nats above the target percentile (mean NLL)
CAL_WORST_MARGIN = 0.15  # nats above the target percentile (worst window)
ALPHA = 0.5               # weight for worst-window penalty


def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


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


def _calibrate(target_capped, model):
    """Leave-one-out target scores under the pass-2 model -> (thr_mean, thr_worst)."""
    try:
        arrs = [a for a in target_capped if a.size > 0]
        if not arrs:
            return float("nan"), float("nan")
        in_fit = np.ones(len(arrs), dtype=bool)
        t_mean, t_worst = _score_chunk(arrs, model, in_fit)
        t_mean = t_mean[np.isfinite(t_mean)]
        t_worst = t_worst[np.isfinite(t_worst)]
        thr_mean = float("nan")
        thr_worst = float("nan")
        if t_mean.size > 0:
            thr_mean = float(np.percentile(t_mean, CAL_PERCENTILE)) + CAL_MEAN_MARGIN
        if t_worst.size > 0:
            thr_worst = float(np.percentile(t_worst, CAL_PERCENTILE)) + CAL_WORST_MARGIN
        return thr_mean, thr_worst
    except Exception:
        return float("nan"), float("nan")


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []
    if not target_sample:
        return _fallback(n_docs, seed)

    enc = tiktoken.get_encoding("gpt2")

    # Tokenise the target sample and fit the pass-1 bigram model (full target arrays).
    target_arrays, _ = _tokenise(enc, list(target_sample))
    if sum(a.size for a in target_arrays) == 0:
        return _fallback(n_docs, seed)
    model1 = _fit(target_arrays)
    # Capped copies for the pass-2 fit, so leave-one-out on a target doc is exact.
    target_capped = [a[:SCORE_CAP] for a in target_arrays]

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

    # Pass 2: refit on capped target + pseudo-target (first 2,000 tokens),
    # rescore candidates leave-one-out.
    in_fit = np.zeros(n_docs, dtype=bool)
    in_fit[pseudo] = True
    model2 = _fit(list(target_capped) + [doc_arrays[i] for i in pseudo])
    scores2, worst2 = _score_all(cand, doc_arrays, model2, in_fit)
    del doc_arrays

    # Calibration: how real target docs score under model2, leave-one-out.
    thr_mean, thr_worst = _calibrate(target_capped, model2)

    # Pass-2 2-means (kept as safety fallback and for logging).
    low2, thr2 = _two_means(scores2, seed)

    cand_len = full_len[cand]

    # Old rule (for logging and fallback).
    old_ok = worst2 <= thr2 + WINDOW_MARGIN
    if not old_ok.any():
        old_ok = np.ones(cand.size, dtype=bool)
    old_keep = low2
    old_sel = old_keep & old_ok
    old_docs = int(old_sel.sum())
    old_tokens = int(cand_len[old_sel].sum())
    
    calibrated = bool(np.isfinite(thr_mean))
    if calibrated:
        # Weighted score: favors content with lower worst-window NLL.
        scores2_weighted = scores2 + ALPHA * worst2
        keep_mask = scores2_weighted <= thr_mean
        worst_limit = thr_worst if np.isfinite(thr_worst) else thr2 + WINDOW_MARGIN
        ok = worst2 <= worst_

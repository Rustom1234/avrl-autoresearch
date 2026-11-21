"""Curator: two-pass (self-trained) target-likeness filter using a bigram language model.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Tokenise target_sample and docs with tiktoken gpt2. Full token arrays are kept;
     passes 1 and 2 score the first 2,000 tokens per doc.
  2. Cheap prefilter: drop duplicates (hash of normalised first 500 chars, or hash of
     tokens 100..300 already seen) and docs under 50 tokens.
  3. Pass 1: fit an interpolated bigram model on the target tokens
     (p = 0.7 * bigram + 0.3 * add-1 unigram), score each doc by mean per-token NLL,
     and take the low-NLL cluster of a 1-D 2-means as pseudo-target.
  4. Pass 2: refit the same model on target_sample plus the first 2,000 tokens of every
     pseudo-target doc, and rescore every candidate leave-one-out (a doc in the fit set
     has its own bigram, context and unigram counts subtracted). 2-means gives thr2.
  5. Full-length damage check: every candidate longer than 2,000 tokens with pass-2
     mean NLL <= thr2 + 0.5 is rescored over its full token array with the pass-2 model
     (leave-one-out, subtracting only the counts of its 2,000-token prefix, clipped
     at 0). Its window statistics are replaced by those of the full doc.
     Two window statistics over non-overlapping 256-token windows:
       - worst window mean NLL; reject if > thr2 + 1.0
       - fraction of windows with mean NLL > thr2 + 0.5; reject if > 0.2
     The rejection rule is applied to every candidate (docs that were not rescored use
     the windows of their scored prefix).
  6. Position-invariant near-duplicate removal: polynomial uint64 hashes of all token
     8-grams of the full doc, content-sampled (hash % 8 == 0). Docs are processed in
     ascending pass-2 NLL; a doc with >= 5 sampled shingles of which >= 40% were already
     seen in kept docs is dropped, otherwise it is kept and its shingles are added.
  7. Keep the pass-2 low-NLL cluster, with a floor of 0.5 T tokens and a cap of T tokens
     (best-scoring first).
  8. Return the kept indices shuffled with CURATE_SEED.

Docs/tokens removed by each step and the final K/T are logged to stderr.
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
WINDOW_MARGIN = 1.0      # nats above the pass-2 threshold (worst window)
FRAC_MARGIN = 0.5        # nats above the pass-2 threshold (bad-window definition)
FRAC_MAX = 0.2           # maximum tolerated fraction of bad windows
FULL_MARGIN = 0.5        # long docs with mean NLL <= thr2 + this get a full rescore

SHINGLE = 8
SHINGLE_P = 1000003
SHINGLE_MOD = 8
SHINGLE_MIN = 5
SHINGLE_OVERLAP = 0.4
_POW = [np.uint64(pow(SHINGLE_P, j, 1 << 64)) for j in range(SHINGLE)]


def _log(msg):
    print("[curate] " + msg, file=sys.stderr, flush=True)


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


def _own_counts(composite, in_prefix=None):
    """For each element, how many times its value occurs in the array.

    If in_prefix (bool mask) is given, occurrences are counted only among the
    masked elements, but the count is reported for every element (0 if absent).
    """
    if in_prefix is None:
        _, inv, c = np.unique(composite, return_inverse=True, return_counts=True)
        return c[np.ravel(inv)].astype(np.float64)
    src = composite[in_prefix]
    if src.size == 0:
        return np.zeros(composite.size, dtype=np.float64)
    uk, uc = np.unique(src, return_counts=True)
    pos = np.searchsorted(uk, composite)
    np.minimum(pos, uk.size - 1, out=pos)
    return np.where(uk[pos] == composite, uc[pos], 0).astype(np.float64)


def _score_chunk(arrs, model, in_fit=None, own_cap=None):
    """Per-doc scores for non-empty arrays.

    Returns (mean NLL, worst window mean NLL, window means, windows per doc).
    Window means are over non-overlapping WINDOW-token windows, concatenated in doc
    order; ignored (too short) windows are -inf.

    in_fit: optional bool array (one per doc); docs flagged True were part of the
    model's fit set and are scored leave-one-out.
    own_cap: if given, only the first own_cap tokens of each flagged doc were in the
    fit set; own counts are taken from that prefix and applied to the whole doc.
    """
    uni, n_total, ctx, keys, counts = model
    n_docs = len(arrs)
    lens = np.array([a.size for a in arrs], dtype=np.int64)
    toks = np.concatenate(arrs)
    n = toks.size
    doc_id = np.repeat(np.arange(n_docs, dtype=np.int64), lens)
    loo = in_fit is not None and bool(np.any(in_fit))

    starts = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1:
        starts[1:] = np.cumsum(lens)[:-1]
    pos_in_doc = np.arange(n, dtype=np.int64) - starts[doc_id]

    # Unigram part (add-1), leave-one-out where needed.
    u_num = uni[toks]
    u_den = np.full(n, n_total + V, dtype=np.float64)
    if loo:
        fit_tok = in_fit[doc_id]
        if own_cap is None:
            tok_pref = None
            fit_len = lens
        else:
            tok_pref = pos_in_doc < own_cap
            fit_len = np.minimum(lens, own_cap)
        own_u = _own_counts(doc_id * V + toks, tok_pref)
        u_num = np.maximum(u_num - np.where(fit_tok, own_u, 0.0), 0.0)
        u_den = u_den - np.where(fit_tok, fit_len[doc_id].astype(np.float64), 0.0)
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
                # A pair was in the fit set iff its second token lies in the prefix.
                pair_pref = None if own_cap is None else (pos_in_doc[pi + 1] < own_cap)
                own_bi = _own_counts(d * (V * V) + q, pair_pref)
                own_ctx = _own_counts(d * V + a, pair_pref)
                num = num - np.where(fit_pair, own_bi, 0.0)
                den = den - np.where(fit_pair, own_ctx, 0.0)
            num = np.maximum(num, 0.0)
            bi[pi + 1] = num / np.maximum(den, 1.0)

    p = LAMBDA_BI * bi + LAMBDA_UNI * p_u
    nll = -np.log(p)
    means = np.bincount(doc_id, weights=nll, minlength=n_docs) / lens

    # Mean NLL over non-overlapping WINDOW-token windows.
    n_win = (lens + WINDOW - 1) // WINDOW
    win_off = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1:
        win_off[1:] = np.cumsum(n_win)[:-1]
    total_win = int(n_win.sum())
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
    return means, worst, w_mean, n_win


def _score_all(ids_all, arrays, model, in_fit_global=None, own_cap=None):
    """Score every doc in ids_all (in order) in token-bounded chunks.

    Returns (mean NLL, worst window, concatenated window means, windows per doc).
    """
    m = len(ids_all)
    scores = np.empty(m, dtype=np.float64)
    worst = np.empty(m, dtype=np.float64)
    n_win = np.zeros(m, dtype=np.int64)
    w_parts = []
    start = 0
    while start < m:
        end = start
        tok_sum = 0
        while end < m and (tok_sum < SCORE_BATCH_TOKENS or end == start):
            tok_sum += arrays[ids_all[end]].size
            end += 1
        ids = ids_all[start:end]
        arrs = [arrays[i] for i in ids]
        fit_flags = in_fit_global[ids] if in_fit_global is not None else None
        s, w, wm, nw = _score_chunk(arrs, model, fit_flags, own_cap)
        scores[start:end] = s
        worst[start:end] = w
        n_win[start:end] = nw
        w_parts.append(wm)
        start = end
    w_mean = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=np.float64)
    return scores, worst, w_mean, n_win


def _frac_above(w_mean, n_win, thr):
    """Per-doc fraction of counted windows whose mean NLL exceeds thr."""
    if n_win.size == 0:
        return np.zeros(0, dtype=np.float64)
    offs = np.zeros(n_win.size, dtype=np.int64)
    if n_win.size > 1:
        offs[1:] = np.cumsum(n_win)[:-1]
    valid = w_mean > -np.inf
    bad = valid & (w_mean > thr)
    n_bad = np.add.reduceat(bad.astype(np.int64), offs)
    n_valid = np.add.reduceat(valid.astype(np.int64), offs)
    return n_bad / np.maximum(n_valid, 1)


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


def _shingles(arr):
    """Set of content-sampled polynomial hashes of all token 8-grams of arr."""
    n = arr.size - SHINGLE + 1
    if n <= 0:
        return set()
    a = arr.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    for j in range(SHINGLE):
        h += a[j:j + n] * _POW[j]       # uint64 wraparound
    sampled = h[(h % np.uint64(SHINGLE_MOD)) == 0]
    return set(sampled.tolist())


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

    # Tokenise the pool: full arrays kept, capped views used for passes 1 and 2.
    full_arrays, full_len = _tokenise(enc, docs)
    doc_arrays = [a[:SCORE_CAP] for a in full_arrays]
    _log("pool: %d docs, %d tokens (T = %d)" % (n_docs, int(full_len.sum()), T))

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
    del seen_head, seen_body
    _log("head/body dedup: removed %d docs, %d tokens"
         % (int((~unique).sum()), int(full_len[~unique].sum())))

    # Length filter.
    valid = unique & (full_len >= MIN_TOKENS)
    short = unique & ~valid
    _log("length filter (<%d tokens): removed %d docs, %d tokens"
         % (MIN_TOKENS, int(short.sum()), int(full_len[short].sum())))
    cand = np.nonzero(valid)[0]
    if cand.size == 0:
        return _fallback(n_docs, seed)

    # Pass 1: score with the target-only model, 2-means, low cluster = pseudo-target.
    scores1, _, _, _ = _score_all(cand, doc_arrays, model1)
    low1, _ = _two_means(scores1, seed)
    pseudo = cand[low1]

    # Pass 2: refit on target + pseudo-target (first 2,000 tokens), rescore leave-one-out.
    in_fit = np.zeros(n_docs, dtype=bool)
    in_fit[pseudo] = True
    model2 = _fit(list(target_arrays) + [doc_arrays[i] for i in pseudo])
    scores2, worst2, wmean2, nwin2 = _score_all(cand, doc_arrays, model2, in_fit)

    low2, thr2 = _two_means(scores2, seed)
    frac_thr = thr2 + FRAC_MARGIN
    frac2 = _frac_above(wmean2, nwin2, frac_thr)
    del wmean2, nwin2

    # Full-length damage check for plausible long docs.
    sel = np.nonzero((full_len[cand] > SCORE_CAP) & (scores2 <= thr2 + FULL_MARGIN))[0]
    if sel.size > 0:
        _, worst_f, wmean_f, nwin_f = _score_all(
            cand[sel], full_arrays, model2, in_fit, own_cap=SCORE_CAP)
        worst2[sel] = worst_f
        frac2[sel] = _frac_above(wmean_f, nwin_f, frac_thr)
        del wmean_f, nwin_f
    _log("full-length rescore: %d docs, %d tokens"
         % (int(sel.size), int(full_len[cand[sel]].sum()) if sel.size else 0))
    del doc_arrays, model1, model2

    # Reject damaged docs: worst window well above threshold, or too many bad windows.
    bad_worst = worst2 > thr2 + WINDOW_MARGIN
    bad_frac = (frac2 > FRAC_MAX) & ~bad_worst
    ok = ~(bad_worst | bad_frac)
    if not ok.any():
        ok = np.ones(cand.size, dtype=bool)
        _log("damage check: rejected everything, check disabled")
    else:
        _log("damage check (worst window): removed %d docs, %d tokens"
             % (int(bad_worst.sum()), int(full_len[cand[bad_worst]].sum())))
        _log("damage check (bad-window fraction): removed %d docs, %d tokens"
             % (int(bad_frac.sum()), int(full_len[cand[bad_frac]].sum())))

    pool = cand[ok]
    pool_scores = scores2[ok]
    pool_low = low2[ok]

    # Sort best (lowest NLL) first; stable for determinism.
    order = np.argsort(pool_scores, kind="stable")
    pool = pool[order]
    pool_low = pool_low[order]

    # Position-invariant near-duplicate removal, best-scoring docs first.
    seen = set()
    keep = np.ones(pool.size, dtype=bool)
    for j in range(pool.size):
        s = _shingles(full_arrays[pool[j]])
        if len(s) >= SHINGLE_MIN:
            hits = len(s & seen)
            if hits >= SHINGLE_OVERLAP * len(s):
                keep[j] = False
                continue
        seen |= s
    del seen, full_arrays
    _log("shingle near-dup: removed %d docs, %d tokens"
         % (int((~keep).sum()), int(full_len[pool[~keep]].sum())))

    pool = pool[keep]
    pool_low = pool_low[keep]
    if pool.size == 0:
        return _fallback(n_docs, seed)
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

    _log("cluster/floor/cap: removed %d docs, %d tokens"
         % (int(pool.size - n_keep), int(cum[-1] - cum[n_keep - 1])))
    k_tokens = int(cum[n_keep - 1])
    _log("final: %d docs, K = %d tokens, K/T = %.4f"
         % (n_keep, k_tokens, k_tokens / float(T)))

    # Shuffle the kept indices with the seed.
    kept = [int(i) for i in pool[:n_keep]]
    kept.sort()
    random.Random(seed).shuffle(kept)
    return kept

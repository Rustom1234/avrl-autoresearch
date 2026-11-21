"""Curator: two-pass (self-trained) target-likeness filter using a bigram language model,
with clean off-domain filler in front of the core when the core is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Core selection:
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
     (best-scoring first).  This is the core, with K tokens.

Output:
  * K >= T: the core only, shuffled with CURATE_SEED.
  * K <  T: filler (seeded shuffle), then the core (seeded shuffle), then the 200
    best-scoring core docs again as padding.  Filler candidates are deduplicated docs
    of at least 50 tokens outside the core, taken in ascending pass-2 mean NLL, that
    pass three junk guards:
      (a) at least 80% of the lowercased alphabetic words in the first 2,000 characters
          are in a vocabulary built from target_sample plus the core docs;
      (b) zlib compression ratio of the first 4,000 characters >= 0.35, and none of the
          50 highest-document-frequency word 6-grams (first 300 words per doc, only
          6-grams in more than 1% of docs) occurs in the doc's first 300 words;
      (c) pass-2 mean NLL below the midpoint between the 2-means high-cluster centre
          and the threshold (skipped if it leaves under 0.2 T tokens of filler).
    Filler stops once its tokens reach T - K - 20,000.

Interpretation notes:
  * The 6-gram document frequencies are counted over the filler candidates (the
    deduplicated, >= 50-token non-core docs), so exact duplicates do not inflate them.
  * The compression ratio is compressed bytes over UTF-8 bytes of the 4,000-char prefix.
  * A filler doc that would push the filler total past T - K is skipped, so filler
    never pushes core docs beyond the T-token budget.
"""
import os
import random
import re
import sys
import zlib

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

# Filler settings.
FILLER_SLACK = 20_000    # stop filler once it reaches T - K - FILLER_SLACK tokens
VOCAB_CHARS = 2000       # chars examined by the vocabulary guard
VOCAB_MIN_FRAC = 0.8
ZLIB_CHARS = 4000
ZLIB_MIN_RATIO = 0.35
GRAM_N = 6
GRAM_WORDS = 300
GRAM_CHARS = 8000        # prefix that is split to obtain the first GRAM_WORDS words
GRAM_TOP = 50
GRAM_MIN_DF_FRAC = 0.01
GUARD_C_MIN_FRAC = 0.2   # guard (c) is skipped if it leaves under this fraction of T
PAD_DOCS = 200

_WORD_RE = re.compile(r"[^\W\d_]+")
_HASH_P = np.uint64(1099511628211)
_HASH_Q = np.uint64(0x9E3779B97F4A7C15)
_ONE = np.uint64(1)


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
    """1-D 2-means.

    Returns (low-cluster mask, threshold, high-cluster centre), or
    (all True, inf, inf) on failure.
    """
    all_true = np.ones(scores.size, dtype=bool)
    fail = (all_true, float("inf"), float("inf"))
    if scores.size < 2 or not np.all(np.isfinite(scores)):
        return fail
    if not scores.max() > scores.min():
        return fail
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=seed)
        labels = km.fit_predict(scores.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        low = int(np.argmin(centers))
        mask = labels == low
        if not mask.any():
            return fail
        return mask, float(0.5 * (centers[0] + centers[1])), float(np.max(centers))
    except Exception:
        return fail


def _build_vocab(docs, target_sample, core_ids):
    """Lowercased alphabetic words of target_sample plus the core docs."""
    vocab = set()
    for t in target_sample:
        if isinstance(t, str):
            vocab.update(_WORD_RE.findall(t.lower()))
    for i in core_ids:
        vocab.update(_WORD_RE.findall(docs[int(i)].lower()))
    return vocab


def _filler_guards(docs, nc, vocab):
    """Guards (a) and (b) for the non-core candidates nc. Returns a bool array."""
    n = nc.size
    ok_vocab = np.zeros(n, dtype=bool)
    ok_zlib = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=np.int64)
    gram_parts = []
    word_id = {}
    span = GRAM_N - 1

    for j in range(n):
        d = docs[int(nc[j])]

        # (a) scrambled text: share of in-vocabulary alphabetic words.
        words = _WORD_RE.findall(d[:VOCAB_CHARS].lower())
        if words:
            hits = sum(1 for w in words if w in vocab)
            ok_vocab[j] = hits >= VOCAB_MIN_FRAC * len(words)

        # (b1) template spam: compression ratio.
        raw = d[:ZLIB_CHARS].encode("utf-8", "replace")
        if raw:
            ok_zlib[j] = len(zlib.compress(raw)) / len(raw) >= ZLIB_MIN_RATIO

        # (b2) word 6-grams of the first 300 words, hashed to uint64.
        toks = d[:GRAM_CHARS].lower().split()[:GRAM_WORDS]
        if len(toks) >= GRAM_N:
            ids = np.fromiter(
                (word_id.setdefault(w, len(word_id)) for w in toks),
                dtype=np.uint64, count=len(toks))
            m = ids.size - span
            h = np.zeros(m, dtype=np.uint64)
            for k in range(GRAM_N):
                h = h * _HASH_P + (ids[k:k + m] + _ONE) * _HASH_Q
            h = np.unique(h)
            gram_parts.append(h)
            sizes[j] = h.size

    has_banned = np.zeros(n, dtype=bool)
    if gram_parts:
        all_h = np.concatenate(gram_parts)
        del gram_parts
        keys, counts = np.unique(all_h, return_counts=True)
        eligible = counts > GRAM_MIN_DF_FRAC * n
        if eligible.any():
            e_keys = keys[eligible]
            e_counts = counts[eligible]
            top = np.argsort(-e_counts, kind="stable")[:GRAM_TOP]
            banned = e_keys[top]
            hit = np.isin(all_h, banned)
            doc_of = np.repeat(np.arange(n, dtype=np.int64), sizes)
            has_banned = np.bincount(doc_of[hit], minlength=n) > 0

    return ok_vocab & ok_zlib & ~has_banned


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
    low1, _, _ = _two_means(scores1, seed)
    pseudo = cand[low1]

    # Pass 2: refit on target + pseudo-target (first 2,000 tokens), rescore leave-one-out.
    in_fit = np.zeros(n_docs, dtype=bool)
    in_fit[pseudo] = True
    model2 = _fit(list(target_arrays) + [doc_arrays[i] for i in pseudo])
    scores2, worst2 = _score_all(cand, doc_arrays, model2, in_fit)
    del doc_arrays

    low2, thr2, high2 = _two_means(scores2, seed)

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

    core_best = pool[:n_keep]                 # best-scoring first
    K = int(cum[n_keep - 1])

    # Core in seeded shuffle (same order as before in both branches).
    kept = [int(i) for i in core_best]
    kept.sort()
    random.Random(seed).shuffle(kept)

    if K >= T:
        _log("core K=%d tokens, T=%d, K/T=%.4f, core docs=%d, filler tokens=0 (docs=0)"
             % (K, T, K / float(T), len(kept)))
        return kept

    # ---- K < T: fill the gap with clean off-domain filler instead of cycling. ----
    need = T - K
    budget = need - FILLER_SLACK
    filler = []
    filler_tok = 0
    guard_c_used = False

    if budget > 0:
        core_mask = np.zeros(n_docs, dtype=bool)
        core_mask[core_best] = True
        nc_sel = ~core_mask[cand]
        nc = cand[nc_sel]
        nc_scores = scores2[nc_sel]

        if nc.size > 0:
            vocab = _build_vocab(docs, target_sample, core_best)
            passes = _filler_guards(docs, nc, vocab)
            del vocab

            # (c) pass-2 mean NLL below the midpoint of high-cluster centre and threshold.
            if np.isfinite(thr2) and np.isfinite(high2):
                limit = 0.5 * (high2 + thr2)
                with_c = passes & (nc_scores < limit)
                if float(full_len[nc[with_c]].sum()) >= GUARD_C_MIN_FRAC * T:
                    passes = with_c
                    guard_c_used = True

            for j in np.argsort(nc_scores, kind="stable"):
                if filler_tok >= budget:
                    break
                if not passes[j]:
                    continue
                n_tok = int(full_len[nc[j]])
                if filler_tok + n_tok > need:
                    continue      # would push core docs past the T-token budget
                filler.append(int(nc[j]))
                filler_tok += n_tok

    filler.sort()
    random.Random(seed + 1).shuffle(filler)

    padding = [int(i) for i in core_best[:PAD_DOCS]]

    _log("core K=%d tokens, T=%d, K/T=%.4f, core docs=%d, filler tokens=%d (docs=%d, "
         "target %d, guard_c=%s), padding docs=%d"
         % (K, T, K / float(T), len(kept), filler_tok, len(filler),
            max(budget, 0), "on" if guard_c_used else "off", len(padding)))

    return filler + kept + padding

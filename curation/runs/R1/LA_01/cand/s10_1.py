import os
import random
import sys
import numpy as np
import tiktoken
from sklearn.cluster import KMeans
from common import T

# Constants
V = 50257
SCORE_CAP = 2000
MIN_TOKENS = 50
DEDUP_CHARS = 500
BODY_START = 100
BODY_END = 300
BODY_MIN = 50
LAMBDA_BI = 0.7
LAMBDA_UNI = 0.3
TOK_BATCH = 1000
SCORE_BATCH_TOKENS = 2_000_000
WINDOW = 256
MIN_WINDOW = 128
CAL_PERCENTILE = 97.5
CAL_MEAN_MARGIN = 0.05
CAL_WORST_MARGIN = 0.15

def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except:
        pass

def _fallback(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx

def _tokenise(enc, texts, cap=None):
    arrays = []
    lengths = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), TOK_BATCH):
        batch = [t if isinstance(t, str) else "" for t in texts[start:start + TOK_BATCH]]
        try:
            toks = enc.encode_ordinary_batch(batch, num_threads=8)
        except:
            toks = [enc.encode_ordinary(t) for t in batch]
        for j, tk in enumerate(toks):
            lengths[start + j] = len(tk)
            if cap is not None and len(tk) > cap:
                tk = tk[:cap]
            arrays.append(np.asarray(tk, dtype=np.int64))
    return arrays, lengths

def _fit(arrays):
    uni = np.zeros(V, dtype=np.float64)
    ctx = np.zeros(V, dtype=np.float64)
    key_parts = []
    for a in arrays:
        if a.size == 0: continue
        uni += np.bincount(a, minlength=V)
        if a.size > 1:
            ctx += np.bincount(a[:-1], minlength=V)
            key_parts.append(a[:-1] * V + a[1:])
    if key_parts:
        keys, counts = np.unique(np.concatenate(key_parts), return_counts=True)
        keys = keys.astype(np.int64)
        counts = counts.astype(np.float64)
    else:
        keys = np.zeros(0, dtype=np.int64)
        counts = np.zeros(0, dtype=np.float64)
    return uni, float(uni.sum()), ctx, keys, counts

def _own_counts(composite):
    _, inv, c = np.unique(composite, return_inverse=True, return_counts=True)
    return c[np.ravel(inv)].astype(np.float64)

def _score_chunk(arrs, model, in_fit=None):
    uni, n_total, ctx, keys, counts = model
    n_docs = len(arrs)
    lens = np.array([a.size for a in arrs], dtype=np.int64)
    toks = np.concatenate(arrs)
    n = toks.size
    doc_id = np.repeat(np.arange(n_docs, dtype=np.int64), lens)
    
    u_num = uni[toks]
    u_den = np.full(n, n_total + V, dtype=np.float64)
    
    if in_fit is not None and np.any(in_fit):
        fit_tok = in_fit[doc_id]
        own_u = _own_counts(doc_id * V + toks)
        u_num = np.maximum(u_num - np.where(fit_tok, own_u, 0.0), 0.0)
        u_den = u_den - np.where(fit_tok, lens[doc_id].astype(np.float64), 0.0)
    
    p_u = (u_num + 1.0) / u_den
    bi = np.zeros(n, dtype=np.float64)
    
    if n > 1 and keys.size > 0:
        pi = np.nonzero(doc_id[:-1] == doc_id[1:])[0]
        if pi.size > 0:
            a = toks[pi]
            b = toks[pi + 1]
            q = a * V + b
            pos = np.searchsorted(keys, q)
            pos = np.minimum(pos, keys.size - 1)
            match = keys[pos] == q
            num = np.where(match, counts[pos], 0.0)
            den = ctx[a].copy()
            if in_fit is not None and np.any(in_fit):
                d = doc_id[pi]
                fit_pair = in_fit[d]
                own_bi = _own_counts(d * (V * V) + q)
                own_ctx = _own_counts(d * V + a)
                num = num - np.where(fit_pair, own_bi, 0.0)
                den = den - np.where(fit_pair, own_ctx, 0.0)
            num = np.maximum(num, 0.0)
            bi[pi + 1] = num / np.maximum(den, 1.0)

    p = LAMBDA_BI * bi + LAMBDA_UNI * p_u
    nll = -np.log(np.maximum(p, 1e-9))
    means = np.bincount(doc_id, weights=nll, minlength=n_docs) / lens
    
    starts = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1: starts[1:] = np.cumsum(lens)[:-1]
    n_win = (lens + WINDOW - 1) // WINDOW
    win_off = np.zeros(n_docs, dtype=np.int64)
    if n_docs > 1: win_off[1:] = np.cumsum(n_win)[:-1]
    
    total_win = int(n_win.sum())
    pos_in_doc = np.arange(n) - starts[doc_id]
    local_win = pos_in_doc // WINDOW
    gwin = win_off[doc_id] + local_win
    w_sum = np.bincount(gwin, weights=nll, minlength=total_win)
    w_cnt = np.bincount(gwin, minlength=total_win)
    w_mean = w_sum / np.maximum(w_cnt, 1)
    is_first = np.zeros(total_win, dtype=bool)
    is_first[win_off] = True
    w_mean = np.where((w_cnt >= MIN_WINDOW) | is_first, w_mean, -1e9)
    worst = np.maximum.reduceat(w_mean, win_off)
    return means, worst

def _score_all(cand, doc_arrays, model, in_fit_global=None):
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
    if scores.size < 2 or not np.any(np.isfinite(scores)): return np.ones(scores.size, dtype=bool), float("inf")
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=seed)
        labels = km.fit_predict(scores.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        low = int(np.argmin(centers))
        return (labels == low), float(0.5 * (centers[0] + centers[1]))
    except:
        return np.ones(scores.size, dtype=bool), float("inf")

def _calibrate(target_capped, model):
    arrs = [a for a in target_capped if a.size > 0]
    if not arrs: return float("nan"), float("nan")
    in_fit = np.ones(len(arrs), dtype=bool)
    t_mean, t_worst = _score_chunk(arrs, model, in_fit)
    t_mean = t_mean[np.isfinite(t_mean)]
    t_worst = t_worst[np.isfinite(t_worst)]
    thr_mean = float(np.percentile(t_mean, CAL_PERCENTILE)) + CAL_MEAN_MARGIN if t_mean.size > 0 else float("nan")
    thr_worst = float(np.percentile(t_worst, CAL_PERCENTILE)) + CAL_WORST_MARGIN if t_worst.size > 0 else float("nan")
    return thr_mean, thr_worst

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0: return []
    if not target_sample: return _fallback(n_docs, seed)

    enc = tiktoken.get_encoding("gpt2")
    target_arrays, _ = _tokenise(enc, list(target_sample))
    if sum(a.size for a in target_arrays) == 0: return _fallback(n_docs, seed)
    
    model1 = _fit(target_arrays)
    target_capped = [a[:SCORE_CAP] for a in target_arrays]
    doc_arrays, full_len = _tokenise(enc, docs, cap=SCORE_CAP)

    unique = np.zeros(n_docs, dtype=bool)
    seen_head = set()
    seen_body = set()
    for i, d in enumerate(docs):
        if not isinstance(d, str): continue
        h = hash(" ".join(d[:DEDUP_CHARS].lower().split()))
        if h not in seen_head:
            seen_head.add(h)
            body = doc_arrays[i][BODY_START:BODY_END]
            if body.size >= BODY_MIN:
                kb = hash(body.tobytes())
                if kb not in seen_body:
                    seen_body.add(kb)
                    unique[i] = True
            else:
                unique[i] = True
        else:
            unique[i] = False

    valid = unique & (full_len >= MIN_TOKENS)
    cand = np.nonzero(valid)[0]
    if cand.size == 0: return _fallback(n_docs, seed)

    scores1, _ = _score_all(cand, doc_arrays, model1)
    low1, _ = _two_means(scores1, seed)
    
    in_fit = np.zeros(n_docs, dtype=bool)
    in_fit[cand[low1]] = True
    model2 = _fit(list(target_capped) + [doc_arrays[i] for i in cand[low1]])
    scores2, worst2 = _score_all(cand, doc_arrays, model2, in_fit)
    
    thr_mean, thr_worst = _calibrate(target_capped, model2)
    low2, _ = _two_means(scores2, seed)
    
    pool_idx = cand[low2]
    if pool_idx.size > 0:
        p_scores, _ = _score_chunk([doc_arrays[i] for i in pool_idx], model2, in_fit[pool_idx])
        var_limit = 5.0 # Simplified heuristic
        # Filter for structural coherence
        keep_mask = np.ones(cand.size, dtype=bool)
        bad_indices = pool_idx[p_scores > (thr_mean if not np.isnan(thr_mean) else 100)]
        keep_mask[bad_indices] = False
        final_cand = cand[keep_mask]
    else:
        final_cand = cand

    indices = final_cand.tolist()
    if len(indices) > 0:
        # Sample to fill T tokens approximately
        num_needed = int(T / (np.mean([doc_arrays[i].size for i in indices]) + 1))
        if num_needed > len(indices):
            # Cycle/Repeat to fill T
            full_list = (indices * (num_needed // len(indices) + 1))[:num_needed]
            return full_list
        else:
            # Select top N based on low2 scores (proxied by original scores2)
            sorted_indices = np.argsort(scores2[final_cand])[::-1]
            return [final_cand[i] for i in sorted_indices[:num_needed]]
    
    return _fallback(n_docs, seed)

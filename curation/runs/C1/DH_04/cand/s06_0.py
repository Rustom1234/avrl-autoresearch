"""Target-similarity curator (purely in-domain stream).

select(docs, target_sample) -> ordered list of indices into docs.

1. Drop junk: exact duplicates, very short docs, docs with a low share of
   alphabetic/space characters, highly compressible (templated spam) docs,
   and docs with damaged word order (words shuffled inside sentences or
   letters scrambled inside words). The word-order filter measures the share
   of a doc's adjacent in-vocabulary token pairs that also occur in
   target_sample; its threshold is 0.85 * the 2nd percentile of 5-fold
   cross-validated hit rates of the target docs.
2. Score every pool doc with a char n-gram TF-IDF + logistic regression model
   (target_sample = positives, seeded random pool sample = negatives).
3. In-domain threshold thr = 5th percentile of 5-fold cross-validated
   decision scores of the target_sample docs (each fold trained on the other
   targets plus the same seeded negatives).
4. D = surviving docs with score >= thr.
   - tok(D) >= T: take the best-scoring docs until about T tokens, shuffle.
   - 0 < tok(D) < T: k = ceil(T / tok(D)) copies of D (no upper limit on k),
     each shuffled independently with the seeded RNG, concatenated and
     truncated at about T tokens. No doc with score < thr is ever emitted.
   - Safety fallback only: if tok(D) < 0.05 * T, thr is lowered to the score
     that gives the top-scoring survivors totalling 0.05 * T tokens, and that
     set is repeated instead.
5. STATS (module-level dict) records which regime the pool is in:
   tok(D)/T, the number of docs in D, k, and the number of filler tokens the
   previous logic (max 3 copies + below-threshold filler) would have added.
"""
import hashlib
import math
import os
import random
import re
import zlib
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

TRUNC_CHARS = 3000        # chars per doc seen by the classifier
MAX_FEATURES = 50000
N_NEG = 3000
MIN_CHARS = 200
MIN_ALPHA_FRAC = 0.7
MIN_ZLIB_RATIO = 0.2
QUALITY_CHARS = 20000     # chars per doc inspected by the junk heuristics
TOKEN_BATCH = 1000
CV_FOLDS = 5
THR_PERCENTILE = 5.0
MIN_D_FRAC = 0.05         # safety fallback: minimum tok(D) / T
OLD_MAX_COPIES = 3        # only used to report what the old logic would do

# Word-order damage filter.
ORDER_VOCAB = 3000        # most frequent target tokens
ORDER_MIN_PAIRS = 30      # min in-vocab adjacent pairs for a reliable hit rate
ORDER_MIN_INVOCAB = 0.15  # below this in-vocab token share => letter-scrambled
ORDER_PERCENTILE = 2.0
ORDER_MARGIN = 0.85

# Filled by select(); describes the regime of the last pool seen.
STATS = {}

_ALPHA_OR_SPACE = re.compile(r"[^\W\d_]|\s")


def _is_junk(text):
    if len(text) < MIN_CHARS:
        return True
    head = text[:QUALITY_CHARS]
    non_alpha = len(_ALPHA_OR_SPACE.sub("", head))
    if 1.0 - non_alpha / len(head) < MIN_ALPHA_FRAC:
        return True
    raw = head.encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw, 6)) / len(raw) < MIN_ZLIB_RATIO:
        return True
    return False


def _word_tokens(text):
    return text[:QUALITY_CHARS].lower().split()


def _token_ids(toks, vid):
    return np.fromiter((vid.get(t, -1) for t in toks), dtype=np.int64,
                       count=len(toks))


def _pair_codes(ids, nv):
    """Codes of adjacent pairs with both tokens in V, and the in-V mask."""
    inv = ids >= 0
    if len(ids) < 2:
        return np.empty(0, dtype=np.int64), inv
    both = inv[:-1] & inv[1:]
    codes = ids[:-1][both] * nv + ids[1:][both]
    return codes, inv


def _hit_rate(ids, nv, table):
    """Bigram hit rate; None means 'leave the doc unfiltered'."""
    if len(ids) == 0:
        return 0.0
    codes, inv = _pair_codes(ids, nv)
    if len(codes) < ORDER_MIN_PAIRS:
        if float(inv.mean()) < ORDER_MIN_INVOCAB:
            return 0.0
        return None
    return float(table[codes].mean())


def _build_table(id_arrays, nv):
    table = np.zeros(nv * nv, dtype=bool)
    for ids in id_arrays:
        codes, _ = _pair_codes(ids, nv)
        if len(codes):
            table[codes] = True
    return table


def _order_model(target_sample, seed):
    """Return (vid, nv, table, hit_thr), or None if no model can be built."""
    target_toks = [_word_tokens(str(t)) for t in target_sample]
    counts = Counter()
    for toks in target_toks:
        counts.update(toks)
    if not counts:
        return None
    vocab = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:ORDER_VOCAB]
    vid = {tok: i for i, (tok, _) in enumerate(vocab)}
    nv = len(vid)

    target_ids = [_token_ids(toks, vid) for toks in target_toks]
    n_t = len(target_ids)

    # Honest calibration: B from the other folds, hit rate on the held-out fold.
    held_hits = []
    n_folds = min(CV_FOLDS, n_t)
    if n_folds >= 2:
        perm = list(range(n_t))
        random.Random(seed + 2).shuffle(perm)
        for f in range(n_folds):
            held = set(perm[f::n_folds])
            table_f = _build_table(
                [target_ids[i] for i in range(n_t) if i not in held], nv)
            for i in sorted(held):
                h = _hit_rate(target_ids[i], nv, table_f)
                if h is not None:
                    held_hits.append(h)
            del table_f

    if held_hits:
        hit_thr = ORDER_MARGIN * float(np.percentile(held_hits, ORDER_PERCENTILE))
    else:
        hit_thr = 0.0

    table = _build_table(target_ids, nv)
    return vid, nv, table, hit_thr


def _survivors(docs, target_sample, seed):
    """Indices of docs that pass dedup, the junk filters and the word-order
    damage filter (first copy kept)."""
    model = _order_model(target_sample, seed)
    seen = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        key = hashlib.md5(" ".join(d.split()).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)
        if _is_junk(d):
            continue
        if model is not None:
            vid, nv, table, hit_thr = model
            h = _hit_rate(_token_ids(_word_tokens(d), vid), nv, table)
            if h is not None and h < hit_thr:
                continue
        keep.append(i)
    return keep


def _fit(X, y):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    return clf


def _score(docs, target_sample, seed):
    """Return (pool decision scores, in-domain threshold).

    The threshold is the 5th percentile of cross-validated (held-out) decision
    scores of the target_sample docs.
    """
    from scipy.sparse import vstack  # scipy ships with scikit-learn

    n_t = len(target_sample)
    n = len(docs)
    texts = [str(t)[:TRUNC_CHARS] for t in target_sample]
    texts += [(d if isinstance(d, str) else "")[:TRUNC_CHARS] for d in docs]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=MAX_FEATURES,
        dtype=np.float32,
    )
    X = vec.fit_transform(texts).tocsr()
    X_t = X[:n_t]
    X_p = X[n_t:]

    # Negatives are drawn from the whole pool (not only the survivors).
    neg = random.Random(seed).sample(range(n), min(N_NEG, n))
    X_neg = X_p[neg]
    y_neg = np.zeros(len(neg), dtype=np.int32)

    # Final model: all targets vs. the seeded negatives.
    X_train = vstack([X_t, X_neg]).tocsr()
    y = np.concatenate([np.ones(n_t, dtype=np.int32), y_neg])
    clf = _fit(X_train, y)
    scores = clf.decision_function(X_p)

    # Honest threshold: held-out scores of the target docs.
    n_folds = min(CV_FOLDS, n_t)
    if n_folds >= 2:
        perm = list(range(n_t))
        random.Random(seed + 1).shuffle(perm)
        perm = np.asarray(perm, dtype=np.int64)
        cv_scores = np.empty(n_t, dtype=np.float64)
        for f in range(n_folds):
            held = perm[f::n_folds]
            mask = np.ones(n_t, dtype=bool)
            mask[held] = False
            train_idx = np.nonzero(mask)[0]
            X_f = vstack([X_t[train_idx], X_neg]).tocsr()
            y_f = np.concatenate([np.ones(len(train_idx), dtype=np.int32), y_neg])
            cv_scores[held] = _fit(X_f, y_f).decision_function(X_t[held])
    else:
        cv_scores = np.asarray(clf.decision_function(X_t), dtype=np.float64)

    thr = float(np.percentile(cv_scores, THR_PERCENTILE))
    return scores, thr


def _token_counts(enc, docs, indices):
    """gpt2 token count for each index in `indices` (same order)."""
    counts = []
    for pos in range(0, len(indices), TOKEN_BATCH):
        batch = indices[pos:pos + TOKEN_BATCH]
        toks = enc.encode_ordinary_batch([docs[int(i)] for i in batch])
        counts.extend(len(t) for t in toks)
    return counts


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)

    STATS.clear()
    STATS.update({
        "T": int(T),
        "regime": "empty",
        "n_pool": n,
        "n_survivors": 0,
        "thr": None,
        "thr_effective": None,
        "n_docs_D": 0,
        "tok_D": 0,
        "tok_D_over_T": 0.0,
        "k": 0,
        "old_filler_tokens": 0,
        "fallback_used": False,
        "n_docs_used": 0,
        "tok_used": 0,
        "tok_used_over_T": 0.0,
    })

    if n == 0:
        return []

    keep = _survivors(docs, target_sample, seed)
    STATS["n_survivors"] = len(keep)
    if not keep:
        STATS["regime"] = "no_survivors"
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    scores, thr = _score(docs, target_sample, seed)

    # Rank surviving docs by score, best first (stable for determinism).
    keep_arr = np.asarray(keep, dtype=np.int64)
    order = keep_arr[np.argsort(-scores[keep_arr], kind="stable")]
    order_scores = scores[order]

    # D = in-domain survivors; a prefix of `order` since it is sorted by score.
    n_in = int(np.sum(order_scores >= thr))
    in_domain = [int(i) for i in order[:n_in]]
    below = [int(i) for i in order[n_in:]]

    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(seed)

    in_counts = _token_counts(enc, docs, in_domain)
    tok_d = int(sum(in_counts))

    STATS.update({
        "thr": float(thr),
        "thr_effective": float(thr),
        "n_docs_D": n_in,
        "tok_D": tok_d,
        "tok_D_over_T": tok_d / float(T),
    })

    # Case 1: enough in-domain text -> top-scoring docs up to ~T, shuffled.
    if tok_d >= T:
        selected = []
        total = 0
        for i, c in zip(in_domain, in_counts):
            selected.append(i)
            total += c
            if total >= T:
                break
        rng.shuffle(selected)
        STATS.update({
            "regime": "enough_in_domain",
            "k": 1,
            "old_filler_tokens": 0,
            "n_docs_used": len(selected),
            "tok_used": int(total),
            "tok_used_over_T": total / float(T),
        })
        return selected

    # Token counts, cached so below-threshold docs are tokenized at most once.
    count_of = dict(zip(in_domain, in_counts))

    def counts_for(batch):
        missing = [i for i in batch if i not in count_of]
        if missing:
            for i, c in zip(missing, _token_counts(enc, docs, missing)):
                count_of[i] = c
        return [count_of[i] for i in batch]

    # Safety fallback: D is tiny -> lower thr until the set has 0.05 * T tokens.
    used = list(in_domain)
    tok_used = tok_d
    min_tok = MIN_D_FRAC * T
    fallback_used = False
    if tok_used < min_tok and below:
        fallback_used = True
        pos = 0
        while pos < len(below) and tok_used < min_tok:
            batch = below[pos:pos + TOKEN_BATCH]
            pos += TOKEN_BATCH
            for i, c in zip(batch, counts_for(batch)):
                used.append(i)
                tok_used += c
                if tok_used >= min_tok:
                    break
    thr_eff = float(thr)
    if fallback_used and used:
        thr_eff = min(thr_eff, float(scores[used[-1]]))

    # Diagnostics: filler tokens the old logic (<= 3 copies, then
    # below-threshold filler up to T) would have added for this pool.
    old_filler = 0
    old_total = min(OLD_MAX_COPIES, int(math.ceil(T / tok_d))) * tok_d if tok_d > 0 else 0
    if old_total < T:
        pos = 0
        while pos < len(below) and old_total < T:
            batch = below[pos:pos + TOKEN_BATCH]
            pos += TOKEN_BATCH
            for c in counts_for(batch):
                old_filler += c
                old_total += c
                if old_total >= T:
                    break

    STATS.update({
        "old_filler_tokens": int(old_filler),
        "fallback_used": fallback_used,
        "thr_effective": thr_eff,
        "n_docs_used": len(used),
        "tok_used": int(tok_used),
        "tok_used_over_T": tok_used / float(T),
    })

    if tok_used <= 0 or not used:
        # Cannot happen with non-empty survivors, but stay safe.
        STATS["regime"] = "degenerate"
        fallback = [int(i) for i in order]
        rng.shuffle(fallback)
        return fallback

    # Case 2: k independently shuffled copies of the in-domain set, truncated
    # at about T tokens. No cap on k, no below-threshold filler.
    k = int(math.ceil(T / float(tok_used)))
    result = []
    total = 0
    for _ in range(k):
        if total >= T:
            break
        copy = list(used)
        rng.shuffle(copy)
        for i in copy:
            result.append(i)
            total += count_of[i]
            if total >= T:
                break

    STATS.update({
        "regime": "fallback_repeat" if fallback_used else "repeat_in_domain",
        "k": k,
    })
    return result

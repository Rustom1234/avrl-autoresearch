"""Target-similarity curator.

select(docs, target_sample) -> ordered list of indices into docs.

1. Score every pool doc with a char n-gram TF-IDF + logistic regression model
   (target_sample = positives, seeded random pool sample = negatives).
2. Drop junk: exact duplicates, very short docs, docs with a low share of
   alphabetic/space characters, and highly compressible (templated spam) docs.
3. In-domain threshold thr = 5th percentile of 5-fold cross-validated
   decision scores of the target_sample docs (each fold trained on the other
   targets plus the same seeded negatives).
4. D = surviving docs with score >= thr.
   - tok(D) >= T: take the best-scoring docs until about T tokens, shuffle.
   - otherwise: k = min(3, ceil(T / tok(D))) copies of D, each shuffled
     independently, concatenated and truncated at about T tokens. Only if
     3 * tok(D) < T, the remaining tokens are covered with the next-best
     surviving docs below thr, mixed uniformly into the whole list.
"""
import hashlib
import math
import os
import random
import re
import zlib

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
MAX_COPIES = 3

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


def _survivors(docs):
    """Indices of docs that pass dedup and the junk filters (first copy kept)."""
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
    if n == 0:
        return []

    keep = _survivors(docs)
    if not keep:
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
        return selected

    # Case 2: repeat the in-domain set instead of padding with filler.
    count_of = dict(zip(in_domain, in_counts))
    base = []
    total = 0
    if tok_d > 0:
        k = min(MAX_COPIES, int(math.ceil(T / tok_d)))
        for _ in range(k):
            if total >= T:
                break
            copy = list(in_domain)
            rng.shuffle(copy)
            for i in copy:
                base.append(i)
                total += count_of[i]
                if total >= T:
                    break

    if total >= T or not below:
        if base:
            return base
        # Nothing in-domain and no filler available: fall back to survivors.
        fallback = [int(i) for i in order]
        rng.shuffle(fallback)
        return fallback

    # Case 3: even MAX_COPIES copies fall short -> next-best docs as filler.
    filler = []
    pos = 0
    while pos < len(below) and total < T:
        batch = below[pos:pos + TOKEN_BATCH]
        pos += TOKEN_BATCH
        for i, c in zip(batch, _token_counts(enc, docs, batch)):
            filler.append(i)
            total += c
            if total >= T:
                break

    if not base:
        rng.shuffle(filler)
        return filler

    # Mix the filler uniformly into the whole list, keeping the order of the
    # repeated in-domain copies intact.
    rng.shuffle(filler)
    n_total = len(base) + len(filler)
    filler_slots = set(rng.sample(range(n_total), len(filler)))
    result = []
    bi = 0
    fi = 0
    for p in range(n_total):
        if p in filler_slots:
            result.append(filler[fi])
            fi += 1
        else:
            result.append(base[bi])
            bi += 1
    return result

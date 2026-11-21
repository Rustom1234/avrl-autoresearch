"""Target-similarity curator.

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
   - 0 < tok(D) < T: two-phase curriculum.
       F = max(min(0.25*T, T - tok(D)), T - 3*tok(D), 0) filler tokens.
       Phase 1 (start of the stream): the best-scoring surviving docs with
       score < thr, taken in descending score order up to about F tokens,
       shuffled, as one contiguous block.
       Phase 2: about T - F tokens of in-domain text as independently
       shuffled copies of D: partial copy (seeded random subset of D covering
       the leftover tokens) -> full copies -> final full copy, so the stream
       ends with one complete shuffled copy of D.
     If there are no surviving docs below thr: k = min(3, ceil(T / tok(D)))
     independently shuffled copies of D, truncated at about T tokens.
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
MAX_COPIES = 3

# Two-phase curriculum (used when 0 < tok(D) < T).
FILLER_FRAC = 0.25        # share of T given to near-domain filler
FILLER_SLACK = 512        # stop collecting filler once within this many tokens of F

# Word-order damage filter.
ORDER_VOCAB = 3000        # most frequent target tokens
ORDER_MIN_PAIRS = 30      # min in-vocab adjacent pairs for a reliable hit rate
ORDER_MIN_INVOCAB = 0.15  # below this in-vocab token share => letter-scrambled
ORDER_PERCENTILE = 2.0
ORDER_MARGIN = 0.85

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


def _collect_filler(enc, docs, below, budget):
    """Best-scoring below-thr survivors, in descending score order, totalling
    about `budget` tokens without exceeding it. Docs that would overshoot the
    remaining budget are skipped. Returns (indices, total tokens)."""
    filler = []
    total = 0
    if budget <= 0:
        return filler, total
    pos = 0
    done = False
    while pos < len(below) and not done:
        batch = below[pos:pos + TOKEN_BATCH]
        pos += TOKEN_BATCH
        for i, c in zip(batch, _token_counts(enc, docs, batch)):
            if total + c <= budget:
                filler.append(i)
                total += c
            if budget - total < FILLER_SLACK:
                done = True
                break
    return filler, total


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    keep = _survivors(docs, target_sample, seed)
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

    # Nothing in-domain: fall back to all survivors, shuffled.
    if tok_d <= 0:
        fallback = [int(i) for i in order]
        rng.shuffle(fallback)
        return fallback

    count_of = dict(zip(in_domain, in_counts))

    # No below-thr survivors: repeat the in-domain set (previous behaviour).
    if not below:
        base = []
        total = 0
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
        return base

    # Case 2: two-phase curriculum (0 < tok(D) < T).
    # Phase 1: near-domain clean filler, one contiguous block at the start.
    budget = int(max(min(FILLER_FRAC * T, T - tok_d), T - MAX_COPIES * tok_d, 0))
    filler, filler_tok = _collect_filler(enc, docs, below, budget)
    rng.shuffle(filler)

    # Phase 2: in-domain text covering the rest. filler_tok <= budget
    # <= T - tok_d, so at least one full copy of D always fits.
    in_budget = min(T - filler_tok, MAX_COPIES * tok_d)
    n_full = max(1, in_budget // tok_d)
    leftover = in_budget - n_full * tok_d

    result = list(filler)

    # Partial copy first: a seeded random subset of D covering the leftover
    # tokens (never overshooting, so the final full copy fits within T).
    if leftover > 0:
        copy = list(in_domain)
        rng.shuffle(copy)
        part_total = 0
        for i in copy:
            c = count_of[i]
            if part_total + c <= leftover:
                result.append(i)
                part_total += c
                if part_total >= leftover:
                    break

    # Full copies; the last one is a complete shuffled copy of D that ends
    # the stream.
    for _ in range(n_full):
        copy = list(in_domain)
        rng.shuffle(copy)
        result.extend(copy)

    return result

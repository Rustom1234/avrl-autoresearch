"""Target-similarity curator with one round of self-training.

select(docs, target_sample) -> ordered list of indices into docs.

1. Drop junk: exact duplicates, very short docs, docs with a low share of
   alphabetic/space characters, highly compressible (templated spam) docs,
   and docs with damaged word order (words shuffled inside sentences or
   letters scrambled inside words). The word-order filter measures the share
   of a doc's adjacent in-vocabulary token pairs that also occur in
   target_sample; its threshold is 0.85 * the 2nd percentile of 5-fold
   cross-validated hit rates of the target docs.
2. Stage 1: score every pool doc with a char n-gram TF-IDF + logistic
   regression model (target_sample = positives, seeded random pool sample =
   negatives) -> s1. thr1 = 5th percentile of 5-fold cross-validated decision
   scores of the target_sample docs (each fold trained on the other targets
   plus the same seeded negatives).
3. Stage 2 (self-training, same TF-IDF matrix):
   - positives = targets + pseudo-positives (surviving pool docs with
     s1 >= thr1; seeded sample of 3000 if there are more);
   - negatives = seeded sample of 3000 pool docs with s1 < thr1;
   - docs with s1 >= thr1 that failed the survivor filters are in neither
     class.
   Fit the same logistic regression -> s2. thr2 = 5th percentile of 5-fold
   cross-validated target scores (each fold trained on the other targets plus
   all pseudo-positives plus the stage-2 negatives).
   If there are no pseudo-positives (or no stage-2 negatives can be drawn),
   s1 / thr1 are used instead.
4. D = surviving docs with score >= thr (stage-2 score and threshold when
   available).
   - tok(D) >= T: take the best-scoring docs until about T tokens, shuffle.
   - otherwise: k = min(3, ceil(T / tok(D))) copies of D, each shuffled
     independently, concatenated and truncated at about T tokens. Only if
     3 * tok(D) < T, the remaining tokens are covered with the next-best
     surviving docs below thr, mixed uniformly into the whole list.
5. Diagnostics of the last select() call are recorded in the module-level
   STATS dict.
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

# Self-training (stage 2).
MAX_PSEUDO_POS = 3000
N_NEG2 = 3000

# Word-order damage filter.
ORDER_VOCAB = 3000        # most frequent target tokens
ORDER_MIN_PAIRS = 30      # min in-vocab adjacent pairs for a reliable hit rate
ORDER_MIN_INVOCAB = 0.15  # below this in-vocab token share => letter-scrambled
ORDER_PERCENTILE = 2.0
ORDER_MARGIN = 0.85

# Diagnostics of the most recent select() call.
#   stage1_docs / stage1_tok_frac : |D| and tok(D)/T under stage 1 (s1, thr1)
#   stage2_docs / stage2_tok_frac : same under stage 2 (None if not run)
#   n_pseudo_pos                  : pseudo-positives used to train stage 2
#   n_pseudo_candidates           : survivors with s1 >= thr1 (before sampling)
#   n_stage2_neg                  : stage-2 negatives
#   stage_used                    : 1 or 2 (which scores built the list)
#   case                          : 1, 2 or 3 (None for degenerate fallbacks)
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


def _cv_threshold(X_t, X_extra, y_extra, perm_seed, full_clf):
    """5th percentile of held-out decision scores of the target docs.

    Each fold trains on the other targets plus the fixed rows
    (X_extra, y_extra) and scores the held-out targets.
    """
    from scipy.sparse import vstack  # scipy ships with scikit-learn

    n_t = X_t.shape[0]
    n_folds = min(CV_FOLDS, n_t)
    if n_folds >= 2:
        perm = list(range(n_t))
        random.Random(perm_seed).shuffle(perm)
        perm = np.asarray(perm, dtype=np.int64)
        cv_scores = np.empty(n_t, dtype=np.float64)
        for f in range(n_folds):
            held = perm[f::n_folds]
            mask = np.ones(n_t, dtype=bool)
            mask[held] = False
            train_idx = np.nonzero(mask)[0]
            X_f = vstack([X_t[train_idx], X_extra]).tocsr()
            y_f = np.concatenate(
                [np.ones(len(train_idx), dtype=np.int32), y_extra])
            cv_scores[held] = _fit(X_f, y_f).decision_function(X_t[held])
    else:
        cv_scores = np.asarray(full_clf.decision_function(X_t),
                               dtype=np.float64)
    return float(np.percentile(cv_scores, THR_PERCENTILE))


def _score(docs, target_sample, seed, keep):
    """Two-stage scoring.

    Returns a dict with:
      s1, thr1            : stage-1 pool scores and threshold
      s2, thr2            : stage-2 pool scores and threshold (None if stage 2
                            could not be run)
      n_pseudo_pos        : pseudo-positives used in stage-2 training
      n_pseudo_candidates : survivors with s1 >= thr1
      n_stage2_neg        : stage-2 negatives
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
    del texts
    X_t = X[:n_t]
    X_p = X[n_t:]

    # ---- Stage 1 -----------------------------------------------------------
    # Negatives are drawn from the whole pool (not only the survivors).
    neg = random.Random(seed).sample(range(n), min(N_NEG, n))
    X_neg = X_p[neg]
    y_neg = np.zeros(len(neg), dtype=np.int32)

    X_train = vstack([X_t, X_neg]).tocsr()
    y = np.concatenate([np.ones(n_t, dtype=np.int32), y_neg])
    clf1 = _fit(X_train, y)
    s1 = np.asarray(clf1.decision_function(X_p), dtype=np.float64)
    thr1 = _cv_threshold(X_t, X_neg, y_neg, seed + 1, clf1)

    out = {
        "s1": s1,
        "thr1": thr1,
        "s2": None,
        "thr2": None,
        "n_pseudo_pos": 0,
        "n_pseudo_candidates": 0,
        "n_stage2_neg": 0,
    }

    # ---- Stage 2 -----------------------------------------------------------
    above = s1 >= thr1
    keep_arr = np.asarray(keep, dtype=np.int64)
    # Pseudo-positive candidates: surviving docs with s1 >= thr1 (ascending
    # index order, so the seeded sample is deterministic).
    cand_pos = [int(i) for i in keep_arr[above[keep_arr]]]
    out["n_pseudo_candidates"] = len(cand_pos)
    if not cand_pos:
        return out

    if len(cand_pos) > MAX_PSEUDO_POS:
        pseudo = sorted(random.Random(seed + 3).sample(cand_pos, MAX_PSEUDO_POS))
    else:
        pseudo = cand_pos

    # Negatives: only docs with s1 < thr1 (never pseudo-positives, never
    # damaged in-domain docs, which have s1 >= thr1).
    cand_neg = [int(i) for i in np.nonzero(~above)[0]]
    if not cand_neg:
        return out
    neg2 = sorted(random.Random(seed + 4).sample(
        cand_neg, min(N_NEG2, len(cand_neg))))

    X_pp = X_p[pseudo]
    X_neg2 = X_p[neg2]
    X_extra = vstack([X_pp, X_neg2]).tocsr()
    y_extra = np.concatenate([
        np.ones(len(pseudo), dtype=np.int32),
        np.zeros(len(neg2), dtype=np.int32),
    ])

    X_train2 = vstack([X_t, X_extra]).tocsr()
    y2 = np.concatenate([np.ones(n_t, dtype=np.int32), y_extra])
    clf2 = _fit(X_train2, y2)
    s2 = np.asarray(clf2.decision_function(X_p), dtype=np.float64)
    thr2 = _cv_threshold(X_t, X_extra, y_extra, seed + 5, clf2)

    out["s2"] = s2
    out["thr2"] = thr2
    out["n_pseudo_pos"] = len(pseudo)
    out["n_stage2_neg"] = len(neg2)
    return out


class _TokenCounter:
    """gpt2 token counts with a cache, so docs shared between the stage-1 and
    stage-2 in-domain sets are tokenised only once."""

    def __init__(self, docs):
        self._enc = tiktoken.get_encoding("gpt2")
        self._docs = docs
        self._cache = {}

    def counts(self, indices):
        """Token count for each index in `indices` (same order)."""
        missing = []
        seen = set()
        for i in indices:
            i = int(i)
            if i not in self._cache and i not in seen:
                seen.add(i)
                missing.append(i)
        for pos in range(0, len(missing), TOKEN_BATCH):
            batch = missing[pos:pos + TOKEN_BATCH]
            toks = self._enc.encode_ordinary_batch(
                [self._docs[i] for i in batch])
            for i, t in zip(batch, toks):
                self._cache[i] = len(t)
        return [self._cache[int(i)] for i in indices]


def _reset_stats():
    STATS.clear()
    STATS.update({
        "stage1_docs": 0,
        "stage1_tok_frac": 0.0,
        "stage2_docs": None,
        "stage2_tok_frac": None,
        "n_pseudo_pos": 0,
        "n_pseudo_candidates": 0,
        "n_stage2_neg": 0,
        "thr1": None,
        "thr2": None,
        "stage_used": None,
        "case": None,
    })


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    _reset_stats()
    n = len(docs)
    if n == 0:
        return []

    keep = _survivors(docs, target_sample, seed)
    if not keep:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    res = _score(docs, target_sample, seed, keep)
    s1, thr1 = res["s1"], res["thr1"]
    s2, thr2 = res["s2"], res["thr2"]
    use_stage2 = s2 is not None

    keep_arr = np.asarray(keep, dtype=np.int64)
    counter = _TokenCounter(docs)

    # Stage-1 in-domain set (for diagnostics, and for the list if stage 2
    # was not run).
    d1 = [int(i) for i in keep_arr[s1[keep_arr] >= thr1]]
    d1_tok = int(sum(counter.counts(d1)))

    STATS["stage1_docs"] = len(d1)
    STATS["stage1_tok_frac"] = d1_tok / float(T)
    STATS["n_pseudo_pos"] = res["n_pseudo_pos"]
    STATS["n_pseudo_candidates"] = res["n_pseudo_candidates"]
    STATS["n_stage2_neg"] = res["n_stage2_neg"]
    STATS["thr1"] = thr1
    STATS["thr2"] = thr2
    STATS["stage_used"] = 2 if use_stage2 else 1

    if use_stage2:
        scores, thr = s2, thr2
    else:
        scores, thr = s1, thr1

    # Rank surviving docs by score, best first (stable for determinism).
    order = keep_arr[np.argsort(-scores[keep_arr], kind="stable")]
    order_scores = scores[order]

    # D = in-domain survivors; a prefix of `order` since it is sorted by score.
    n_in = int(np.sum(order_scores >= thr))
    in_domain = [int(i) for i in order[:n_in]]
    below = [int(i) for i in order[n_in:]]

    rng = random.Random(seed)

    in_counts = counter.counts(in_domain)
    tok_d = int(sum(in_counts))

    if use_stage2:
        STATS["stage2_docs"] = len(in_domain)
        STATS["stage2_tok_frac"] = tok_d / float(T)

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
        STATS["case"] = 1
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
            STATS["case"] = 2
            return base
        # Nothing in-domain and no filler available: fall back to survivors.
        fallback = [int(i) for i in order]
        rng.shuffle(fallback)
        STATS["case"] = None
        return fallback

    # Case 3: even MAX_COPIES copies fall short -> next-best docs as filler.
    STATS["case"] = 3
    filler = []
    pos = 0
    while pos < len(below) and total < T:
        batch = below[pos:pos + TOKEN_BATCH]
        pos += TOKEN_BATCH
        for i, c in zip(batch, counter.counts(batch)):
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

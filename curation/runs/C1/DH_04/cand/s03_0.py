"""Target-similarity curator.

select(docs, target_sample) -> ordered list of indices into docs.

1. Score every pool doc with a char n-gram TF-IDF + logistic regression model
   (target_sample = positives, seeded random pool sample = negatives).
2. Drop junk: exact duplicates, very short docs, docs with a low share of
   alphabetic/space characters, and highly compressible (templated spam) docs.
3. Damage filter: build a reference vocabulary (words in target_sample or
   with pool document frequency >= 5) and drop docs whose out-of-vocabulary
   rate exceeds max(1.5 * 95th percentile of target OOV rates, 0.02). Target
   OOV rates use the pool-frequency criterion only. Docs with mojibake
   markers are dropped too.
4. Near-duplicate removal: docs sharing a normalised first line, or at least
   4 of 8 min-hash values (word 5-gram shingles of the first 2000 chars), are
   grouped; only the doc with the lowest OOV rate (ties: longer) is kept.
5. In-domain threshold thr = 5th percentile of 5-fold cross-validated
   decision scores of the target_sample docs (each fold trained on the other
   targets plus the same seeded negatives).
6. D = surviving docs with score >= thr.
   - tok(D) >= T: take the best-scoring docs until about T tokens, shuffle.
   - otherwise: k = min(3, ceil(T / tok(D))) copies of D, each shuffled
     independently, concatenated and truncated at about T tokens. Only if
     3 * tok(D) < T, the remaining tokens are covered with the next-best
     surviving docs below thr, mixed uniformly into the whole list.
"""
import hashlib
import itertools
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

# Damage filter / near-duplicate removal.
VOCAB_CHARS = 20000       # chars per doc used for the vocabulary / OOV rate
MIN_WORD_LEN = 3
MIN_POOL_DF = 5
OOV_PERCENTILE = 95.0
OOV_MULT = 1.5
OOV_FLOOR = 0.02
MOJIBAKE_MARKERS = ("\u00e2\u20ac", "\u00c3", "\u00c2")   # 'â€', 'Ã', 'Â'
MOJIBAKE_MAX_PER_10K = 3.0
MINHASH_CHARS = 2000
SHINGLE_WORDS = 5
N_MINHASH = 8
MIN_SHARED = 4
TITLE_CHARS = 300

_ALPHA_OR_SPACE = re.compile(r"[^\W\d_]|\s")
_WORD = re.compile(r"[A-Za-z]+")


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


# --------------------------------------------------------------------------
# Damage filter
# --------------------------------------------------------------------------

def _words(text):
    """Lowercased [A-Za-z]+ tokens of the first VOCAB_CHARS chars."""
    found = _WORD.findall(text[:VOCAB_CHARS])
    if not found:
        return []
    return " ".join(found).lower().split()


def _long_words(text):
    return [w for w in _words(text) if len(w) >= MIN_WORD_LEN]


def _oov_rate(tokens, known):
    if not tokens:
        return None
    unknown = 0
    for w in tokens:
        if w not in known:
            unknown += 1
    return unknown / len(tokens)


def _has_mojibake(text):
    n = len(text)
    if n == 0:
        return False
    count = 0
    for m in MOJIBAKE_MARKERS:
        count += text.count(m)
    return count * 10000.0 / n > MOJIBAKE_MAX_PER_10K


def _damage_filter(docs, keep, target_sample):
    """Return (kept indices, {index: oov rate}) after the OOV/mojibake filter."""
    # Pool document frequency over the surviving docs.
    df = Counter()
    for i in keep:
        df.update(set(_long_words(docs[i])))
    pool_known = {w for w, c in df.items() if c >= MIN_POOL_DF}
    del df

    target_tokens = [_long_words(str(t)) for t in target_sample]
    target_words = set()
    for toks in target_tokens:
        target_words.update(toks)
    known = pool_known | target_words

    # Target OOV rates: pool-frequency criterion only.
    target_rates = []
    for toks in target_tokens:
        r = _oov_rate(toks, pool_known)
        if r is not None:
            target_rates.append(r)
    if target_rates:
        oov_thr = max(
            float(np.percentile(np.asarray(target_rates), OOV_PERCENTILE)) * OOV_MULT,
            OOV_FLOOR,
        )
    else:
        oov_thr = OOV_FLOOR

    kept = []
    oov = {}
    for i in keep:
        d = docs[i]
        r = _oov_rate(_long_words(d), known)
        if r is None:
            r = 1.0
        if r > oov_thr:
            continue
        if _has_mojibake(d):
            continue
        kept.append(i)
        oov[i] = r
    return kept, oov


# --------------------------------------------------------------------------
# Near-duplicate removal
# --------------------------------------------------------------------------

def _title_key(text):
    s = text.lstrip()
    line = s.split("\n", 1)[0][:TITLE_CHARS].lower()
    return "".join(ch for ch in line if ch.isalpha())


def _mix(x):
    x = x ^ (x >> np.uint64(31))
    x = x * np.uint64(0x9E3779B97F4A7C15)
    x = x ^ (x >> np.uint64(29))
    return x


def _signatures(docs, idx, seed):
    """Min-hash signatures (N_MINHASH values) of word 5-gram shingles."""
    rs = np.random.RandomState((seed + 7919) % (2 ** 32))
    a = rs.randint(1, 2 ** 63, size=N_MINHASH, dtype=np.uint64) | np.uint64(1)
    b = rs.randint(1, 2 ** 63, size=N_MINHASH, dtype=np.uint64)
    coef = rs.randint(1, 2 ** 63, size=SHINGLE_WORDS, dtype=np.uint64) | np.uint64(1)

    n = len(idx)
    sig = np.zeros((n, N_MINHASH), dtype=np.uint64)
    has = np.zeros(n, dtype=bool)
    cache = {}
    for r, i in enumerate(idx):
        words = docs[i][:MINHASH_CHARS].lower().split()
        m = len(words) - SHINGLE_WORDS + 1
        if m < 1:
            continue
        hs = []
        for w in words:
            h = cache.get(w)
            if h is None:
                h = int.from_bytes(
                    hashlib.blake2b(
                        w.encode("utf-8", "ignore"), digest_size=8
                    ).digest(),
                    "little",
                )
                cache[w] = h
            hs.append(h)
        wh = np.array(hs, dtype=np.uint64)
        sh = np.zeros(m, dtype=np.uint64)
        for j in range(SHINGLE_WORDS):
            sh = sh + _mix(wh[j:j + m]) * coef[j]
        x = _mix(sh[:, None] * a[None, :] + b[None, :])
        sig[r] = x.min(axis=0)
        has[r] = True
    return sig, has


def _near_dedup(docs, keep, oov, seed):
    """Keep one doc (lowest oov, then longest) per near-duplicate group."""
    n = len(keep)
    if n < 2:
        return list(keep)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    # Title key groups.
    first_with = {}
    for r, i in enumerate(keep):
        key = _title_key(docs[i])
        if not key:
            continue
        other = first_with.get(key)
        if other is None:
            first_with[key] = r
        else:
            union(r, other)
    del first_with

    # Min-hash groups: at least MIN_SHARED of N_MINHASH values equal.
    sig, has = _signatures(docs, keep, seed)
    rows = np.nonzero(has)[0]
    if len(rows) >= 2:
        sub = sig[rows]
        mult = np.uint64(0xC2B2AE3D27D4EB4F)
        for ci, combo in enumerate(
            itertools.combinations(range(N_MINHASH), MIN_SHARED)
        ):
            key = np.full(len(rows), ci + 1, dtype=np.uint64)
            for c in combo:
                key = _mix((key ^ sub[:, c]) * mult)
            order = np.argsort(key, kind="stable")
            sk = key[order]
            eq = np.nonzero(sk[1:] == sk[:-1])[0]
            for j in eq:
                union(int(rows[order[j]]), int(rows[order[j + 1]]))

    best = {}
    for r, i in enumerate(keep):
        root = find(r)
        cand = (oov.get(i, 1.0), -len(docs[i]), i)
        cur = best.get(root)
        if cur is None or cand < cur:
            best[root] = cand
    winners = {v[2] for v in best.values()}
    return [i for i in keep if i in winners]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

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

    # Damage filter and near-duplicate removal (before thresholding).
    clean, oov = _damage_filter(docs, keep, target_sample)
    if clean:
        clean = _near_dedup(docs, clean, oov, seed)
    if clean:
        keep = clean

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

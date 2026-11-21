"""Curator: keep only clean, unique, target-style documents.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. Dedupe (exact: hash of whitespace-normalised text; near: hash of first 300 chars).
  2. Junk guard (>= 200 chars, >= 85% alphabetic / whitespace / common punctuation).
  3. Word-order score (bigram hit rate against target bigrams) to reject shuffled docs.
  4. Domain score (char n-gram TF-IDF + logistic regression), keep p > 0.5.
  5. Seeded shuffle of the kept indices. No padding: if the kept set is shorter
     than T tokens, the trainer cycles it.
The seed for all randomness is the environment variable CURATE_SEED.
"""
import hashlib
import os
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CHUNK_CHARS = 2000        # positive chunk size / negative truncation
SCORE_CHARS = 3000        # prefix of each pool doc that is scored
N_NEG = 3000              # random pool docs used as negatives
ORDER_TOKENS = 400        # tokens used for the word-order score
MIN_CHARS = 200
MIN_CLEAN_FRAC = 0.85
GUARD_CHARS = 20000       # prefix inspected by the junk guard
SCORE_CHAR_BUDGET = 30_000_000  # cap on total chars vectorised, to stay within time

# Letters (unicode), whitespace, and punctuation common in WikiText-style text.
_GOOD = re.compile(r"[^\W\d_]|[\s.,;:'\"()\-!?=@]")


def _h(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).digest()


def _bigrams(text, limit=None):
    toks = text.lower().split()
    if limit is not None:
        toks = toks[:limit]
    return list(zip(toks, toks[1:]))


def _order_score(text, bigram_set):
    bg = _bigrams(text, ORDER_TOKENS)
    if not bg:
        return 0.0
    return sum(1 for b in bg if b in bigram_set) / len(bg)


def _clean_enough(text):
    if len(text) < MIN_CHARS:
        return False
    head = text[:GUARD_CHARS]
    good = len(_GOOD.findall(head))
    return good >= MIN_CLEAN_FRAC * len(head)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    # 1. Dedupe -----------------------------------------------------------
    seen_full, seen_head = set(), set()
    unique = []
    for i, d in enumerate(docs):
        norm = " ".join(d.split())
        hf = _h(norm)
        if hf in seen_full:
            continue
        hh = _h(norm[:300])
        if hh in seen_head:
            continue
        seen_full.add(hf)
        seen_head.add(hh)
        unique.append(i)

    targets = [t for t in target_sample if t and t.strip()]
    if not targets or not unique:
        idx = unique or list(range(n))
        rng.shuffle(idx)
        return idx

    # 3. Word-order score: calibrate on a 150/50 split of the target sample
    t_order = list(range(len(targets)))
    random.Random(seed + 1).shuffle(t_order)
    n_build = max(1, int(round(len(targets) * 0.75)))
    build, calib = t_order[:n_build], t_order[n_build:]

    build_set = set()
    for j in build:
        build_set.update(_bigrams(targets[j]))
    if calib:
        cal_scores = [_order_score(targets[j], build_set) for j in calib]
        order_thresh = 0.6 * float(np.percentile(cal_scores, 5))
    else:
        order_thresh = 0.0

    full_set = set(build_set)
    for j in calib:
        full_set.update(_bigrams(targets[j]))

    # 2/4a. Junk guard + order test (cheap filters first) -----------------
    candidates = []
    for i in unique:
        d = docs[i]
        if not _clean_enough(d):
            continue
        if _order_score(d, full_set) < order_thresh:
            continue
        candidates.append(i)

    if not candidates:
        idx = list(unique)
        rng.shuffle(idx)
        return idx

    # 2. Domain classifier ------------------------------------------------
    pos = []
    for t in targets:
        for s in range(0, len(t), CHUNK_CHARS):
            chunk = t[s:s + CHUNK_CHARS]
            if s == 0 or len(chunk.strip()) >= MIN_CHARS:
                pos.append(chunk)

    neg_idx = random.Random(seed + 2).sample(unique, min(N_NEG, len(unique)))
    neg = [docs[i][:CHUNK_CHARS] for i in neg_idx]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_train = vec.fit_transform(pos + neg)
    y_train = np.array([1] * len(pos) + [0] * len(neg))
    # Balanced weights: the random negatives contain target-style docs, and
    # balancing keeps their probability above 0.5 despite that label noise.
    clf = LogisticRegression(C=10.0, max_iter=300, class_weight="balanced")
    clf.fit(X_train, y_train)

    score_chars = max(500, min(SCORE_CHARS, SCORE_CHAR_BUDGET // len(candidates)))
    probs = np.empty(len(candidates), dtype=np.float64)
    batch = 2000
    for s in range(0, len(candidates), batch):
        part = candidates[s:s + batch]
        X = vec.transform([docs[i][:score_chars] for i in part])
        probs[s:s + len(part)] = clf.predict_proba(X)[:, 1]

    # 4. Keep ---------------------------------------------------------------
    kept = [i for i, p in zip(candidates, probs) if p > 0.5]
    if not kept:
        # Degenerate case only: never return an empty list.
        kept = list(candidates)

    # 5. Seeded shuffle; no padding with off-domain docs. If the kept tokens
    # come to less than T, the trainer cycles this list.
    rng.shuffle(kept)
    return kept

"""Target-similarity curator.

Pipeline:
1. Drop exact duplicates (hash of stripped text).
2. Drop low-alphabetic-fraction docs and docs matching spam phrases.
3. Score survivors by TF-IDF cosine similarity to the mean target_sample vector.
4. Sort by score descending, breaking ties with the CURATE_SEED-seeded RNG.
5. Take docs in that order until about T tokens (len(text)/4 per doc) are covered.

If the survivors cannot cover T tokens, fall back to the seeded shuffle of the whole pool.
The seed is the environment variable CURATE_SEED.
"""
import hashlib
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import T

MIN_ALPHA_FRAC = 0.7
SPAM_PATTERNS = ("click here", "coupon", "discount code", "free shipping")


def _seed():
    return int(os.environ.get("CURATE_SEED", 0))


def _est_tokens(text):
    return len(text) / 4.0


def _fallback(docs, seed):
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    return idx


def _passes_quality(text):
    s = text.strip()
    if not s:
        return False
    alpha = sum(c.isalpha() for c in s)
    if alpha / len(s) < MIN_ALPHA_FRAC:
        return False
    low = s.lower()
    if any(p in low for p in SPAM_PATTERNS):
        return False
    return True


def select(docs, target_sample):
    seed = _seed()
    rng = random.Random(seed)

    if not docs:
        return []

    # 1. Exact duplicates on stripped text (keep first occurrence).
    seen = set()
    deduped = []
    for i, d in enumerate(docs):
        h = hashlib.sha1(d.strip().encode("utf-8", "replace")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(i)

    # 2. Quality and spam filter.
    kept = [i for i in deduped if _passes_quality(docs[i])]

    # 3. TF-IDF similarity to the mean target vector.
    scores = np.zeros(len(kept))
    if kept and target_sample:
        try:
            texts_kept = [docs[i] for i in kept]
            vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
            vec.fit(list(target_sample) + texts_kept)
            X = normalize(vec.transform(texts_kept))
            centroid = normalize(np.asarray(vec.transform(target_sample).mean(axis=0)))
            scores = np.asarray(X @ centroid.T).ravel()
        except ValueError:
            # Empty vocabulary after min_df/stop-word filtering: rank by tie-break only.
            scores = np.zeros(len(kept))

    # 4. Sort by score descending; ties broken by seeded RNG.
    tiebreak = [rng.random() for _ in kept]
    ranking = sorted(range(len(kept)), key=lambda j: (-float(scores[j]), tiebreak[j]))
    order = [kept[j] for j in ranking]

    # 5. Accumulate approximate tokens until T is reached.
    total = sum(_est_tokens(docs[i]) for i in order)
    if total < T:
        return _fallback(docs, seed)

    out = []
    acc = 0.0
    for i in order:
        out.append(i)
        acc += _est_tokens(docs[i])
        if acc >= T:
            break
    return out

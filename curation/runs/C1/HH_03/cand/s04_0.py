"""Filter-then-rank curator: drop exact duplicates and junk, then order the
surviving pool by a kNN-style TF-IDF similarity score against the target sample.

Each surviving document is scored as the mean of its top-k cosine similarities
to target_sample documents (k=5). Ties are broken by a seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (no repeats).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SPAM_MARKERS = [
    "discount code", "coupon", "cheap", "buy now", "free shipping", "click here",
]
HEADER_RE = re.compile(r"^\s*=\s*[^=\n].*?[^=\n]\s*=\s*$", re.MULTILINE)
WS_RE = re.compile(r"\s+")
K_NEIGHBORS = 5


def _seed():
    return int(os.environ.get("CURATE_SEED", 0))


def _normalize(text):
    return WS_RE.sub(" ", text.lower()).strip()


def _non_ascii_share(text):
    if not text:
        return 1.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text)


def _is_junk(text):
    if _non_ascii_share(text) > 0.02:
        return True
    lowered = text.lower()
    for marker in SPAM_MARKERS:
        if marker in lowered:
            return True
    if not HEADER_RE.search(text):
        return True
    return False


def _knn_scores(doc_mat, target_mat, k=K_NEIGHBORS):
    """Mean of each survivor's top-k cosine similarities to target documents."""
    n_docs = doc_mat.shape[0]
    n_target = target_mat.shape[0]
    if n_docs == 0 or n_target == 0:
        return np.zeros(n_docs, dtype=np.float64)

    S = doc_mat @ target_mat.T  # survivors x 200
    if hasattr(S, "toarray"):
        S = S.toarray()
    S = np.asarray(S, dtype=np.float64)

    kk = min(k, n_target)
    # Indices of the kk largest entries in each row lie in the last kk columns
    # after partitioning; the values there are exactly the top-kk similarities.
    top = np.partition(S, n_target - kk, axis=1)[:, n_target - kk:]
    return top.mean(axis=1)


def select(docs, target_sample):
    seed = _seed()

    # (1) exact duplicates by hash of normalized text
    seen = set()
    survivors = []
    for i, doc in enumerate(docs):
        key = hashlib.sha1(_normalize(doc).encode("utf-8", "replace")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        survivors.append(i)

    # (2) junk filter
    survivors = [i for i in survivors if not _is_junk(docs[i])]

    if not survivors:
        # Nothing survived; fall back to the seeded shuffle of all docs so
        # training still has data to read.
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    # (3) kNN-style TF-IDF similarity to target sample
    surv_texts = [docs[i] for i in survivors]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    try:
        vec.fit(list(target_sample) + surv_texts)
        target_mat = normalize(vec.transform(target_sample))
        doc_mat = normalize(vec.transform(surv_texts))
        scores = _knn_scores(doc_mat, target_mat, K_NEIGHBORS)
    except ValueError:
        # Empty vocabulary (e.g. min_df=2 removes everything); keep survivor order.
        scores = np.zeros(len(survivors))

    # Seeded tie-breaking shuffle, then stable sort by descending score.
    order = list(range(len(survivors)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -scores[k])

    return [survivors[k] for k in order]

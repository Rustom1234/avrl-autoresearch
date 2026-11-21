"""Filter-then-rank curator: drop exact duplicates and junk, then order the
surviving pool by a background-contrastive TF-IDF score:

    score(doc) = cos(doc, target_centroid) - 0.5 * cos(doc, pool_centroid)

where target_centroid is the normalized mean of the L2-normalized target
sample vectors and pool_centroid is the normalized mean of the L2-normalized
survivor vectors.

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
BACKGROUND_WEIGHT = 0.5


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

    # (3) background-contrastive TF-IDF score
    surv_texts = [docs[i] for i in survivors]
    vec = TfidfVectorizer(
        token_pattern=r"[^\s]+",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    try:
        vec.fit(list(target_sample) + surv_texts)

        # Target centroid: normalized mean of L2-normalized target vectors.
        target_mat = normalize(vec.transform(target_sample))
        target_centroid = normalize(np.asarray(target_mat.mean(axis=0)))

        # Pool centroid: normalized mean of L2-normalized survivor vectors.
        surv_mat = normalize(vec.transform(surv_texts))
        pool_centroid = normalize(np.asarray(surv_mat.mean(axis=0)))

        target_sim = np.asarray(surv_mat @ target_centroid.T).ravel()
        pool_sim = np.asarray(surv_mat @ pool_centroid.T).ravel()
        scores = target_sim - BACKGROUND_WEIGHT * pool_sim
    except ValueError:
        # Empty vocabulary (e.g. min_df=2 removes everything); keep survivor order.
        scores = np.zeros(len(survivors))

    # Seeded tie-breaking shuffle, then stable sort by descending score.
    order = list(range(len(survivors)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -scores[k])

    return [survivors[k] for k in order]

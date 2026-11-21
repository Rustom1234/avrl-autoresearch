"""Filter-then-rank curator: drop exact duplicates and junk, order the surviving
pool by TF-IDF cosine similarity to the centroid of the target sample, then
greedily remove near-duplicates (cosine >= 0.8 to an already-kept doc) by
pushing them to the end of the list.

select(docs, target_sample) -> ordered list of indices into docs (no repeats).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import hashlib
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SPAM_MARKERS = [
    "discount code", "coupon", "cheap", "buy now", "free shipping", "click here",
]
HEADER_RE = re.compile(r"^\s*=\s*[^=\n].*?[^=\n]\s*=\s*$", re.MULTILINE)
WS_RE = re.compile(r"\s+")

NEAR_DUP_THRESHOLD = 0.8
NEAR_DUP_CHUNK = 256
TIME_BUDGET_S = 150.0  # leave headroom under the 180 s limit


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


def _near_dup_split(mat, deadline):
    """Greedy near-duplicate filter over rows of an L2-normalized sparse matrix
    already in ranked order.

    Returns (kept_positions, dropped_positions), both lists of row indices in
    ranked order. A row is dropped if its cosine similarity to any already-kept
    row is >= NEAR_DUP_THRESHOLD. If the time budget runs out, the remaining
    rows are kept without checking, which preserves ranked order.
    """
    n = mat.shape[0]
    kept = []
    dropped = []
    kept_mat = None
    kept_dirty = True

    for start in range(0, n, NEAR_DUP_CHUNK):
        if time.time() > deadline:
            kept.extend(range(start, n))
            break

        pos = list(range(start, min(start + NEAR_DUP_CHUNK, n)))
        chunk = mat[pos]

        # Similarity of each chunk row to the max over all kept rows so far.
        if kept:
            if kept_dirty:
                kept_mat = mat[kept]
                kept_dirty = False
            cross = (chunk @ kept_mat.T).max(axis=1)
            cross = np.asarray(cross).ravel()
        else:
            cross = np.zeros(len(pos))

        # Similarity among rows within this chunk, for in-chunk greedy decisions.
        self_sim = (chunk @ chunk.T).toarray()

        new_kept_local = []
        for j, p in enumerate(pos):
            if cross[j] >= NEAR_DUP_THRESHOLD:
                dropped.append(p)
                continue
            if any(self_sim[j, k] >= NEAR_DUP_THRESHOLD for k in new_kept_local):
                dropped.append(p)
                continue
            new_kept_local.append(j)
            kept.append(p)

        if new_kept_local:
            kept_dirty = True

    return kept, dropped


def select(docs, target_sample):
    t0 = time.time()
    deadline = t0 + TIME_BUDGET_S
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

    # (3) TF-IDF similarity to target centroid
    surv_texts = [docs[i] for i in survivors]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    doc_mat = None
    try:
        vec.fit(list(target_sample) + surv_texts)
        target_mat = normalize(vec.transform(target_sample))
        centroid = np.asarray(target_mat.mean(axis=0))
        centroid = normalize(centroid)
        doc_mat = normalize(vec.transform(surv_texts))
        scores = np.asarray(doc_mat @ centroid.T).ravel()
    except ValueError:
        # Empty vocabulary (e.g. min_df=2 removes everything); keep survivor order.
        scores = np.zeros(len(survivors))
        doc_mat = None

    # Seeded tie-breaking shuffle, then stable sort by descending score.
    order = list(range(len(survivors)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -scores[k])

    ranked = [survivors[k] for k in order]

    # (4) greedy near-duplicate filter over the ranked survivors.
    # Near-duplicates are appended after the kept docs, in ranked order, so
    # every surviving index still appears in the output.
    if doc_mat is None:
        return ranked

    ranked_mat = doc_mat[order]
    kept_pos, dropped_pos = _near_dup_split(ranked_mat, deadline)

    return [ranked[p] for p in kept_pos] + [ranked[p] for p in dropped_pos]

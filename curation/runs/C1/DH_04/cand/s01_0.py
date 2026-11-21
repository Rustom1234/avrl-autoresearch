"""Target-similarity curator.

select(docs, target_sample) -> ordered list of indices into docs.

1. Score every pool doc with a char n-gram TF-IDF + logistic regression model
   (target_sample = positives, seeded random pool sample = negatives).
2. Drop junk: exact duplicates, very short docs, docs with a low share of
   alphabetic/space characters, and highly compressible (templated spam) docs.
3. Take the best-scoring surviving docs until about 1.0 T tokens are collected.
4. Shuffle the selection with the CURATE_SEED seed.
"""
import hashlib
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


def _score(docs, target_sample, seed):
    """decision_function score of every pool doc (higher = more target-like)."""
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
    X = vec.fit_transform(texts)
    X_t = X[:n_t]
    X_p = X[n_t:]

    neg = random.Random(seed).sample(range(n), min(N_NEG, n))
    from scipy.sparse import vstack  # scipy ships with scikit-learn

    X_train = vstack([X_t, X_p[neg]]).tocsr()
    y = np.concatenate([np.ones(n_t, dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y)
    return clf.decision_function(X_p)


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

    scores = _score(docs, target_sample, seed)

    # Rank surviving docs by score, best first (stable for determinism).
    keep_arr = np.asarray(keep, dtype=np.int64)
    order = keep_arr[np.argsort(-scores[keep_arr], kind="stable")]

    # Take top-scoring docs until ~1.0 T tokens. Docs above the 5th-percentile
    # target score come first by construction; if they hold fewer than T tokens
    # the budget is filled with the next best-scoring docs instead of cycling.
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total = 0
    pos = 0
    while pos < len(order) and total < T:
        batch = order[pos:pos + TOKEN_BATCH]
        pos += TOKEN_BATCH
        toks = enc.encode_ordinary_batch([docs[int(i)] for i in batch])
        for i, t in zip(batch, toks):
            selected.append(int(i))
            total += len(t)
            if total >= T:
                break

    random.Random(seed).shuffle(selected)
    return selected

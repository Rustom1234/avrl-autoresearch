"""Domain-classifier curator: dedup, score pool docs with a logistic-regression
target-domain classifier, and take the top-scoring docs up to T tokens.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import T


def _seed():
    return int(os.environ.get("CURATE_SEED", 0))


def _seeded_shuffle(idx, seed):
    idx = list(idx)
    random.Random(seed).shuffle(idx)
    return idx


def _dedup(docs):
    """Indices of first occurrence of each exact duplicate (hash of stripped text)."""
    seen = set()
    keep = []
    for i, d in enumerate(docs):
        h = hashlib.sha1(d.strip().encode("utf-8", "replace")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        keep.append(i)
    return keep


def _scores(docs, target_sample, uniq, seed):
    """Score each unique pool doc with P(target domain). Returns array over len(docs)."""
    n_t = len(target_sample)

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    X_all = vec.fit_transform(list(target_sample) + list(docs))

    # Negatives: seeded random sample of unique pool docs.
    rng = random.Random(seed)
    n_neg = min(200, len(uniq))
    neg_idx = rng.sample(uniq, n_neg)
    neg_set = set(neg_idx)

    train_rows = list(range(n_t)) + [n_t + i for i in neg_idx]
    y = np.array([1] * n_t + [0] * n_neg)
    X_train = X_all[train_rows]

    scores = np.zeros(len(docs), dtype=float)

    # 5-fold cross-fitted predictions for the sampled negatives,
    # so no negative is scored by a model that was trained on it.
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X_train, y):
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        m.fit(X_train[tr], y[tr])
        oof[te] = m.predict_proba(X_train[te])[:, 1]
    for k, i in enumerate(neg_idx):
        scores[i] = oof[n_t + k]

    # Full model scores every other unique pool doc.
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y)
    rest = [i for i in uniq if i not in neg_set]
    if rest:
        p = clf.predict_proba(X_all[[n_t + i for i in rest]])[:, 1]
        for i, s in zip(rest, p):
            scores[i] = s

    return scores


def select(docs, target_sample):
    seed = _seed()
    uniq = _dedup(docs)
    fallback = _seeded_shuffle(uniq, seed)

    if len(target_sample) < 5 or len(uniq) < 10:
        return fallback

    try:
        scores = _scores(docs, target_sample, uniq, seed)
    except Exception:
        return fallback

    # Tie-break order among equal scores with a seeded random key.
    rng = random.Random(seed)
    tie = {i: rng.random() for i in uniq}
    order = sorted(uniq, key=lambda i: (-scores[i], tie[i]))

    out, total = [], 0.0
    for i in order:
        out.append(i)
        total += len(docs[i]) / 4
        if total >= T:
            break

    if total < T:
        return fallback
    return out

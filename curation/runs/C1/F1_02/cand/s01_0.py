"""Curator: target-similarity filter.

1. Drop docs under 200 chars and exact duplicates (hash of whitespace-normalised text).
2. Score every remaining doc with a char n-gram TF-IDF + logistic regression
   (target_sample = positives, seeded random pool docs = negatives), plus a small
   bonus for WikiText-style formatting.
3. Keep top-scoring docs until T tokens are reached or the score drops below 0.5.
4. Return the kept indices in a CURATE_SEED-seeded shuffled order.
"""
import hashlib
import os
import random

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

MIN_CHARS = 200
MAX_CHARS = 3000          # chars per doc used for scoring
CHAR_BUDGET = 30_000_000  # total chars vectorised, to stay inside the time limit
N_NEG = 3000
THRESHOLD = 0.5
MAX_BONUS = 0.25
TOK_BATCH = 1000


def _wiki_like(text):
    return text.lstrip().startswith("= ") and (" @-@ " in text or " , " in text)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # (2) length filter + exact dedup, keeping the first copy
    seen = set()
    valid = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        valid.append(i)

    if not valid:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if not targets:
        out = list(valid)
        rng.shuffle(out)
        return out

    # (1) char n-gram TF-IDF + logistic regression
    n = len(valid)
    cap = max(500, min(MAX_CHARS, CHAR_BUDGET // max(1, n)))
    pool_texts = [docs[i][:cap] for i in valid]
    target_texts = [t[:cap] for t in targets]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=50000,
        dtype=np.float32,
    )
    X = vec.fit_transform(pool_texts + target_texts)
    X_pool = X[:n]
    X_pos = X[n:]

    neg_rows = rng.sample(range(n), min(N_NEG, n))
    X_neg = X_pool[neg_rows]

    from scipy.sparse import vstack  # scipy ships with scikit-learn

    X_train = vstack([X_pos, X_neg]).tocsr()
    y_train = np.concatenate(
        [np.ones(X_pos.shape[0], dtype=np.int32), np.zeros(X_neg.shape[0], dtype=np.int32)]
    )
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_pool)[:, 1].astype(np.float64)

    # WikiText formatting bonus, scaled by how often the target sample shows it
    target_frac = sum(1 for t in targets if _wiki_like(t)) / len(targets)
    bonus = MAX_BONUS * target_frac
    if bonus > 0:
        flags = np.fromiter((_wiki_like(docs[i]) for i in valid), dtype=bool, count=n)
        scores = scores + bonus * flags

    # (3) rank, then keep until T tokens or score < threshold
    order = np.argsort(-scores, kind="stable")
    candidates = [int(r) for r in order if scores[r] >= THRESHOLD]
    if not candidates:
        # Nothing passed the threshold: fall back to pure ranking
        candidates = [int(r) for r in order]

    enc = tiktoken.get_encoding("gpt2")
    kept = []
    total = 0
    done = False
    for start in range(0, len(candidates), TOK_BATCH):
        batch = candidates[start:start + TOK_BATCH]
        toks = enc.encode_ordinary_batch([docs[valid[r]] for r in batch])
        for r, tk in zip(batch, toks):
            kept.append(valid[r])
            total += len(tk)
            if total >= T:
                done = True
                break
        if done:
            break

    # (4) seeded shuffle
    rng.shuffle(kept)
    return kept

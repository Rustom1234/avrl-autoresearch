"""Curator: keep only clean, target-like documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

1. Word-order score h: fraction of a document's first ~400 adjacent lowercase
   word bigrams that occur in the target sample.
2. Domain score p (first pass): TF-IDF (word 1-2 grams) + logistic regression,
   with the target sample as positives and 3000 seeded-random pool documents as
   negatives.
3. Self-training round: the first-pass negatives are contaminated with
   target-like documents, so the classifier is refit once.
   Positives: the target sample plus up to 3000 seeded-random pool documents
   that pass the first-pass filter (h > h_thr and p >= 0.5, after dedup).
   Negatives: 4000 seeded-random pool documents that confidently fail
   (h <= h_thr or p < 0.2). This gives the second-pass score p2.
4. Keep documents with h above a threshold calibrated leave-one-out on the
   target sample and p2 >= 0.5, dropping short documents and (near-)duplicates.

If the kept set holds fewer than T tokens it is returned as is (the trainer
cycles it). If it holds more, the highest-p2 documents up to T tokens are kept.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

N_BIGRAMS = 400        # bigrams scored per document
HEAD_CHARS = 6000      # characters scanned to get the first ~400 words
CLF_CHARS = 3000       # characters per document seen by the classifier
N_NEG = 3000           # random pool documents used as first-pass negatives
N_POS2 = 3000          # max pool documents added as second-pass positives
N_NEG2 = 4000          # confidently failing pool documents used as second-pass negatives
MIN_CHARS = 200        # shorter documents are dropped
DEDUP_CHARS = 300      # prefix length for the near-duplicate key
P_MIN = 0.5
P_NEG_MAX = 0.2        # p below this counts as a confident fail


def _head_bigrams(text):
    """The first N_BIGRAMS adjacent lowercase word bigrams of text."""
    w = text[:HEAD_CHARS].lower().split()[: N_BIGRAMS + 1]
    return list(zip(w, w[1:]))


def _all_bigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:]))


def _count_tokens(texts):
    enc = tiktoken.get_encoding("gpt2")
    out = []
    for s in range(0, len(texts), 2000):
        chunk = texts[s:s + 2000]
        try:
            out.extend(len(t) for t in enc.encode_ordinary_batch(chunk, num_threads=8))
        except Exception:
            out.extend(len(enc.encode_ordinary(t)) for t in chunk)
    return out


def _fit_and_score(pos_txt, neg_txt, docs, seed):
    """Fit TF-IDF + logistic regression and return P(target) for every doc."""
    n = len(docs)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_train = vec.fit_transform(pos_txt + neg_txt)
    y_train = np.array([1] * len(pos_txt) + [0] * len(neg_txt))
    # Balanced weights: the two classes can differ a lot in size.
    clf = LogisticRegression(
        C=10.0, max_iter=1000, class_weight="balanced", random_state=seed
    )
    clf.fit(X_train, y_train)

    p = np.zeros(n, dtype=np.float64)
    step = 5000
    for s in range(0, n, step):
        X = vec.transform([d[:CLF_CHARS] for d in docs[s:s + step]])
        p[s:s + step] = clf.predict_proba(X)[:, 1]
    return p


def _passing(docs, h, h_thr, p):
    """Indices of documents that pass the filter, after MIN_CHARS and dedup."""
    kept = []
    seen_full = set()
    seen_head = set()
    for i in range(len(docs)):
        d = docs[i]
        if len(d) < MIN_CHARS:
            continue
        if not (h[i] > h_thr and p[i] >= P_MIN):
            continue
        # Dedup only among documents that pass, so a damaged copy cannot
        # shadow its clean twin.
        full_key = hash(d)
        head_key = " ".join(d[:DEDUP_CHARS].lower().split())
        if full_key in seen_full or head_key in seen_head:
            continue
        seen_full.add(full_key)
        seen_head.add(head_key)
        kept.append(i)
    return kept


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def fallback():
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    if n == 0:
        return []
    if not target_sample:
        return fallback()

    # ---------- 1. word-order score ----------
    target_sets = [_all_bigrams(t) for t in target_sample]
    df = Counter()
    for s in target_sets:
        df.update(s)
    target_bigrams = set(df)

    h = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(docs):
        bg = _head_bigrams(d)
        if bg:
            hits = 0
            for b in bg:
                if b in target_bigrams:
                    hits += 1
            h[i] = hits / len(bg)

    # Leave-one-out h of each target document against the other 199.
    # A bigram of document i is in "the others" iff it occurs in >= 2 target docs
    # (document i itself always accounts for one).
    loo = []
    for t in target_sample:
        bg = _head_bigrams(t)
        if not bg:
            continue
        loo.append(sum(1 for b in bg if df[b] >= 2) / len(bg))
    h_thr = 0.9 * float(np.percentile(loo, 5)) if loo else 0.0

    # ---------- 2. domain score, first pass ----------
    target_txt = [t[:CLF_CHARS] for t in target_sample]
    neg_idx = rng.sample(range(n), min(N_NEG, n))
    neg_txt = [docs[i][:CLF_CHARS] for i in neg_idx]
    p = _fit_and_score(target_txt, neg_txt, docs, seed)

    # ---------- 3. self-training round ----------
    # Positives: target sample + pool documents that pass the current filter.
    # Negatives: pool documents that confidently fail, so no target-like
    # document sits in the negatives.
    pass1 = _passing(docs, h, h_thr, p)
    fail1 = [i for i in range(n) if h[i] <= h_thr or p[i] < P_NEG_MAX]

    pos2_idx = rng.sample(pass1, min(N_POS2, len(pass1)))
    neg2_idx = rng.sample(fail1, min(N_NEG2, len(fail1)))

    if neg2_idx:
        pos2_txt = target_txt + [docs[i][:CLF_CHARS] for i in pos2_idx]
        neg2_txt = [docs[i][:CLF_CHARS] for i in neg2_idx]
        p2 = _fit_and_score(pos2_txt, neg2_txt, docs, seed)
    else:
        # No confident negatives to train on: keep the first-pass score.
        p2 = p

    # ---------- 4. select ----------
    kept = _passing(docs, h, h_thr, p2)

    if not kept:
        return fallback()

    ntok = _count_tokens([docs[i] for i in kept])
    total = sum(ntok)

    if total > T:
        order = sorted(range(len(kept)), key=lambda j: (-p2[kept[j]], kept[j]))
        chosen = []
        acc = 0
        for j in order:
            chosen.append(kept[j])
            acc += ntok[j]
            if acc >= T:
                break
        kept = sorted(chosen)

    rng.shuffle(kept)
    return kept

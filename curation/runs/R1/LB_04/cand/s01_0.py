"""Curator: target-similarity filter + dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Char n-gram TF-IDF + logistic regression (target vs. random pool subsample),
     followed by one round of self-training on the pool.
  2. Drop exact / near duplicates (hash of normalized first 500 chars).
  3. Drop junk (too short, low alphabetic fraction, low in-target-vocabulary fraction).
  4. Keep the top-scoring documents until ~1.0 * T tokens.
  5. Seeded shuffle of the kept set.
"""
import os
import random
import re
import hashlib
from collections import Counter

import numpy as np

from common import T

HEAD_CHARS = 3000          # chars of each doc used for featurization
POOL_CHAR_BUDGET = 60_000_000  # deterministic cap on total chars vectorized (time safety)
NEG_SUBSAMPLE = 4000
MAX_SELF_POS = 6000        # caps on self-training set size (time safety)
MAX_SELF_NEG = 12000
TOKEN_BUDGET = 1.0 * T

_WORD_RE = re.compile(r"[^\W\d_]+")
_ALPHA_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_WS_RE = re.compile(r"\s+")


def _seed():
    try:
        return int(os.environ.get("CURATE_SEED", 0))
    except ValueError:
        return 0


def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    rng = random.Random(seed)

    # Deterministic per-doc char cap so vectorizing the pool stays within budget.
    cap = int(max(600, min(HEAD_CHARS, POOL_CHAR_BUDGET // max(n, 1))))

    tgt_heads = [t[:HEAD_CHARS] for t in target_sample]
    sub_idx = rng.sample(range(n), min(NEG_SUBSAMPLE, n))
    sub_heads = [docs[i][:HEAD_CHARS] for i in sub_idx]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=200_000,
        dtype=np.float32,
    )
    X_fit = vec.fit_transform(tgt_heads + sub_heads)
    y_fit = np.array([1] * len(tgt_heads) + [0] * len(sub_heads))
    X_tgt = X_fit[: len(tgt_heads)]

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
    clf.fit(X_fit, y_fit)

    X_pool = vec.transform([d[:cap] for d in docs])
    scores = clf.predict_proba(X_pool)[:, 1]

    # One round of self-training: top 15% positives, bottom 50% negatives.
    try:
        from scipy.sparse import vstack

        order = np.argsort(-scores, kind="stable")
        n_pos = int(0.15 * n)
        n_neg = int(0.50 * n)
        if n_pos >= 10 and n_neg >= 10:
            pos = order[:n_pos].tolist()
            neg = order[n - n_neg:].tolist()
            if len(pos) > MAX_SELF_POS:
                pos = rng.sample(pos, MAX_SELF_POS)
            if len(neg) > MAX_SELF_NEG:
                neg = rng.sample(neg, MAX_SELF_NEG)
            X2 = vstack([X_tgt, X_pool[pos], X_pool[neg]]).tocsr()
            y2 = np.array([1] * (X_tgt.shape[0] + len(pos)) + [0] * len(neg))
            clf2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
            clf2.fit(X2, y2)
            scores = clf2.predict_proba(X_pool)[:, 1]
    except Exception:
        pass  # keep first-round scores

    return np.asarray(scores, dtype=np.float64)


def _target_vocab(target_sample):
    cnt = Counter()
    for t in target_sample:
        cnt.update(_WORD_RE.findall(t.lower()))
    return {w for w, c in cnt.items() if c >= 2}


def _is_junk(doc, vocab):
    if len(doc.strip()) < 200:
        return True
    head = doc[:5000]
    if not head:
        return True
    alpha_space = len(_ALPHA_SPACE_RE.findall(head))
    if alpha_space / len(head) < 0.7:
        return True
    if vocab:
        words = _WORD_RE.findall(head.lower())
        if not words:
            return True
        in_vocab = sum(1 for w in words if w in vocab)
        if in_vocab / len(words) < 0.5:
            return True
    return False


def _dedup_key(doc):
    norm = _WS_RE.sub(" ", doc[:2000].lower()).strip()[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _token_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            return [len(x) for x in enc.encode_ordinary_batch(texts)]

        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]

        return count


def select(docs, target_sample):
    seed = _seed()
    n = len(docs)
    if n == 0:
        return []

    # (1) target-similarity scores
    try:
        scores = _score(docs, target_sample, seed)
    except Exception:
        scores = np.zeros(n, dtype=np.float64)

    # (2) + (3) dedup and junk filters
    vocab = _target_vocab(target_sample)
    seen = set()
    keep_mask = np.zeros(n, dtype=bool)
    for i, d in enumerate(docs):
        if _is_junk(d, vocab):
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        keep_mask[i] = True

    cand = np.nonzero(keep_mask)[0]
    if len(cand) == 0:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    # (4) best-scored docs first, fill up to ~T tokens
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    count = _token_counter()
    kept = []
    total = 0
    chunk = 2000
    for s in range(0, len(cand), chunk):
        part = cand[s:s + chunk]
        counts = count([docs[int(i)] for i in part])
        for i, c in zip(part, counts):
            kept.append(int(i))
            total += c
            if total >= TOKEN_BUDGET:
                break
        if total >= TOKEN_BUDGET:
            break

    # (5) seeded shuffle
    random.Random(seed).shuffle(kept)
    return kept

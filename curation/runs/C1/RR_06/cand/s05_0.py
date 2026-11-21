"""Curator: dedup, word-bigram domain classifier with self-training, shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Drop duplicates (hash of the whitespace-normalised, lowercased first 300
     characters; first occurrence kept) and docs under 50 tokens.
  2. Hash case-preserved word uni+bigrams of the first 3000 characters.
  3. Round 1: logistic regression, target chunks vs. 4000 random pool docs.
  4. Round 2: refit with target chunks + top 15% of the pool as positives and
     the bottom 50% as negatives; rescore the pool.
  5. Take docs in descending round-2 score until T tokens are reached.
  6. Shuffle the selection with the seed.
"""
import hashlib
import os
import random

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

_HEAD_CHARS = 3000      # characters of each doc used for features
_DEDUP_CHARS = 300      # characters of normalised prefix used for dedup
_MIN_TOKENS = 50        # docs shorter than this are dropped
_N_NEG_ROUND1 = 4000    # random pool docs used as round-1 negatives
_TOP_FRAC = 0.15        # round-2 pseudo-positives
_BOTTOM_FRAC = 0.50     # round-2 pseudo-negatives


def _token_counts(docs):
    enc = tiktoken.get_encoding("gpt2")
    counts = []
    batch = 2048
    threads = max(1, min(8, os.cpu_count() or 1))
    for start in range(0, len(docs), batch):
        chunk = docs[start:start + batch]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=threads)
            counts.extend(len(e) for e in encoded)
        except Exception:
            for d in chunk:
                try:
                    counts.append(len(enc.encode_ordinary(d)))
                except Exception:
                    counts.append(max(1, len(d) // 4))
    return counts


def _dedup_key(text):
    # Normalise a generous prefix, then keep the first 300 normalised chars.
    norm = " ".join(text[:4 * _DEDUP_CHARS].lower().split())[:_DEDUP_CHARS]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _fit(X, y, seed):
    clf = LogisticRegression(
        C=10, max_iter=200, class_weight="balanced", random_state=seed
    )
    clf.fit(X, y)
    return clf


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # ---- 1. Dedup (first occurrence wins) and minimum length ----------------
    seen = set()
    unique = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or not d.strip():
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        unique.append(i)

    counts = _token_counts([docs[i] for i in unique])
    kept = [i for i, c in zip(unique, counts) if c >= _MIN_TOKENS]
    tok = {i: c for i, c in zip(unique, counts)}

    if not kept:
        # Degenerate pool: fall back to whatever is non-empty.
        fallback = unique if unique else list(range(len(docs)))
        random.Random(seed).shuffle(fallback)
        return fallback

    # ---- 2. Features --------------------------------------------------------
    target_chunks = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        for s in range(0, len(t), _HEAD_CHARS):
            chunk = t[s:s + _HEAD_CHARS]
            if chunk.strip():
                target_chunks.append(chunk)

    n_pos = len(target_chunks)
    n_pool = len(kept)

    vec = HashingVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        lowercase=False,
        ngram_range=(1, 2),
        n_features=2 ** 20,
        norm="l2",
        alternate_sign=False,
    )
    # Target chunks first, then pool docs, in one matrix (rows sliced below).
    X_all = vec.transform(
        target_chunks + [docs[i][:_HEAD_CHARS] for i in kept]
    ).tocsr()
    pos_rows = np.arange(n_pos, dtype=np.int64)
    pool_rows = np.arange(n_pos, n_pos + n_pool, dtype=np.int64)
    X_pool = X_all[pool_rows]

    scores = np.zeros(n_pool, dtype=np.float64)

    if n_pos > 0 and n_pool > 1:
        # ---- 3. Round 1: target chunks vs. random pool docs -----------------
        neg_local = rng.sample(range(n_pool), min(_N_NEG_ROUND1, n_pool))
        neg_rows = pool_rows[np.asarray(neg_local, dtype=np.int64)]
        rows = np.concatenate([pos_rows, neg_rows])
        y = np.concatenate([
            np.ones(n_pos, dtype=np.int8),
            np.zeros(len(neg_rows), dtype=np.int8),
        ])
        clf = _fit(X_all[rows], y, seed)
        scores = clf.decision_function(X_pool)

        # ---- 4. Round 2: self-training --------------------------------------
        order1 = np.argsort(-scores, kind="stable")
        n_top = max(1, int(round(_TOP_FRAC * n_pool)))
        n_bot = max(1, int(round(_BOTTOM_FRAC * n_pool)))
        n_bot = min(n_bot, n_pool - n_top)
        if n_bot >= 1:
            top_rows = pool_rows[order1[:n_top]]
            bot_rows = pool_rows[order1[n_pool - n_bot:]]
            rows = np.concatenate([pos_rows, top_rows, bot_rows])
            y = np.concatenate([
                np.ones(n_pos + len(top_rows), dtype=np.int8),
                np.zeros(len(bot_rows), dtype=np.int8),
            ])
            clf = _fit(X_all[rows], y, seed)
            scores = clf.decision_function(X_pool)

    # ---- 5. Selection: best-scoring docs until the budget is filled ---------
    order = np.argsort(-scores, kind="stable")
    selected = []
    total = 0
    for j in order:
        i = kept[int(j)]
        selected.append(i)
        total += tok[i]
        if total >= T:
            break

    # ---- 6. Seeded shuffle so weaker docs are not all read last -------------
    random.Random(seed).shuffle(selected)
    return selected

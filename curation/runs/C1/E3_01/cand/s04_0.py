"""Curator: dedupe + word-bigram PU classifier with one self-training round.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import hashlib

import numpy as np
import tiktoken
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

HEAD = 3000          # characters of each document used for features
PREFIX = 200         # characters used for the near-duplicate prefix check
N_NEG_ROUND1 = 4000  # unlabeled pool documents used as negatives in round 1
TOP_FRAC = 0.15      # round-2 positives: top fraction of the pool
BOTTOM_FRAC = 0.50   # round-2 negatives: bottom fraction of the pool


def _dedupe(docs):
    """First occurrence of each normalised text; also drop repeated 200-char prefixes."""
    seen_hash, seen_prefix, kept = set(), set(), []
    for i, d in enumerate(docs):
        norm = " ".join(d.lower().split())
        if not norm:
            continue  # empty / whitespace-only documents carry no tokens
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen_hash:
            continue
        p = d[:PREFIX]
        if p in seen_prefix:
            continue
        seen_hash.add(h)
        seen_prefix.add(p)
        kept.append(i)
    return kept


def _chunks(text, size=HEAD):
    out = [text[i:i + size] for i in range(0, len(text), size)]
    return [c for c in out if c.strip()]


def _fit_score(X_pos, X_neg, X_all):
    X = vstack([X_pos, X_neg]).tocsr()
    y = np.concatenate([np.ones(X_pos.shape[0], dtype=np.int8),
                        np.zeros(X_neg.shape[0], dtype=np.int8)])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    clf.fit(X, y)
    return clf.predict_proba(X_all)[:, 1]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe.
    kept = _dedupe(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))
    heads = [docs[i][:HEAD] for i in kept]

    # 2. Features: hashed word uni+bigrams over whitespace tokens.
    vec = HashingVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                            lowercase=False, n_features=2 ** 20, binary=True, norm="l2")
    X_pool = vec.transform(heads).tocsr()
    target_chunks = []
    for t in target_sample:
        target_chunks.extend(_chunks(t))
    X_tgt = vec.transform(target_chunks).tocsr()

    # 3. Round 1: target chunks (positive) vs random pool documents (unlabeled).
    neg_idx = np.array(sorted(rng.sample(range(n), min(N_NEG_ROUND1, n))))
    p1 = _fit_score(X_tgt, X_pool[neg_idx], X_pool)

    # 4. Round 2: self-training on confident pool documents.
    order1 = np.argsort(-p1, kind="stable")
    n_top = max(1, int(round(TOP_FRAC * n)))
    n_bot = max(1, int(round(BOTTOM_FRAC * n)))
    n_bot = min(n_bot, n - n_top) if n - n_top > 0 else 0
    top_idx = np.sort(order1[:n_top])
    if n_bot > 0:
        bot_idx = np.sort(order1[n - n_bot:])
        p2 = _fit_score(vstack([X_pool[top_idx], X_tgt]).tocsr(), X_pool[bot_idx], X_pool)
    else:
        p2 = p1

    # 5. Word-order check: fraction of a document's bigrams seen in the target sample.
    target_bigrams = set()
    for t in target_sample:
        w = t.split()
        target_bigrams.update(zip(w, w[1:]))
    frac = np.zeros(n)
    for j, h in enumerate(heads):
        w = h.split()
        m = len(w) - 1
        if m > 0:
            hit = 0
            for b in zip(w, w[1:]):
                if b in target_bigrams:
                    hit += 1
            frac[j] = hit / m
    threshold = 0.5 * float(np.median(frac[top_idx]))
    flagged = frac < threshold

    # Final ranking: round-2 probability, with flagged documents pushed to the bottom.
    score = np.where(flagged, p2 - 2.0, p2)
    ranked = np.argsort(-score, kind="stable")

    # 6. Selection: walk the ranking until T tokens are collected. Everything with
    # score > 0.5 comes first; if that is short of T the next most similar documents
    # fill the gap, and if it exceeds T only the highest-scoring T tokens are kept.
    enc = tiktoken.get_encoding("gpt2")
    selected, total = [], 0
    BATCH = 2000
    for s in range(0, n, BATCH):
        batch = ranked[s:s + BATCH]
        toks = enc.encode_ordinary_batch([docs[kept[j]] for j in batch])
        for j, tk in zip(batch, toks):
            selected.append(kept[j])
            total += len(tk)
            if total >= T:
                break
        if total >= T:
            break

    # 7. Order: seeded shuffle.
    selected = [int(i) for i in selected]
    random.Random(seed).shuffle(selected)
    return selected

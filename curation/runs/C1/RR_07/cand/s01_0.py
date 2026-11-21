"""Curator: rank pool documents by vocabulary overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Steps:
  1. Count the tokens of every pool document with tiktoken ("gpt2").
  2. Score every document by its IDF-weighted word overlap with the vocabulary
     of target_sample (share of the document's word mass that is made of words
     the target domain also uses).
  3. Drop exact duplicates, pick the highest-overlap documents first until the
     training budget of T tokens is filled, then shuffle the picked set with the
     seed from CURATE_SEED so training does not see a sorted stream.
"""
import os
import random
import re

import numpy as np

from common import T

_TOKEN_PATTERN = r"[a-z0-9']+"
_WS = re.compile(r"\s+")


def _token_counts(docs):
    """Number of gpt2 tokens in each document."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = np.empty(len(docs), dtype=np.int64)
        chunk = 2000
        for start in range(0, len(docs), chunk):
            batch = docs[start:start + chunk]
            try:
                encoded = enc.encode_ordinary_batch(batch, num_threads=8)
                lens = [len(e) for e in encoded]
            except Exception:
                lens = []
                for d in batch:
                    try:
                        lens.append(len(enc.encode_ordinary(d)))
                    except Exception:
                        lens.append(max(1, len(d) // 4))
            counts[start:start + len(batch)] = lens
        return counts
    except Exception:
        return np.array([max(1, len(d) // 4) for d in docs], dtype=np.int64)


def _overlap_scores(docs, target_sample):
    """IDF-weighted share of each document's words found in the target vocabulary."""
    from sklearn.feature_extraction.text import CountVectorizer

    n = len(docs)
    vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN_PATTERN, dtype=np.float32)
    try:
        X = vec.fit_transform(docs).tocsr()
    except ValueError:
        return np.zeros(n, dtype=np.float64)

    # Target vocabulary.
    word_re = re.compile(_TOKEN_PATTERN)
    target_vocab = set()
    for t in target_sample:
        target_vocab.update(word_re.findall(t.lower()))

    vocab = vec.vocabulary_
    mask = np.zeros(X.shape[1], dtype=np.float64)
    for w in target_vocab:
        j = vocab.get(w)
        if j is not None:
            mask[j] = 1.0

    # Document frequency -> idf, so rare words (junk, other domains) weigh more.
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0

    # Sublinear term frequency: repeated words do not dominate.
    X.data = np.log1p(X.data)

    num = np.asarray(X @ (idf * mask)).ravel()
    den = np.asarray(X @ idf).ravel()
    # Small shrinkage so very short documents cannot get a perfect score.
    return num / (den + 30.0)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    counts = _token_counts(docs)
    scores = _overlap_scores(docs, target_sample)

    # Exact duplicates (after whitespace/case normalisation): keep the first copy.
    seen = set()
    unique = np.ones(n, dtype=bool)
    for i, d in enumerate(docs):
        key = hash(_WS.sub(" ", d).strip().lower())
        if key in seen:
            unique[i] = False
        else:
            seen.add(key)

    usable = unique & (counts > 0)
    order = np.argsort(-scores, kind="stable")

    selected = []
    total = 0
    for i in order:
        if not usable[i]:
            continue
        selected.append(int(i))
        total += int(counts[i])
        if total >= T:
            break

    if not selected:
        selected = list(range(n))

    random.Random(seed).shuffle(selected)
    return selected

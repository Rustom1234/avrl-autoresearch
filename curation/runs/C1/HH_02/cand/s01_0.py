"""Curator: exact dedup and junk filter, then rank by TF-IDF cosine similarity to the target domain, keeping docs until T tokens.

select(docs, target_sample) -> ordered list of indices into docs. Each index appears at most once.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import hashlib

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import T

SPAM_RE = re.compile(r"click here|\border\b[^.\n]{0,60}\bonline\b", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def _dedup_key(text):
    norm = WS_RE.sub(" ", text).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _is_junk(text):
    if not text.strip():
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii / len(text) > 0.05:
        return True
    if SPAM_RE.search(text):
        return True
    return False


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # 1) Exact dedup (whitespace-normalized, lowercased) and junk filter.
    seen = set()
    keep = []  # indices into docs
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        key = _dedup_key(d)
        if key in seen:
            continue
        seen.add(key)
        if _is_junk(d):
            continue
        keep.append(i)

    if not keep:
        return []

    texts = [docs[i] for i in keep]

    # 2) Score each remaining doc by cosine similarity to the mean target vector.
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        vec.fit(target_sample)
        target_mean = np.asarray(vec.transform(target_sample).mean(axis=0)).ravel()
        target_norm = np.linalg.norm(target_mean)
        target_vec = target_mean / (target_norm if target_norm > 0 else 1.0)
        X = normalize(vec.transform(texts))
        scores = np.asarray(X @ target_vec).ravel()
    except ValueError:
        # e.g. empty vocabulary from the target sample; fall back to neutral scores
        scores = np.zeros(len(texts))

    # Seeded shuffle first, then a stable sort by descending score,
    # so ties are broken reproducibly by CURATE_SEED.
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    order.sort(key=lambda j: -scores[j])

    # 3) Walk the ranked list, counting gpt2 tokens, until T is reached.
    enc = tiktoken.get_encoding("gpt2")
    out = []
    total = 0
    for j in order:
        if total >= T:
            break
        n = len(enc.encode_ordinary(texts[j]))
        if n == 0:
            continue
        out.append(keep[j])
        total += n

    return out

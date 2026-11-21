"""Curator: rank pool documents by TF-IDF weighted overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Scoring:
  TF(w)  = frequency of word w in the target sample.
  IDF(w) = log((1 + N) / (1 + df(w))), where df(w) is the number of pool docs
           containing w and N is the pool size (the pool is the background corpus).
  score(doc) = sum over target words w present in doc of TF(w) * IDF(w),
               divided by log1p(number of unique words in doc).
"""
import math
import os
import random
import re
from collections import Counter

import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")


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


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Token count for each document in the pool.
    tok_counts = _token_counts(docs)

    # Term frequencies from the target sample.
    target_tf = Counter()
    for t in target_sample:
        target_tf.update(_WORD_RE.findall(t.lower()))
    total_tf = float(sum(target_tf.values())) or 1.0
    target_vocab = set(target_tf)

    # Single pass over the pool: unique-word count, shared target words,
    # and document frequency of each target word.
    n_docs = len(docs)
    df = Counter()
    shared_per_doc = []
    unique_counts = []
    for d in docs:
        w_set = set(_WORD_RE.findall(d.lower()))
        unique_counts.append(len(w_set))
        shared = w_set & target_vocab
        df.update(shared)
        shared_per_doc.append(tuple(shared))

    # TF-IDF weight for each target word (pool as background corpus).
    weight = {}
    for w, c in target_tf.items():
        idf = math.log((1.0 + n_docs) / (1.0 + df.get(w, 0)))
        weight[w] = (c / total_tf) * idf

    # Score each document.
    scores = []
    for shared, n_unique in zip(shared_per_doc, unique_counts):
        if n_unique == 0 or not shared:
            scores.append(0.0)
            continue
        s = 0.0
        for w in shared:
            s += weight[w]
        scores.append(s / math.log1p(n_unique))

    # Seeded random tie-breaker, then sort with highest score first.
    tiebreak = [rng.random() for _ in docs]
    order = sorted(range(n_docs), key=lambda i: (-scores[i], tiebreak[i]))

    # Keep the most relevant documents up to the T-token budget.
    selected = []
    total = 0
    for i in order:
        if tok_counts[i] <= 0:
            continue
        selected.append(i)
        total += tok_counts[i]
        if total >= T:
            break

    return selected if selected else order

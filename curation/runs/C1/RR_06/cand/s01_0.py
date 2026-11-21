"""Curator: rank pool documents by Jaccard similarity to the target vocabulary.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")


def _words(text):
    return set(_WORD_RE.findall(text.lower()))


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

    # Set of words in the target sample.
    target_words = set()
    for t in target_sample:
        target_words |= _words(t)
    n_target = len(target_words)

    # Jaccard similarity between each document's word set and the target word set.
    scores = []
    for d in docs:
        w = _words(d)
        if not w:
            scores.append(0.0)
            continue
        inter = len(w & target_words)
        union = len(w) + n_target - inter
        scores.append(inter / union if union else 0.0)

    # Seeded random tie-breaker, then sort with highest similarity first.
    tiebreak = [rng.random() for _ in docs]
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], tiebreak[i]))

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

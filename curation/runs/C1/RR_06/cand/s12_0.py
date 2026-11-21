"""Curator: rank pool documents by 'shared-density' of target-frequency weights.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Score: for each document, sum the target word frequencies of its unique words
that appear in the target sample, divided by the number of those shared unique
words (not by the document's total unique word count). Documents sharing no
words score 0.
"""
import os
import random
import re
from collections import Counter

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

    # Target word frequencies (TF-like weighting).
    target_word_freqs = Counter()
    for t in target_sample:
        target_word_freqs.update(_WORD_RE.findall(t.lower()))

    # Shared-density score: sum of weights of shared unique words divided by
    # the number of shared unique words.
    get = target_word_freqs.get
    scores = []
    for d in docs:
        shared_sum = 0.0
        shared_n = 0
        for w in _words(d):
            f = get(w, 0)
            if f > 0:
                shared_sum += f
                shared_n += 1
        scores.append(shared_sum / shared_n if shared_n else 0.0)

    # Seeded random tie-breaker, then sort with highest score first.
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

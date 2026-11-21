"""Curator: rank pool documents by target-frequency weighted shared word count.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import math
import os
import random
import re
from collections import Counter

import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")
_MIN_TOKENS = 100


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

    # Target word frequencies, computed over the whole target sample corpus.
    target_word_freqs = Counter()
    for t in target_sample:
        target_word_freqs.update(_WORD_RE.findall(t.lower()))

    # Weighted overlap score:
    # sum of target frequencies of shared words / log1p(unique words in doc).
    scores = []
    for d in docs:
        w_set = _words(d)
        if not w_set:
            scores.append(0.0)
            continue

        shared_score = 0.0
        for w in w_set:
            shared_score += target_word_freqs.get(w, 0)

        scores.append(shared_score / math.log1p(len(w_set)))

    # Explicitly filter out documents with fewer than _MIN_TOKENS tokens.
    eligible = [i for i in range(len(docs)) if tok_counts[i] >= _MIN_TOKENS]

    # Seeded random tie-breaker, then sort with highest score first.
    tiebreak = [rng.random() for _ in docs]
    order = sorted(eligible, key=lambda i: (-scores[i], tiebreak[i]))

    # Keep the most relevant documents up to the T-token budget.
    selected = []
    total = 0
    for i in order:
        selected.append(i)
        total += tok_counts[i]
        if total >= T:
            break

    if selected:
        return selected

    # Fallback: nothing passed the length filter; rank everything non-empty.
    fallback = sorted(
        (i for i in range(len(docs)) if tok_counts[i] > 0),
        key=lambda i: (-scores[i], tiebreak[i]),
    )
    return fallback if fallback else list(range(len(docs)))

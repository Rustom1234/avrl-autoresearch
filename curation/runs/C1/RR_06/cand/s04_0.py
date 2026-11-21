"""Curator: rank pool documents by a weighted word-overlap score against the target vocabulary.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Score = (number of unique words shared with the target vocabulary)
        / log1p(number of unique words in the document)
Documents with fewer than 100 tokens are filtered out before scoring.
"""
import math
import os
import random
import re

import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")

MIN_TOKENS = 100


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

    # Seeded random tie-breaker (drawn for every doc to keep it deterministic per seed).
    tiebreak = [rng.random() for _ in docs]

    # Filter out short documents before scoring.
    candidates = [i for i in range(len(docs)) if tok_counts[i] >= MIN_TOKENS]

    # Weighted overlap score: shared words / log1p(unique words in document).
    scores = {}
    for i in candidates:
        w = _words(docs[i])
        if not w:
            scores[i] = 0.0
            continue
        inter = len(w & target_words)
        scores[i] = inter / math.log1p(len(w))

    # Sort with highest score first.
    order = sorted(candidates, key=lambda i: (-scores[i], tiebreak[i]))

    # Keep the most relevant documents up to the T-token budget.
    selected = []
    total = 0
    for i in order:
        selected.append(i)
        total += tok_counts[i]
        if total >= T:
            break

    return selected

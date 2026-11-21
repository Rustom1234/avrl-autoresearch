"""Curator: score documents by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its total GPT-2 tokens (counting
repeats) that occur in the target sample's vocabulary, weighted by a length
factor that favors documents long enough to be representative (>= 100 tokens).
Exact duplicates are dropped, the best-scoring documents are taken until T
tokens are covered, and the selection is shuffled with the seed from
CURATE_SEED.
"""
import os, random

import numpy as np
import tiktoken

from common import T

MIN_TOKENS = 16      # documents shorter than this are dropped
FULL_LENGTH = 100    # documents at least this long get full weight


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary as a boolean lookup table over GPT-2 token ids.
    in_target = np.zeros(enc.n_vocab, dtype=bool)
    for toks in enc.encode_ordinary_batch(list(target_sample)):
        if toks:
            in_target[np.asarray(toks, dtype=np.int64)] = True

    n = len(docs)
    scores = [0.0] * n
    lengths = [0] * n
    seen = set()
    keep = []

    chunk = 2000
    for start in range(0, n, chunk):
        batch = docs[start:start + chunk]
        encoded = enc.encode_ordinary_batch(batch)
        for off, toks in enumerate(encoded):
            i = start + off
            length = len(toks)
            lengths[i] = length
            if length < MIN_TOKENS:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            arr = np.asarray(toks, dtype=np.int64)
            # fraction of all token occurrences that are in the target vocab
            common = int(np.count_nonzero(in_target[arr]))
            overlap = common / length
            # length weight: ramps up to 1.0 at FULL_LENGTH tokens
            weight = min(1.0, length / FULL_LENGTH)
            scores[i] = overlap * weight
            keep.append(i)
        del encoded

    if not keep:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Random tie-break, then sort by score descending.
    rng.shuffle(keep)
    keep.sort(key=lambda i: scores[i], reverse=True)

    selected = []
    total = 0
    for i in keep:
        selected.append(i)
        total += lengths[i]
        if total >= T:
            break

    rng.shuffle(selected)
    return selected

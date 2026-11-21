"""Curator: score documents by bigram overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its unique GPT-2 token bigrams that
also occur in the target sample. Exact duplicates are dropped, the best-scoring
documents are taken until T tokens are covered, and the selection is shuffled
with the seed from CURATE_SEED.
"""
import os, random

import numpy as np
import tiktoken

from common import T

_V = 50257  # GPT-2 vocabulary size, used to pack a bigram into one int64


def _bigrams(toks):
    """Sorted unique bigram codes of a token list."""
    arr = np.asarray(toks, dtype=np.int64)
    if arr.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.unique(arr[:-1] * _V + arr[1:])


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target bigram set: sorted unique bigram codes seen in the target sample.
    parts = [_bigrams(toks) for toks in enc.encode_ordinary_batch(list(target_sample))]
    parts = [p for p in parts if p.size]
    if parts:
        target_bigrams = np.unique(np.concatenate(parts))
    else:
        target_bigrams = np.empty(0, dtype=np.int64)
    n_target = target_bigrams.size

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
            lengths[i] = len(toks)
            if len(toks) < 16:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            bg = _bigrams(toks)
            if n_target and bg.size:
                pos = np.searchsorted(target_bigrams, bg)
                pos[pos == n_target] = n_target - 1
                common = int(np.count_nonzero(target_bigrams[pos] == bg))
                # fraction of the doc's unique bigrams that are common with the target
                scores[i] = common / bg.size
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

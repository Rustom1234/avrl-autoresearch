"""Curator: score documents by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document with more than MIN_COMMON unique GPT-2 tokens in common with the
target sample is scored by the fraction of its unique tokens that also occur in
the target sample. Documents with very low overlap get a small baseline score
that grows with their length, so longer low-overlap documents are still
preferred over short ones. Exact duplicates are dropped, the best-scoring
documents are taken until T tokens are covered, and the selection is shuffled
with the seed from CURATE_SEED.
"""
import os, random

import tiktoken

from common import T

# A document needs more than this many unique common tokens to get an overlap score.
MIN_COMMON = 10
# Low-overlap documents get a baseline in [0, BASELINE_MAX], rising with length
# and saturating at BASELINE_LEN tokens.
BASELINE_MAX = 0.25
BASELINE_LEN = 1024


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary: unique tokens seen in the target sample.
    target_vocab = set()
    for toks in enc.encode_ordinary_batch(list(target_sample)):
        target_vocab.update(toks)

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
            uniq = set(toks)
            common = len(uniq & target_vocab)
            if common > MIN_COMMON:
                # fraction of the doc's unique tokens that are common with the target
                scores[i] = common / len(uniq)
            else:
                # very low overlap: baseline score based on length
                scores[i] = BASELINE_MAX * min(1.0, len(toks) / BASELINE_LEN)
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

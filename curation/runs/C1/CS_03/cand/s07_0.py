"""Curator: score documents by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its unique GPT-2 tokens that also
occur in the target sample, weighted by a mild length bonus:

    score = (common / len(set(toks))) * (1 + log1p(len(toks)))

Documents with fewer than MIN_COMMON tokens in common with the target sample
are filtered out as noise. Exact duplicates are dropped, the best-scoring
documents are taken until T tokens are covered, and the selection is shuffled
with the seed from CURATE_SEED.
"""
import os, random
from math import log1p

import tiktoken

from common import T

MIN_COMMON = 2   # minimum number of unique tokens shared with the target
MIN_TOKENS = 16  # documents shorter than this are too short to be useful


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
            ntok = len(toks)
            lengths[i] = ntok
            if ntok < MIN_TOKENS:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            uniq = set(toks)
            common = len(uniq & target_vocab)
            if common < MIN_COMMON:
                continue  # too little overlap with the target: noise
            # overlap ratio, with a mild preference for longer documents
            scores[i] = (common / len(uniq)) * (1.0 + log1p(ntok))
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

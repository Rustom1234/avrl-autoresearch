"""Curator: score documents by Jaccard similarity with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by a Jaccard-like similarity between its set of unique
GPT-2 tokens and the set of tokens occurring in the target sample:

    score = |doc_tokens & target_vocab| / |doc_tokens | target_vocab|

Exact duplicates are dropped, the best-scoring documents are taken until T
tokens are covered, and the selection is shuffled with the seed from
CURATE_SEED.
"""
import os, random

import tiktoken

from common import T


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary: unique tokens seen in the target sample.
    target_vocab = set()
    for toks in enc.encode_ordinary_batch(list(target_sample)):
        target_vocab.update(toks)
    n_target = len(target_vocab)

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
            # Jaccard: |A & B| / |A | B|, with |A | B| = |A| + |B| - |A & B|
            union = len(uniq) + n_target - common
            scores[i] = common / union if union else 0.0
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

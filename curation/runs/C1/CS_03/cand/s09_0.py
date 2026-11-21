"""Curator: score documents by a TF-IDF inspired overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored as

    (common_tokens_count / document_total_tokens) * (1 / log(1 + len(target_vocab)))

where common_tokens_count is the number of the document's unique GPT-2 tokens
that also occur in the target sample, and document_total_tokens is the
document's full token length. Documents with fewer than 50 tokens are dropped,
exact duplicates are dropped, the best-scoring documents are taken until T
tokens are covered, and the selection is shuffled with the seed from
CURATE_SEED.
"""
import math
import os
import random

import tiktoken

from common import T

MIN_TOKENS = 50


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary: unique tokens seen in the target sample.
    target_vocab = set()
    for toks in enc.encode_ordinary_batch(list(target_sample)):
        target_vocab.update(toks)

    # Global weighting factor: 1 / log(1 + |target_vocab|).
    denom = math.log(1 + len(target_vocab))
    vocab_weight = 1.0 / denom if denom > 0 else 0.0

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
            total_tokens = len(toks)
            lengths[i] = total_tokens
            if total_tokens < MIN_TOKENS:
                continue  # too short to be contextually rich
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            common = len(set(toks) & target_vocab)
            # (common_tokens_count / document_total_tokens) * (1 / log(1 + |V_target|))
            scores[i] = (common / total_tokens) * vocab_weight
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

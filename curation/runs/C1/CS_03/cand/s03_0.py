"""Curator: score documents by length-weighted density of target tokens.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by
    score = (common / total_tokens) * log(1 + total_tokens)
where total_tokens is the document's GPT-2 token count and common is the number
of those tokens that also occur in the target sample. Exact duplicates are
dropped, the best-scoring documents are taken until T tokens are covered, and
the selection is shuffled with the seed from CURATE_SEED.
"""
import math, os, random

import tiktoken

from common import T


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary: unique tokens seen in the target sample.
    target_vocab = set()
    for toks in enc.encode_ordinary_batch(list(target_sample)):
        target_vocab.update(toks)
    in_target = target_vocab.__contains__

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
            if total_tokens < 16:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            # number of tokens in the doc that are in the target vocabulary
            common = sum(map(in_target, toks))
            # density of target-relevant tokens, weighted by log volume
            scores[i] = (common / total_tokens) * math.log(1 + total_tokens)
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

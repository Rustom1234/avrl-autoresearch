"""Curator: score documents by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its unique GPT-2 tokens that also
occur in the target sample (common_tokens / len(set(toks_in_doc))). Exact
duplicates are dropped and the best-scoring documents are taken, in descending
score order, until T tokens are covered.

If the usable documents run out before T tokens are covered, the remaining
buffer is filled by over-sampling: the top 20% of the high-scoring documents
(those with score > 0.1) are repeated, best first, until T tokens are reached.
The final selection is shuffled with the seed from CURATE_SEED.
"""
import math, os, random

import tiktoken

from common import T

HIGH_SCORE = 0.1     # a document is "high-scoring" above this score
TOP_FRACTION = 0.2   # fraction of high-scoring documents that get over-sampled


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
            # fraction of the doc's unique tokens that are common with the target
            scores[i] = common / len(uniq)
            keep.append(i)
        del encoded

    if not keep:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Random tie-break, then sort by score descending.
    rng.shuffle(keep)
    keep.sort(key=lambda i: scores[i], reverse=True)

    # Base selection: best-scoring documents until T tokens are covered.
    selected = []
    total = 0
    for i in keep:
        selected.append(i)
        total += lengths[i]
        if total >= T:
            break

    # Over-sampling: fill any remaining buffer by repeating the top 20% of the
    # high-scoring documents (score > 0.1), best first.
    if total < T:
        high = [i for i in keep if scores[i] > HIGH_SCORE]  # already sorted
        if high:
            top = high[:max(1, math.ceil(TOP_FRACTION * len(high)))]
            while total < T:
                for i in top:
                    selected.append(i)
                    total += lengths[i]
                    if total >= T:
                        break

    rng.shuffle(selected)
    return selected

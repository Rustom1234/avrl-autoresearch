"""Curator: score documents by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its unique GPT-2 tokens that also
occur in the target sample (common_tokens_count / len(set(toks))). Exact
duplicates are dropped. The selection is a greedy fill: the best available
document is added repeatedly until T tokens are covered, drawing only from the
pool of 'high-quality' documents (score > 0.05), so low-quality noise never
fills the final tokens. If the high-quality pool holds fewer than T tokens, the
list is left short and the trainer cycles it. The selection is shuffled with
the seed from CURATE_SEED.
"""
import os, random

import tiktoken

from common import T

QUALITY_THRESHOLD = 0.05
MIN_TOKENS = 16


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
            if len(toks) < MIN_TOKENS:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            uniq = set(toks)
            common_tokens_count = len(uniq & target_vocab)
            scores[i] = common_tokens_count / len(uniq)
            keep.append(i)
        del encoded

    if not keep:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Random tie-break, then sort by score descending.
    rng.shuffle(keep)
    keep.sort(key=lambda i: scores[i], reverse=True)

    # Pool of high-quality documents; only these may be used for filling.
    pool = [i for i in keep if scores[i] > QUALITY_THRESHOLD]
    if not pool:
        # Nothing clears the threshold: fall back to the full ranked list.
        pool = keep

    # Greedy fill: repeatedly add the best available document until T is reached.
    selected = []
    total = 0
    for i in pool:
        if total >= T:
            break
        selected.append(i)
        total += lengths[i]

    # If the pool ran out before T, the list stays short and the trainer
    # cycles through it, rather than topping up with low-quality documents.
    rng.shuffle(selected)
    return selected

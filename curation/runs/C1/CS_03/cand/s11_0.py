"""Curator: multi-pass selection by token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored by the fraction of its unique GPT-2 tokens that also
occur in the target sample. Exact duplicates and very short documents are dropped.

Pass 1: documents with score > HIGH_Q are taken in full (best first; if they
        alone exceed T tokens, the best-scoring ones are kept until T is covered).
Pass 2: the remaining budget is filled by score-weighted sampling without
        replacement from the rest of the pool, until T tokens are reached.

The final selection is shuffled with the seed from CURATE_SEED.
"""
import os, random

import numpy as np
import tiktoken

from common import T

HIGH_Q = 0.15


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
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

    # ---- Pass 1: high-quality documents, taken in full (best first). ----
    high = [i for i in keep if scores[i] > HIGH_Q]
    rest = [i for i in keep if scores[i] <= HIGH_Q]

    rng.shuffle(high)  # random tie-break
    high.sort(key=lambda i: scores[i], reverse=True)

    selected = []
    total = 0
    for i in high:
        if total >= T:
            break
        selected.append(i)
        total += lengths[i]

    # ---- Pass 2: score-weighted sampling without replacement for the remainder. ----
    if total < T and rest:
        cand = np.array([i for i in rest if scores[i] > 0.0], dtype=np.int64)
        if cand.size:
            w = np.array([scores[i] for i in cand], dtype=np.float64)
            # Efraimidis-Spirakis: key = log(u) / w, take the largest keys first.
            u = np_rng.random(cand.size)
            u = np.clip(u, 1e-12, 1.0)
            keys = np.log(u) / w
            order = np.argsort(-keys, kind="stable")
            for j in order:
                if total >= T:
                    break
                i = int(cand[j])
                selected.append(i)
                total += lengths[i]

        # Last resort: zero-score documents, in random order.
        if total < T:
            zeros = [i for i in rest if scores[i] <= 0.0]
            rng.shuffle(zeros)
            for i in zeros:
                if total >= T:
                    break
                selected.append(i)
                total += lengths[i]

    if not selected:
        selected = list(keep)

    rng.shuffle(selected)
    return selected

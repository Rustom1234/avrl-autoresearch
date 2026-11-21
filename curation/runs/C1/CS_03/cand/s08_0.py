"""Curator: score documents by weighted token overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Each document is scored as

    (common_tokens_count / total_tokens_in_doc) * log10(1 + len(set(toks_in_doc)))

where common_tokens_count is the number of the document's GPT-2 tokens
(occurrences) that also occur in the target sample. Documents with at least
MIN_TOKENS tokens are prioritised over shorter ones, so the training buffer is
not filled with short, high-density fragments that lack context. Exact
duplicates are dropped, the best documents are taken until T tokens are
covered, and the selection is shuffled with the seed from CURATE_SEED.
"""
import math
import os
import random

import numpy as np
import tiktoken

from common import T

MIN_TOKENS = 100   # documents at least this long are prioritised
HARD_MIN_TOKENS = 16  # documents shorter than this are dropped outright


def select(docs, target_sample):
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    enc = tiktoken.get_encoding("gpt2")

    # Target vocabulary as a boolean lookup table over token ids.
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
            total = len(toks)
            lengths[i] = total
            if total < HARD_MIN_TOKENS:
                continue  # too short to be useful
            h = hash(batch[off])
            if h in seen:
                continue  # exact duplicate
            seen.add(h)
            arr = np.asarray(toks, dtype=np.int64)
            common = int(np.count_nonzero(in_target[arr]))
            n_uniq = len(set(toks))
            scores[i] = (common / total) * math.log10(1 + n_uniq)
            keep.append(i)
        del encoded

    if not keep:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Random tie-break, then sort: long-enough documents first, each group by
    # score descending. Short documents are only used if the long ones do not
    # cover T tokens.
    rng.shuffle(keep)
    keep.sort(key=lambda i: (lengths[i] >= MIN_TOKENS, scores[i]), reverse=True)

    selected = []
    total = 0
    for i in keep:
        selected.append(i)
        total += lengths[i]
        if total >= T:
            break

    rng.shuffle(selected)
    return selected

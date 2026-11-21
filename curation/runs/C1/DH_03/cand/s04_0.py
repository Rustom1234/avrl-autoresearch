"""Curator: keep clean, target-like documents, deduplicate, repeat with fresh shuffles.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

The output is built explicitly: the clean, deduplicated set is repeated, with a
new shuffle for each pass, until it covers T tokens plus one pass of margin.
Rejected (low-scoring) docs are only used if the clean set is below a small
safety floor of FLOOR_FRACTION * T tokens.
"""
import os
import random
import re
import hashlib
from collections import Counter

import numpy as np
import tiktoken

from common import T

TOP_BIGRAMS = 5000
MIN_CHARS = 200
PREFIX_CHARS = 300
FLOOR_FRACTION = 0.08
TOKEN_BATCH = 2000
FILL_BATCH = 256

_ODD_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-]")


def _bigram_set(texts, k=TOP_BIGRAMS):
    counts = Counter()
    for t in texts:
        w = t.split()
        counts.update(zip(w, w[1:]))
    return {b for b, _ in counts.most_common(k)}


def _score(text, bigrams):
    w = text.split()
    n = len(w) - 1
    if n <= 0:
        return 0.0
    return sum(map(bigrams.__contains__, zip(w, w[1:]))) / n


def _nonascii_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (n - len(text.encode("ascii", "ignore"))) / n


def _odd_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return len(_ODD_RE.findall(text)) / n


def _hashes(text):
    norm = " ".join(text.lower().split())
    full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
    prefix = hashlib.md5(norm[:PREFIX_CHARS].encode("utf-8", "ignore")).digest()
    return full, prefix


def _token_counts(enc, docs, indices):
    """Exact gpt2 token counts for docs[i], i in indices (batched to bound memory)."""
    counts = []
    for start in range(0, len(indices), TOKEN_BATCH):
        chunk = indices[start:start + TOKEN_BATCH]
        encoded = enc.encode_ordinary_batch([docs[i] for i in chunk])
        counts.extend(len(e) for e in encoded)
    return counts


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target = list(target_sample)

    # 1. Bigram fluency: held-out scores for the target sample.
    half = len(target) // 2
    part_a, part_b = target[:half], target[half:]
    set_a = _bigram_set(part_a)
    set_b = _bigram_set(part_b)
    heldout = [_score(t, set_a) for t in part_b] + [_score(t, set_b) for t in part_a]

    final_set = _bigram_set(target)

    # 2. Threshold.
    threshold = float(np.percentile(heldout, 5)) * 0.95 if heldout else 0.0

    # 3. Junk guard limits from the target sample.
    max_nonascii = max((_nonascii_share(t) for t in target), default=0.0) * 1.5
    max_odd = max((_odd_share(t) for t in target), default=0.0) * 1.5

    n = len(docs)
    scores = np.zeros(n, dtype=np.float64)
    passed_guard = np.zeros(n, dtype=bool)

    for i, d in enumerate(docs):
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_share(d) > max_nonascii:
            continue
        if _odd_share(d) > max_odd:
            continue
        passed_guard[i] = True
        scores[i] = _score(d, final_set)

    # 4. Deduplicate the docs that pass the threshold (first occurrence wins).
    seen_full = set()
    seen_prefix = set()
    kept = []
    rejected = []
    for i in range(n):
        if not passed_guard[i]:
            continue
        if scores[i] >= threshold:
            full, prefix = _hashes(docs[i])
            if full in seen_full or prefix in seen_prefix:
                continue
            seen_full.add(full)
            seen_prefix.add(prefix)
            kept.append(i)
        else:
            rejected.append(i)

    # 5. Exact token counts for the kept docs only.
    enc = tiktoken.get_encoding("gpt2")
    kept_tokens = sum(_token_counts(enc, docs, kept))

    # 6. Safety floor: only if the clean set is very small, add the best
    #    rejected (deduplicated) docs until the floor is reached.
    floor_tokens = FLOOR_FRACTION * T
    if kept_tokens < floor_tokens and rejected:
        rejected.sort(key=lambda i: (-scores[i], i))
        pos = 0
        while kept_tokens < floor_tokens and pos < len(rejected):
            batch = []
            while pos < len(rejected) and len(batch) < FILL_BATCH:
                i = rejected[pos]
                pos += 1
                full, prefix = _hashes(docs[i])
                if full in seen_full or prefix in seen_prefix:
                    continue
                seen_full.add(full)
                seen_prefix.add(prefix)
                batch.append(i)
            if not batch:
                continue
            for i, c in zip(batch, _token_counts(enc, docs, batch)):
                if kept_tokens >= floor_tokens:
                    break
                kept.append(i)
                kept_tokens += c

    if not kept:
        kept = list(range(n))
        kept_tokens = sum(_token_counts(enc, docs, kept))

    # 7. Build the output: repeated passes over the kept set, each with a fresh
    #    shuffle from the same RNG, until T tokens plus one pass of margin.
    rng = random.Random(seed)
    if kept_tokens <= 0:
        order = list(kept)
        rng.shuffle(order)
        return order

    target_tokens = T + kept_tokens
    out = []
    total = 0
    while total < target_tokens:
        order = list(kept)
        rng.shuffle(order)
        out.extend(order)
        total += kept_tokens
    return out

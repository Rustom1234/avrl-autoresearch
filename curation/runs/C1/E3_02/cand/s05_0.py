"""Curator: order-sensitive target-bigram scoring, dedup, fill ~1.02*T tokens.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import hashlib
import os
import random

import tiktoken

from common import T

MIN_TOKENS = 50          # minimum whitespace tokens per doc
SCORE_PREFIX = 2000      # whitespace tokens used for scoring
TITLE_BONUS = 0.05       # bonus for wikitext title line
FILL_FACTOR = 1.02       # take docs until cumulative tokens reach this * T
BATCH_SIZE = 256         # docs per tiktoken batch


def _hash(s):
    return hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=12).digest()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Target whitespace-token bigram set (case kept).
    target_bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        target_bigrams.update(zip(toks, toks[1:]))

    # 2-4. Score, length filter, dedup.
    seen_full = set()
    seen_prefix = set()
    scored = []  # (score, index)

    for i, doc in enumerate(docs):
        toks = doc.split()

        # Length filter.
        if len(toks) < MIN_TOKENS:
            continue

        # Dedup on lowercased, whitespace-collapsed text and its first 300 chars.
        norm = " ".join(toks).lower()
        h_full = _hash(norm)
        h_prefix = _hash(norm[:300])
        if h_full in seen_full or h_prefix in seen_prefix:
            continue
        seen_full.add(h_full)
        seen_prefix.add(h_prefix)

        # Order-sensitive score: fraction of bigrams found in the target set.
        head = toks[:SCORE_PREFIX]
        n_bigrams = len(head) - 1
        hits = 0
        for bg in zip(head, head[1:]):
            if bg in target_bigrams:
                hits += 1
        score = hits / n_bigrams if n_bigrams > 0 else 0.0

        if doc.startswith("= "):
            score += TITLE_BONUS

        scored.append((score, i))

    # 5. Sort by score descending (index as deterministic tie-break).
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Take from the top until cumulative gpt2 token count reaches ~1.02*T.
    enc = tiktoken.get_encoding("gpt2")
    budget = int(FILL_FACTOR * T)
    selected = []
    total = 0
    pos = 0
    n = len(scored)
    while pos < n and total < budget:
        batch = [idx for _, idx in scored[pos:pos + BATCH_SIZE]]
        pos += BATCH_SIZE
        encoded = enc.encode_ordinary_batch([docs[idx] for idx in batch])
        for idx, ids in zip(batch, encoded):
            selected.append(idx)
            total += len(ids)
            if total >= budget:
                break

    if not selected:
        # Degenerate pool: fall back to everything so training has data.
        selected = list(range(len(docs)))

    # 6. Shuffle with the seeded rng.
    rng.shuffle(selected)
    return selected

"""Curator: bigram-overlap quality score, exact dedup, cut at ~1.02*T tokens, seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import time
import random
import hashlib

from common import T

MAX_SCORE_TOKENS = 2000   # score each doc on its first N whitespace tokens
MIN_TOKENS = 50           # drop docs shorter than this (whitespace tokens)
BUDGET_FACTOR = 1.02      # keep about 1.02 * T GPT-2 tokens
TIME_BUDGET_EXACT = 110.0 # seconds after which token counting falls back to len/4
BATCH = 512


def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Target unigram vocabulary and adjacent-bigram set (case preserved).
    vocab = set()
    bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        vocab.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    # Score, filter and deduplicate the pool.
    seen = set()
    scored = []
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        head = toks[:MAX_SCORE_TOKENS]
        n = len(head)
        u = sum(1 for t in head if t in vocab) / n
        if n > 1:
            b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)
        else:
            b = 0.0

        score = b + 0.25 * u
        if _has_title_line(text):
            score += 0.1
        scored.append((score, i))
    del seen

    if not scored:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Best first; index as a deterministic tie-break.
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    # Take from the top until about 1.02*T GPT-2 tokens.
    budget = BUDGET_FACTOR * T
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0
    pos = 0
    done = False
    while pos < len(ranked) and not done:
        batch_idx = ranked[pos:pos + BATCH]
        pos += BATCH

        lengths = None
        if enc is not None and (time.time() - t_start) < TIME_BUDGET_EXACT:
            try:
                encoded = enc.encode_ordinary_batch([docs[i] for i in batch_idx])
                lengths = [len(e) for e in encoded]
            except Exception:
                lengths = None
        if lengths is None:
            lengths = [len(docs[i]) / 4.0 for i in batch_idx]

        for i, n_tok in zip(batch_idx, lengths):
            selected.append(i)
            total += n_tok
            if total >= budget:
                done = True
                break

    # Uniform mix rather than best-to-worst order.
    rng.shuffle(selected)
    return selected

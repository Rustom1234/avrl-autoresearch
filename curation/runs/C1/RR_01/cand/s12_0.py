"""Curator: relevance x coherence score, spam filter, exact dedup, cut at ~1.02*T tokens, seeded shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stages:
  1) coherence = unique bigrams / total bigrams over the first 1000 whitespace tokens
  2) relevance = b + 0.5 * u + 0.1 * title
  3) spam filter: coherence < 0.2, or text contains 'http' or 'click here'
  4) score = relevance * coherence for surviving docs
  5) take top-scored docs until ~1.02*T tokens; if under budget, fill with the
     highest-relevance non-spam docs not yet selected (e.g. short docs)
  6) global seeded shuffle
"""
import os
import time
import random
import hashlib

from common import T

MAX_SCORE_TOKENS = 2000    # relevance is scored on the first N whitespace tokens
COHERENCE_TOKENS = 1000    # coherence is scored on the first N whitespace tokens
MIN_TOKENS = 50            # main stage drops docs shorter than this (whitespace tokens)
FILL_MIN_TOKENS = 5        # fill stage may use shorter docs, but not near-empty ones
MIN_COHERENCE = 0.2        # below this a doc is treated as spam / repetitive junk
BUDGET_FACTOR = 1.02       # keep about 1.02 * T GPT-2 tokens
TIME_BUDGET_EXACT = 110.0  # seconds after which token counting falls back to len/4
BATCH = 512


def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")


def _is_spam_text(text):
    low = text.lower()
    return ("http" in low) or ("click here" in low)


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
    main = []   # (relevance * coherence, relevance, index) for non-spam docs of normal length
    fill = []   # (relevance, index) for non-spam docs that are too short for the main stage
    for i, text in enumerate(docs):
        toks = text.split()
        n_all = len(toks)
        if n_all < FILL_MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        # Stage 3a: keyword spam.
        if _is_spam_text(text):
            continue

        # Stage 1: coherence on the first COHERENCE_TOKENS tokens.
        chead = toks[:COHERENCE_TOKENS]
        n_cb = len(chead) - 1
        if n_cb > 0:
            coherence = len(set(zip(chead, chead[1:]))) / n_cb
        else:
            coherence = 0.0

        # Stage 3b: repetition spam.
        if coherence < MIN_COHERENCE:
            continue

        # Stage 2: relevance.
        head = toks[:MAX_SCORE_TOKENS]
        n = len(head)
        u = sum(1 for t in head if t in vocab) / n
        if n > 1:
            b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)
        else:
            b = 0.0
        relevance = b + 0.5 * u
        if _has_title_line(text):
            relevance += 0.1

        if n_all >= MIN_TOKENS:
            # Stage 4: final score.
            main.append((relevance * coherence, relevance, i))
        else:
            fill.append((relevance, i))
    del seen

    if not main and not fill:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Best first; index as a deterministic tie-break.
    main.sort(key=lambda x: (-x[0], x[2]))
    ranked_main = [i for _, _, i in main]
    fill.sort(key=lambda x: (-x[0], x[1]))
    ranked_fill = [i for _, i in fill]

    # Stage 5: take from the top until about 1.02*T GPT-2 tokens.
    budget = BUDGET_FACTOR * T
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0.0

    def take(ranked, total):
        pos = 0
        while pos < len(ranked) and total < budget:
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
                    break
        return total

    total = take(ranked_main, total)
    if total < budget:
        # Under budget: fill with the highest-relevance non-spam docs not yet selected.
        total = take(ranked_fill, total)

    if not selected:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    # Stage 6: uniform mix rather than best-to-worst order.
    rng.shuffle(selected)
    return selected
